"""Cluster class-entropy temperature on trained-expert embeddings (post-hoc).

No training. Uses the *trained experts'* backbone embeddings (default: the
single best expert on val NLL, or the mean of all experts) to build a 2D UMAP
+ KMeans structure on the pooled train+val features, then assigns each cluster
a temperature proportional to how mixed its class distribution is:

    H_k     = class entropy of the pooled train+val labels in cluster k
    Hn_k    = H_k normalized to [0, 1]
    T_k     = T_global * (1 + alpha * (Hn_k - mean(Hn_k))), clipped
    p(x)    = softmax(avg_logits(x) / T_cluster(x))

A test sample landing in a class-mixed "middle" cluster gets a larger T (more
uncertain); one in a class-pure cluster gets a smaller T. Temperature is
applied to the *averaged* logits of the ensemble.

Everything expensive is cached under <run-dir>/posthoc/, so re-running with
different temperature parameters (--alpha-grid, --entropy, --temp-clip-*)
is instant; only --force-* recomputes.

Usage:
  python cluster_entropy_temp.py --run-dir <run_dir>                          # best expert, K=10
  python cluster_entropy_temp.py --run-dir <run_dir> --embed-source average
  python cluster_entropy_temp.py --run-dir <run_dir> --alpha-grid "0 .5 1 2" --temp-clip-max 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_dinov3_ensemble as M
from student.data import IWildCamChallengeDataset
from student.metrics import compute_all_metrics
from student.predict import write_submission
from student.train import fit_temperature

DEFAULT_SPLITS = ["test_public", "test_private"]
MIN_CLUSTER = 20  # min val points per cluster for it to get its own T


def softmax(x: np.ndarray, T: float | np.ndarray = 1.0) -> np.ndarray:
    T = np.asarray(T)
    if T.ndim == 1:
        T = T[:, None]
    x = x / T
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def nll_np(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(i) for i in x]
    return x


def cluster_entropy(counts: np.ndarray, measure: str) -> float:
    """Normalized class-entropy of a cluster's pooled label counts."""
    from scipy.stats import entropy as shannon

    p = counts.astype(np.float64)
    p = p / p.sum()
    if measure == "gini":
        return float(1.0 - (p ** 2).sum()) / (1.0 - 1.0 / len(p))
    H = shannon(p) if len(p) > 1 and (p > 0).sum() > 1 else 0.0
    return float(H) / float(np.log(len(p))) if len(p) > 1 else 0.0


def load_experts(out_dir: Path, num_classes: int, cfg: dict, device) -> list:
    ckpts = sorted(out_dir.glob("experts/expert_seed*.pt"),
                   key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not ckpts:
        raise FileNotFoundError(f"no expert_seed*.pt in {out_dir/'experts'}")
    models = []
    for ckpt in ckpts:
        sd = torch.load(ckpt, map_location=device)
        m = M.DinoV3LoraExpert(
            num_classes, backbone_id=cfg["backbone"],
            lora_r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            lora_last_layers=int(cfg["lora_last_layers"]),
        ).to(device)
        m.load_state_dict(sd["state_dict"])
        m.eval()
        models.append(m)
    return models


def cached_embeddings(model, loader, device, amp, path: Path, force: bool) -> np.ndarray:
    if path.exists() and not force:
        arr = np.load(path)["emb"]
        print(f"  loaded cached embeddings {arr.shape} ({path.name})")
        return arr
    t0 = time.time()
    arr = M.collect_embeddings(model, loader, device, amp)
    np.savez_compressed(path, emb=arr)
    print(f"  computed embeddings {arr.shape} in {time.time() - t0:.0f}s -> {path.name}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="override the data_root recorded in config.json")
    parser.add_argument("--embed-source", choices=["best", "average"], default="best",
                        help="embedding source: single best expert on val NLL, or mean of all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--umap-components", type=int, default=2)
    parser.add_argument("--alpha-grid", type=str, default="0 0.25 0.5 0.75 1 1.5",
                        help="candidate alpha values (space-separated)")
    parser.add_argument("--entropy", choices=["shannon", "gini"], default="shannon")
    parser.add_argument("--temp-clip-min", type=float, default=0.3)
    parser.add_argument("--temp-clip-max", type=float, default=3.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--method", choices=["auto", "cluster_entropy_T", "temp_scaled_average", "plain_average"],
                        default="auto")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--force-pipeline", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args()

    out_dir = Path(args.run_dir)
    cfg = json.loads((out_dir / "config.json").read_text())
    ph = out_dir / "posthoc"
    ph.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = args.data_root if args.data_root is not None else cfg["data_root"]
    splits = args.splits or cfg.get("splits", DEFAULT_SPLITS)
    amp = bool(cfg.get("amp_bf16", False))
    alphas = [float(a) for a in args.alpha_grid.split()]

    tf = M.eval_transform(int(cfg["img_size"]))
    val_ds = IWildCamChallengeDataset(data_root, "val", tf)
    train_ds = IWildCamChallengeDataset(data_root, "train", tf)
    val_labels = np.asarray(val_ds.labels, dtype=np.int64)
    train_labels = np.asarray(train_ds.labels, dtype=np.int64)
    num_classes = val_ds.num_classes
    print(f"train {len(train_ds)} / val {len(val_ds)} | {num_classes} classes | "
          f"run {out_dir.name} | embed-source={args.embed_source} K={args.k}")

    models = load_experts(out_dir, num_classes, cfg, device)
    seeds = [int(re.search(r"\d+", p.stem).group())
             for p in sorted(out_dir.glob("experts/expert_seed*.pt"),
                             key=lambda p: int(re.search(r"\d+", p.stem).group()))]
    print(f"loaded {len(models)} experts: seeds {seeds}")

    bs = int(cfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    # ---- per-expert val logits (cached) ----
    lp = ph / "val_logits.npz"
    if lp.exists() and not args.force_embeddings:
        stack = np.load(lp)["logits"]
        print(f"loaded cached val logits {stack.shape}")
    else:
        t0 = time.time()
        stack = np.stack([M.collect_logits(m, val_loader, device, amp).cpu().numpy() for m in models], 0)
        np.savez_compressed(lp, logits=stack)
        print(f"computed val logits {stack.shape} in {time.time() - t0:.0f}s")
    per_expert_nll = [nll_np(softmax(stack[i]), val_labels) for i in range(len(models))]
    print("per-expert val NLL:", [f"{v:.4f}" for v in per_expert_nll])
    best_i = int(np.argmin(per_expert_nll))
    if args.embed_source == "best":
        use_i = [best_i]
        print(f"embedding source: best expert (seed {seeds[best_i]}, val NLL {per_expert_nll[best_i]:.4f})")
    else:
        use_i = list(range(len(models)))
        print(f"embedding source: mean of all {len(models)} experts")

    # ---- per-expert train/val embeddings (cached) ----
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=args.num_workers)
    train_embs, val_embs = [], []
    for i in use_i:
        tp = ph / f"train_emb_seed{seeds[i]}.npz"
        vp = ph / f"val_emb_seed{seeds[i]}.npz"
        train_embs.append(cached_embeddings(models[i], train_loader, device, amp, tp, args.force_embeddings))
        val_embs.append(cached_embeddings(models[i], val_loader, device, amp, vp, args.force_embeddings))
    train_emb = np.mean(np.stack(train_embs, 0), 0) if args.embed_source == "average" else train_embs[0]
    val_emb = np.mean(np.stack(val_embs, 0), 0) if args.embed_source == "average" else val_embs[0]
    print(f"train emb {train_emb.shape} | val emb {val_emb.shape}")

    # ---- UMAP + KMeans on pooled train+val (cached) ----
    import umap
    from sklearn.cluster import KMeans

    pipe_path = ph / f"entropy_pipeline_k{args.k}_c{args.umap_components}_{args.embed_source}.pkl"
    if pipe_path.exists() and not args.force_pipeline:
        pipe = pickle.loads(pipe_path.read_bytes())
        print(f"loaded cached pipeline (k={args.k})")
    else:
        t0 = time.time()
        all_emb = np.concatenate([train_emb, val_emb], 0)
        um = umap.UMAP(n_components=args.umap_components, n_neighbors=15, min_dist=0.1,
                       metric="cosine", random_state=int(cfg.get("base_seed", 0)))
        z = um.fit_transform(all_emb)
        km = KMeans(n_clusters=args.k, n_init=10, random_state=int(cfg.get("base_seed", 0))).fit(z)
        pipe = {"umap": um, "kmeans": km, "k": args.k,
                "umap_components": args.umap_components, "embed_source": args.embed_source}
        pipe_path.write_bytes(pickle.dumps(pipe))
        print(f"fit UMAP+KMeans on {all_emb.shape[0]} pts in {time.time() - t0:.0f}s -> {pipe_path.name}")

    z_val = pipe["umap"].transform(val_emb)
    val_cl = pipe["kmeans"].predict(z_val)

    # ---- pooled train+val class entropy per cluster ----
    all_labels = np.concatenate([train_labels, val_labels])
    all_emb_cat = np.concatenate([train_emb, val_emb], 0)
    all_cl = pipe["kmeans"].predict(pipe["umap"].transform(all_emb_cat))
    centers = pipe["kmeans"].cluster_centers_
    Hn = np.zeros(args.k)
    sizes = np.bincount(all_cl, minlength=args.k)
    for k in range(args.k):
        m = all_cl == k
        counts = np.bincount(all_labels[m], minlength=num_classes).astype(np.float64)
        Hn[k] = cluster_entropy(counts, args.entropy)

    val_sizes = np.bincount(val_cl, minlength=args.k)
    print("\nper-cluster pooled train+val stats:")
    for k in range(args.k):
        flag = " <-- tiny val support" if val_sizes[k] < MIN_CLUSTER else ""
        print(f"  cl {k:2d}: n_trainval={sizes[k]:5d} n_val={val_sizes[k]:4d} "
              f"centroid=({centers[k][0]:6.2f},{centers[k][1]:6.2f}) "
              f"{args.entropy}_Hn={Hn[k]:.3f}{flag}")

    # ---- temperature law & fit ----
    avg_logits = stack.mean(axis=0)
    T_global = float(fit_temperature(torch.tensor(avg_logits), torch.tensor(val_labels)))

    def cluster_Tk(alpha, T0):
        T = T0 * (1.0 + alpha * (Hn - Hn.mean()))
        np.clip(T, args.temp_clip_min * T0, args.temp_clip_max * T0, out=T)
        T[val_sizes < MIN_CLUSTER] = T0
        return T

    def nll_at(alpha, T0, logits, labels, cl):
        T = cluster_Tk(alpha, T0)[cl]
        return nll_np(softmax(logits, T), labels)

    best_alpha, best_ent = min((a, nll_at(a, T_global, avg_logits, val_labels, val_cl)) for a in alphas)
    print(f"\nT_global={T_global:.4f} | alpha grid -> best alpha={best_alpha} (val NLL {best_ent:.4f})")
    Tk = cluster_Tk(best_alpha, T_global)
    print("per-cluster T_k:", [f"{t:.3f}" for t in Tk])

    # ---- within-val CV (fits T_global and alpha per fold) ----
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(val_labels))
    folds = np.array_split(idx, args.n_folds)
    cv_nlls, cv_Ts = [], []
    for f in folds:
        tr = np.setdiff1d(np.arange(len(val_labels)), f)
        T0 = float(fit_temperature(torch.tensor(avg_logits[tr]), torch.tensor(val_labels[tr])))
        cv_Ts.append(T0)
        a_f = min(alphas, key=lambda a: nll_at(a, T0, avg_logits[tr], val_labels[tr], val_cl[tr]))
        cv_nlls.append(nll_at(a_f, T0, avg_logits[f], val_labels[f], val_cl[f]))
    cv_nll = float(np.mean(cv_nlls))
    cv_T = float(np.mean(cv_Ts))
    print(f"within-val CV: cluster_entropy_T NLL={cv_nll:.4f} | temp_scaled NLL={cv_T:.4f}")

    # ---- evaluation vs baselines ----
    probs_plain = softmax(avg_logits)
    probs_ts = softmax(avg_logits, T_global)
    probs_ce = softmax(avg_logits, Tk[val_cl])
    report = {
        "embed_source": args.embed_source,
        "k": args.k,
        "entropy": args.entropy,
        "alpha": best_alpha,
        "T_global": T_global,
        "per_cluster_T": Tk.tolist(),
        "per_cluster_Hn": Hn.tolist(),
        "per_cluster_sizes": sizes.tolist(),
        "best_expert": {"seed": int(seeds[best_i]), "val_nll": float(per_expert_nll[best_i])},
        "per_expert_val_nll": per_expert_nll,
    }
    methods = {
        "plain_average": (probs_plain, None),
        "temp_scaled_average": (probs_ts, cv_T),
        "cluster_entropy_T": (probs_ce, cv_nll),
    }
    for name, (p, cv) in methods.items():
        report[name] = {"metrics": compute_all_metrics(p, val_labels), "cv_nll": cv}
        cv_s = f"cv {cv:.4f}" if cv is not None else "cv  -    "
        m = report[name]["metrics"]
        print(f"  {name:22s} val NLL={m['nll']:.4f} ECE={m['ece']:.4f} "
              f"Brier={m['brier']:.4f} acc={m['accuracy']:.4f} ({cv_s})")

    chosen = args.method
    if chosen == "auto":
        chosen = min(report, key=lambda k: (report[k]["metrics"]["nll"], report[k]["cv_nll"] or np.inf))
    report["chosen"] = chosen
    print(f"\nchosen: {chosen}")

    (ph / "report.json").write_text(json.dumps(jsonable(report), indent=2))
    (ph / "chosen.json").write_text(json.dumps(jsonable({"method": chosen, "alpha": best_alpha,
                                                         "T_global": T_global}), indent=2))
    (ph / "pipeline.pkl").write_bytes(pickle.dumps(pipe))

    if args.skip_submission:
        return

    # ---- submission ----
    all_uids, all_probs = [], []

    def test_logits(split, loader):
        p = ph / f"test_logits_{split}.npz"
        if p.exists() and not args.force_embeddings:
            return np.load(p)["logits"]
        arr = np.stack([M.collect_logits(m, loader, device, amp).cpu().numpy() for m in models], 0)
        np.savez_compressed(p, logits=arr)
        return arr

    for split in splits:
        test_ds = IWildCamChallengeDataset(data_root, split, tf)
        uids = test_ds.uids
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=args.num_workers)
        tstack = test_logits(split, loader)
        tavg = tstack.mean(axis=0)

        if chosen in ("plain_average", "temp_scaled_average"):
            T = 1.0 if chosen == "plain_average" else T_global
            tprobs = softmax(tavg, T)
        else:
            tembs = []
            for i in use_i:
                ep = ph / f"test_emb_{split}_seed{seeds[i]}.npz"
                tembs.append(cached_embeddings(models[i], loader, device, amp, ep, args.force_embeddings))
            t_emb = np.mean(np.stack(tembs, 0), 0) if args.embed_source == "average" else tembs[0]
            t_cl = pipe["kmeans"].predict(pipe["umap"].transform(t_emb))
            tprobs = softmax(tavg, Tk[t_cl])
        all_uids.extend(uids)
        all_probs.append(tprobs)
        print(f"  {split}: {len(uids)} images")

    sub = ph / "submission.csv"
    write_submission(all_uids, np.concatenate(all_probs, axis=0), sub)
    print(f"\nwrote {sub} (method={chosen}, T_global={T_global:.4f}, alpha={best_alpha})")


if __name__ == "__main__":
    main()
