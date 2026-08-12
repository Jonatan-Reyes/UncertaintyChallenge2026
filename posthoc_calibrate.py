"""Post-hoc cluster-conditional calibration (UMAP+KMeans on frozen-backbone embeddings).

Trains nothing. Uses the UMAP+KMeans structure of the *frozen backbone* val
embeddings (see notebooks/explore_giant_embeddings.ipynb) as a post-inference
calibration map:

  * cluster k gets its own temperature T_k (minimizes that cluster's val NLL)
  * cluster k gets a prior-smoothing weight lam_k blending predictions toward
    the cluster's (Laplace-smoothed) empirical class distribution pi_k:

        p'(x) = (1 - lam_k) * softmax(logits / T_k) + lam_k * pi_k

    High-entropy clusters have a near-flat pi_k, so overconfidence is damped
    exactly where the class is genuinely ambiguous.

Test time: embed with the frozen backbone, UMAP-transform, nearest cluster,
apply (T_k, lam_k). Per-cluster scalars are shrunk toward the global value and
cross-validated within val (5-fold) to guard the 918 val points.

Usage:
  python posthoc_calibrate.py --run-dir runs/ensemble_dinov3_raw_full_448
  python posthoc_calibrate.py --run-dir <edda_legA_dir> --k-list 5 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

import train_dinov3_ensemble as M
from student.data import IWildCamChallengeDataset
from student.metrics import compute_all_metrics
from student.predict import write_submission
from student.train import fit_temperature

DEFAULT_RUN = Path("/home/alice/work/dtu_ss_26/runs/ensemble_dinov3_raw_full_448")
DEFAULT_SPLITS = ["test_public", "test_private"]
MIN_PTS = 5  # minimum points per cluster to fit its own scalar


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    x = x / T
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def nll_np(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())


def laplace_prior(labels: np.ndarray, num_classes: int) -> np.ndarray:
    p = np.bincount(labels, minlength=num_classes).astype(np.float64) + 1.0
    return p / p.sum()


def fit_lambda(probs: np.ndarray, labels: np.ndarray, prior: np.ndarray) -> float:
    """lambda in [0,1] minimizing NLL of (1-lam)*probs + lam*prior."""
    if len(labels) < MIN_PTS:
        return 0.0

    def loss(lam: float) -> float:
        p = (1.0 - lam) * probs + lam * prior
        return nll_np(p, labels)

    res = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def fit_pipeline(emb: np.ndarray, n_clusters: int, umap_components: int, seed: int) -> dict:
    import umap
    from sklearn.cluster import KMeans

    um = umap.UMAP(
        n_components=umap_components, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=seed,
    )
    z = um.fit_transform(emb)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(z)
    return {"umap": um, "kmeans": km}


def assign_clusters(pipeline: dict, emb: np.ndarray) -> np.ndarray:
    z = pipeline["umap"].transform(emb)
    return pipeline["kmeans"].predict(z)


def embed_frozen(backbone, loader, device, amp_bf16: bool) -> np.ndarray:
    backbone.eval()
    chunks = []
    with torch.no_grad():
        for imgs, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_bf16 and device.type == "cuda"):
                chunks.append(backbone(imgs.to(device)).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def jsonable(x):
    """Convert numpy scalar/array values to JSON-native types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(i) for i in x]
    return x


# --------------------------------------------------------------------------- #
# calibration methods (fit on val, return probs + params)
# --------------------------------------------------------------------------- #
def calib_global_t(avg_logits: np.ndarray, labels: np.ndarray):
    T = fit_temperature(torch.tensor(avg_logits), torch.tensor(labels))
    return softmax(avg_logits, T), {"T": T}


def calib_cluster(avg_logits, labels, cl, n_clusters, num_classes, T_global,
                  fit_t: bool, fit_lam: bool, shrink: float = 0.5):
    """Per-cluster temperature and/or prior smoothing, shrunk toward global."""
    n = len(labels)
    probs = np.empty((n, avg_logits.shape[1]), dtype=np.float64)
    Ts, lams, priors = {}, {}, {}
    for k in range(n_clusters):
        m = cl == k
        if m.sum() < MIN_PTS:
            T_k, lam_k = T_global, 0.0
        else:
            T_raw = fit_temperature(torch.tensor(avg_logits[m]), torch.tensor(labels[m])) if fit_t else T_global
            T_k = shrink * T_global + (1.0 - shrink) * T_raw  # shrink toward global T
            p_k = softmax(avg_logits[m], T_k)
            pi_k = laplace_prior(labels[m], num_classes)
            lam_k = fit_lambda(p_k, labels[m], pi_k) if fit_lam else 0.0
            probs[m] = (1.0 - lam_k) * p_k + lam_k * pi_k
            Ts[k] = T_k; lams[k] = lam_k; priors[k] = pi_k
            continue
        p_k = softmax(avg_logits[m], T_k)
        pi_k = laplace_prior(labels[m], num_classes)
        probs[m] = (1.0 - lam_k) * p_k + lam_k * pi_k
        Ts[k] = T_k; lams[k] = lam_k; priors[k] = pi_k
    return probs, {"T": Ts, "lam": lams, "prior": priors, "shrink": shrink}


# --------------------------------------------------------------------------- #
# within-val 5-fold CV of a cluster-fit (used to guard overfitting)
# --------------------------------------------------------------------------- #
def cv_nll(avg_logits, labels, cl, n_clusters, num_classes, fit_t, fit_lam,
           T_global, shrink=0.5, n_folds: int = 5) -> float:
    """Within-val NLL: fit the calibration scalars on each fold-train, apply to
    the fold. A high cv vs val gap flags overfitting of the 918 val points."""
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(labels))
    folds = np.array_split(idx, n_folds)
    tot, cnt = 0.0, 0
    for f in folds:
        tr = np.setdiff1d(np.arange(len(labels)), f)
        Tg = fit_temperature(torch.tensor(avg_logits[tr]), torch.tensor(labels[tr]))
        _, params = calib_cluster(avg_logits[tr], labels[tr], cl[tr], n_clusters,
                                  num_classes, Tg, fit_t, fit_lam, shrink)
        T_s, la_s, pri = params["T"], params["lam"], params["prior"]
        probs = np.empty((len(f), avg_logits.shape[1]), dtype=np.float64)
        for k in range(n_clusters):
            m = cl[f] == k
            if m.sum() == 0:
                continue
            p_k = softmax(avg_logits[f][m], T_s.get(k, Tg))
            lam_k = la_s.get(k, 0.0)
            probs[m] = (1.0 - lam_k) * p_k + lam_k * pri.get(k, np.full(num_classes, 1 / num_classes))
        tot += nll_np(probs, labels[f]) * len(f)
        cnt += len(f)
    return tot / cnt


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--k-list", nargs="+", type=int, default=[5, 10],
                        help="cluster counts to try (e.g. 5 10)")
    parser.add_argument("--umap-components", type=int, default=2)
    parser.add_argument("--shrink", type=float, default=0.5,
                        help="how much per-cluster T is pulled toward the global T (0=raw, 1=fully global)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force-logits", action="store_true", help="recompute val/test logits")
    parser.add_argument("--skip-submission", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.run_dir)
    cfg = json.loads((out_dir / "config.json").read_text())
    ph = out_dir / "posthoc"
    ph.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tf = M.eval_transform(int(cfg["img_size"]))
    val_ds = IWildCamChallengeDataset(cfg["data_root"], "val", tf)
    val_labels = np.asarray(val_ds.labels, dtype=np.int64)
    print(f"val: {len(val_ds)} images, {val_ds.num_classes} classes | run {out_dir.name}")

    # ---- experts ----
    ckpts = sorted(out_dir.glob("experts/expert_seed*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no expert_seed*.pt in {out_dir/'experts'}")
    print(f"loading {len(ckpts)} experts")
    models = []
    for ckpt in ckpts:
        sd = torch.load(ckpt, map_location=device)
        m = M.DinoV3LoraExpert(
            val_ds.num_classes, backbone_id=cfg["backbone"],
            lora_r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            lora_last_layers=int(cfg["lora_last_layers"]),
        ).to(device)
        m.load_state_dict(sd["state_dict"])
        m.eval()
        models.append(m)

    amp = bool(cfg.get("amp_bf16", False))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    # ---- per-expert val logits (cached) ----
    logits_path = ph / "val_logits.npz"
    if logits_path.exists() and not args.force_logits:
        z = np.load(logits_path)
        stack = z["logits"]; print(f"loaded cached val logits {stack.shape}")
    else:
        t0 = time.time()
        logits = [M.collect_logits(m, val_loader, device, amp) for m in models]
        stack = torch.stack(logits, dim=0).cpu().numpy()
        np.savez_compressed(logits_path, logits=stack)
        print(f"computed val logits {stack.shape} in {time.time()-t0:.0f}s")

    # ---- frozen-backbone val embeddings (cached) ----
    emb_path = ph / "val_emb.npz"
    if emb_path.exists() and not args.force_logits:
        val_emb = np.load(emb_path)["emb"]; print(f"loaded cached val embeddings {val_emb.shape}")
    else:
        backbone = M.timm_create_backbone(cfg["backbone"]).to(device)
        val_emb = embed_frozen(backbone, val_loader, device, amp)
        np.savez_compressed(emb_path, emb=val_emb)
        print(f"computed val embeddings {val_emb.shape}")

    avg_logits = stack.mean(axis=0)  # (N, C) logits averaged across experts
    num_classes = val_ds.num_classes

    # ---- global baselines ----
    report = {}
    probs_plain = softmax(avg_logits)
    report["plain_average"] = {"metrics": compute_all_metrics(probs_plain, val_labels), "cv_nll": None}
    probs_t, _gp = calib_global_t(avg_logits, val_labels)
    T_global = float(_gp["T"])
    report["temp_scaled_average"] = {"metrics": compute_all_metrics(probs_t, val_labels),
                                     "cv_nll": cv_nll(avg_logits, val_labels,
                                                      np.zeros(len(val_labels), dtype=int), 1,
                                                      num_classes, True, False, T_global,
                                                      shrink=0.0, n_folds=args.n_folds),
                                     "params": {"T": T_global}}
    print(f"baselines: plain NLL={report['plain_average']['metrics']['nll']:.4f} | "
          f"temp-scaled NLL={report['temp_scaled_average']['metrics']['nll']:.4f} (T={T_global:.3f})")

    # ---- cluster-conditional methods ----
    pipelines, chosen = {}, {"name": "temp_scaled_average", "params": {"T": T_global}}
    for K in args.k_list:
        print(f"\n=== K = {K} ===")
        pipe = fit_pipeline(val_emb, K, args.umap_components, int(cfg.get("base_seed", 0)))
        cl = assign_clusters(pipe, val_emb)
        pipelines[K] = pipe
        sizes = np.bincount(cl, minlength=K).tolist()
        print("  val cluster sizes:", sizes)

        methods = {
            "cluster_T":      {"fit_t": True,  "fit_lam": False},
            "cluster_prior":  {"fit_t": False, "fit_lam": True},
            "cluster_both":   {"fit_t": True,  "fit_lam": True},
        }
        for name, spec in methods.items():
            probs, params = calib_cluster(avg_logits, val_labels, cl, K, num_classes,
                                          T_global, spec["fit_t"], spec["fit_lam"], args.shrink)
            mkey = f"{name}_k{K}"
            cv = cv_nll(avg_logits, val_labels, cl, K, num_classes, spec["fit_t"],
                        spec["fit_lam"], T_global, args.shrink, args.n_folds)
            report[mkey] = {"metrics": compute_all_metrics(probs, val_labels),
                            "cv_nll": cv, "params": params, "K": K, "clusters": sizes}
            print(f"  {mkey:22s} val NLL={report[mkey]['metrics']['nll']:.4f} "
                  f"(cv {cv:.4f}) ECE={report[mkey]['metrics']['ece']:.4f} "
                  f"Brier={report[mkey]['metrics']['brier']:.4f} acc={report[mkey]['metrics']['accuracy']:.4f}")

    # ---- choose by val NLL, tie-break by cv NLL ----
    best_name = min(report, key=lambda k: (report[k]["metrics"]["nll"],
                                           report[k]["cv_nll"] or np.inf))
    chosen = {"name": best_name, "K": report[best_name].get("K"), "params": report[best_name].get("params")}
    print(f"\nchosen: {best_name}  (val NLL {report[best_name]['metrics']['nll']:.4f})")

    (ph / "report.json").write_text(json.dumps(
        jsonable({"chosen": chosen,
                  "methods": {k: {"metrics": v["metrics"], "cv_nll": v["cv_nll"]} for k, v in report.items()},
                  "val_labels": val_labels.tolist()}), indent=2))

    # save pipeline + params for reuse
    pipe = pipelines[chosen["K"]] if chosen["K"] else None
    (ph / "pipeline.pkl").write_bytes(pickle.dumps(pipe))
    (ph / "chosen.json").write_text(json.dumps(jsonable(chosen), indent=2))
    print(f"saved {ph/'report.json'} {ph/'pipeline.pkl'} {ph/'chosen.json'}")

    if args.skip_submission:
        return

    # ---- test-time application ----
    if pipe is None:
        print("chosen method is global; writing submission (no cluster step)")
    else:
        print("cluster-calibrating test images...")

    backbone = None
    all_uids, all_probs = [], []

    def test_logits(split):
        p = ph / f"test_logits_{split}.npz"
        if p.exists() and not args.force_logits:
            return np.load(p)["logits"]
        tlog = [M.collect_logits(m, loader, device, amp) for m in models]
        arr = torch.stack(tlog, dim=0).cpu().numpy()
        np.savez_compressed(p, logits=arr)
        return arr

    for split in cfg.get("splits", DEFAULT_SPLITS):
        test_ds = IWildCamChallengeDataset(cfg["data_root"], split, tf)
        uids = test_ds.uids
        loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
        tstack = test_logits(split)
        tavg = tstack.mean(axis=0)

        if pipe is None:
            T = chosen["params"]["T"]
            tprobs = softmax(tavg, T)
        else:
            if backbone is None:
                backbone = M.timm_create_backbone(cfg["backbone"]).to(device)
            ep = ph / f"test_emb_{split}.npz"
            if ep.exists() and not args.force_logits:
                t_emb = np.load(ep)["emb"]
            else:
                t_emb = embed_frozen(backbone, loader, device, amp)
                np.savez_compressed(ep, emb=t_emb)
            tcl = assign_clusters(pipe, t_emb)
            T_s = chosen["params"]["T"]; la_s = chosen["params"]["lam"]; pr_s = chosen["params"]["prior"]

            def lookup(d, k):
                return d.get(k, d.get(str(k)))

            tprobs = np.empty_like(softmax(tavg))
            for k in range(chosen["K"]):
                m = tcl == k
                if m.sum() == 0:
                    continue
                p_k = softmax(tavg[m], float(lookup(T_s, k)))
                lam_k = float(lookup(la_s, k))
                tprobs[m] = (1.0 - lam_k) * p_k + lam_k * np.asarray(lookup(pr_s, k), dtype=np.float64)
        all_uids.extend(uids)
        all_probs.append(tprobs)
        print(f"  {split}: {len(uids)} images")

    sub = ph / "submission.csv"
    write_submission(all_uids, np.concatenate(all_probs, axis=0), sub)
    print(f"wrote {sub}  (method={chosen['name']})")


if __name__ == "__main__":
    main()
