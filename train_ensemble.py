"""Train a gated deep ensemble of K cluster-expert models (iWildCam challenge).

Pipeline:
1. Cluster the validation 64x64 corner images into K groups. Two feature
   spaces are tried -- raw pixels and pretrained-backbone embeddings -- and
   the one with the higher silhouette score is kept.
2. Route every training image to a cluster via nearest centroid, using its
   corner signature, producing K sub-training sets.
3. Train K experts, each on its sub-domain's full-resolution training images
   (no corner cropping).
4. At inference, weight each expert by soft cluster membership of the query
   (softmax over -distance/tau to centroids) and combine temperature-scaled
   probabilities, so the nearest-cluster expert dominates.

Usage:
    python train_ensemble.py [--data-root ...] [--corner-root ...] ...
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from tqdm import tqdm

from student.data import (
    IWildCamChallengeDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    default_eval_transform,
    default_train_transform,
)
from student.metrics import compute_all_metrics
from student.model import Classifier
from student.predict import write_submission
from student.train import Trainer, make_optimizer, save_checkpoint, temperature_scale

DEFAULT_DATA_ROOT = Path("/home/alice/work/dtu_ss_26/challenge_data_prep")
DEFAULT_CORNER_ROOT = Path("/home/alice/work/dtu_ss_26/challenge_data_prep_corners")
DEFAULT_OUTPUT_DIR = Path("/home/alice/work/dtu_ss_26/runs/ensemble_experts")

CORNER_SIZE = 64
PCA_COMPONENTS = 100
GATE_TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
EMBED_INPUT = 224


def load_corner_arrays(corner_root: Path, split: str, uids: list[str]) -> np.ndarray:
    imgs_dir = corner_root / split / "images"
    out = np.empty((len(uids), CORNER_SIZE * CORNER_SIZE * 3), dtype=np.float32)
    for i, u in enumerate(tqdm(uids, desc="load corner arrays", leave=False)):
        img = Image.open(imgs_dir / f"{u}.jpg").convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        if i == 0:
            if arr.size != out.shape[1]:
                out = np.empty((len(uids), arr.size), dtype=np.float32)
        out[i] = arr.reshape(-1) / 255.0
    return out


def pretrained_embeddings(
    uids: list[str],
    corner_root: Path,
    split: str,
    backbone: str,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model = timm.create_model(backbone, pretrained=True, num_classes=0).to(device).eval()
    transform = T.Compose(
        [
            T.Resize((EMBED_INPUT, EMBED_INPUT)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    imgs_dir = corner_root / split / "images"
    chunks = []
    with torch.no_grad():
        for i in range(0, len(uids), batch_size):
            batch = [
                transform(Image.open(imgs_dir / f"{u}.jpg").convert("RGB"))
                for u in uids[i : i + batch_size]
            ]
            x = torch.stack(batch).to(device)
            chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def fit_cluster_candidate(
    X: np.ndarray, n_clusters: int, seed: int
) -> dict:
    scaler = StandardScaler().fit(X)
    z = scaler.transform(X)
    n_comp = min(PCA_COMPONENTS, z.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed).fit(z)
    zp = pca.transform(z)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(zp)
    return {
        "scaler": scaler,
        "pca": pca,
        "kmeans": km,
        "labels": labels,
        "silhouette": float(silhouette_score(zp, labels)),
        "pca_components": int(n_comp),
    }


def apply_pipeline(
    pipeline: dict,
    corner_root: Path,
    split: str,
    uids: list[str],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    if pipeline["method"] == "pixels":
        X = load_corner_arrays(corner_root, split, uids)
    else:
        X = pretrained_embeddings(uids, corner_root, split, pipeline["backbone"], device, batch_size)
    z = pipeline["pca"].transform(pipeline["scaler"].transform(X))
    return z


def train_expert(
    k: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    full_val_idx: np.ndarray,
    train_ds,
    val_ds,
    cfg,
    device: torch.device,
    out_dir: Path,
    hyper: dict,
):
    if len(val_idx) >= 20:
        es_val_idx = val_idx
    else:
        es_val_idx = full_val_idx
        print(f"  (val subset {len(val_idx)} < 20 -> early stopping on full val)")

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_ds, es_val_idx), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = Classifier(
        train_ds.num_classes, backbone_name=cfg.backbone, pretrained=True
    ).to(device)
    optimizer = make_optimizer(
        model, lr_backbone=cfg.lr, lr_head=cfg.head_lr, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=nn.CrossEntropyLoss(),
        device=device, patience=cfg.patience, early_stop_metric="accuracy",
    )

    print(f"\n=== expert {k}: {len(train_idx)} train / {len(val_idx)} val ===")
    trainer.fit(cfg.epochs)

    T = temperature_scale(model, val_loader, device) if len(es_val_idx) >= 8 else 1.0
    ckpt = out_dir / "experts" / f"expert_{k}.pt"
    save_checkpoint(
        model, train_ds.num_classes, T, ckpt,
        hyperparameters={**hyper, "cluster": int(k), "val_subset": int(len(val_idx))},
    )
    print(f"  saved {ckpt} (T={T:.4f})")
    return model, T


def collect_probs(model, loader: DataLoader, device, T: float = 1.0) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            chunks.append(torch.softmax(model(imgs) / T, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def gated_probs(probs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """probs: (K, n, C), w: (n, K) -> (n, C)."""
    return np.einsum("nk,kns->ns", w, probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corner-root", type=Path, default=DEFAULT_CORNER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["test_public", "test_private"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    cluster_dir = out_dir / "cluster"
    expert_dir = out_dir / "experts"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    expert_dir.mkdir(parents=True, exist_ok=True)

    hyper = {
        "n_clusters": int(args.n_clusters),
        "backbone": args.backbone,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "head_lr": float(args.head_lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "data_root": str(args.data_root),
        "corner_root": str(args.corner_root),
    }
    (out_dir / "config.json").write_text(json.dumps(hyper, indent=2))

    val_ds = IWildCamChallengeDataset(args.data_root, "val", default_eval_transform())
    val_uids = val_ds.uids
    val_labels = np.asarray(val_ds.labels)
    domains = val_ds.domains
    print(f"val: {len(val_uids)} images, {val_ds.num_classes} classes")

    # ---- Stage 1: clustering on the val corner images, auto-pick method ----
    print("\n[stage 1] clustering val corner images...")
    X_pix = load_corner_arrays(args.corner_root, "val", val_uids)
    cand_pix = fit_cluster_candidate(X_pix, args.n_clusters, args.seed)
    cand_pix["method"] = "pixels"
    print(f"  pixels:     silhouette={cand_pix['silhouette']:.4f}")

    cand_emb = fit_cluster_candidate(
        pretrained_embeddings(val_uids, args.corner_root, "val", args.backbone, device),
        args.n_clusters,
        args.seed,
    )
    cand_emb["method"] = "pretrained"
    cand_emb["backbone"] = args.backbone
    print(f"  pretrained: silhouette={cand_emb['silhouette']:.4f}")

    pipeline = cand_pix if cand_pix["silhouette"] >= cand_emb["silhouette"] else cand_emb
    print(f"  -> using {pipeline['method']} features "
          f"(silhouette={pipeline['silhouette']:.4f})")
    val_assign = np.asarray(pipeline["labels"])
    sizes = np.bincount(val_assign, minlength=args.n_clusters)
    print("  val cluster sizes:", sizes.tolist())

    with open(cluster_dir / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    (cluster_dir / "method.txt").write_text(pipeline["method"])
    (cluster_dir / "silhouette.json").write_text(
        json.dumps({"pixels": cand_pix["silhouette"], "pretrained": cand_emb["silhouette"],
                    "chosen": pipeline["method"]}, indent=2)
    )
    pd.DataFrame({"uid": val_uids, "y": val_labels, "cluster": val_assign}).to_csv(
        cluster_dir / "val_assignments.csv", index=False
    )

    # ---- Stage 2: route training images to clusters ----
    print("\n[stage 2] routing training images to clusters...")
    train_ds = IWildCamChallengeDataset(args.data_root, "train", default_train_transform())
    train_uids = train_ds.uids
    train_labels = np.asarray(train_ds.labels)
    z_train = apply_pipeline(pipeline, args.corner_root, "train", train_uids, device)
    train_assign = pipeline["kmeans"].predict(z_train)
    for k in range(args.n_clusters):
        mask = train_assign == k
        pd.DataFrame({"uid": np.asarray(train_uids)[mask], "y": train_labels[mask]}).to_csv(
            cluster_dir / f"train_{k}.csv", index=False
        )
    print("  train cluster sizes:", np.bincount(train_assign, minlength=args.n_clusters).tolist())

    # ---- Stage 3: train K experts ----
    print("\n[stage 3] training experts...")
    full_val_idx = np.arange(len(val_ds))
    expert_models, expert_Ts = [], []
    for k in range(args.n_clusters):
        train_idx = np.where(train_assign == k)[0]
        val_idx = np.where(val_assign == k)[0]
        if len(train_idx) == 0:
            print(f"  expert {k}: empty training set, skipping")
            expert_models.append(None)
            expert_Ts.append(1.0)
            continue
        model, T = train_expert(
            k, train_idx, val_idx, full_val_idx, train_ds, val_ds, args,
            device, out_dir, hyper,
        )
        expert_models.append(model)
        expert_Ts.append(T)
    torch.cuda.empty_cache()

    # ---- Stage 4: gating + validation metrics + submission ----
    print("\n[stage 4] gating and evaluation...")
    z_val = apply_pipeline(pipeline, args.corner_root, "val", val_uids, device)
    centers = pipeline["kmeans"].cluster_centers_
    D = cdist(z_val, centers)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    expert_val_probs = []
    for k, model in enumerate(expert_models):
        if model is None:
            expert_val_probs.append(np.ones((len(val_ds), val_ds.num_classes)) / val_ds.num_classes)
        else:
            expert_val_probs.append(collect_probs(model, val_loader, device, expert_Ts[k]))
    probs_stack = np.stack(expert_val_probs, axis=0)

    avg_probs = probs_stack.mean(axis=0)
    avg_metrics = compute_all_metrics(avg_probs, val_labels)

    best = None
    for tau in GATE_TAUS:
        w = torch.softmax(torch.tensor(-D, dtype=torch.float64) / tau, dim=1).numpy()
        g = gated_probs(probs_stack, w)
        m = compute_all_metrics(g, val_labels)
        if best is None or m["nll"] < best["metrics"]["nll"]:
            best = {"tau": tau, "weights": w, "probs": g, "metrics": m}

    report = {
        "gate_tau": best["tau"],
        "plain_average": avg_metrics,
        "gated": best["metrics"],
    }
    if domains is not None:
        domains_arr = np.asarray(domains)
        for dom in sorted(set(domains_arr)):
            mask = domains_arr == dom
            report.setdefault("gated_by_domain", {})[dom] = compute_all_metrics(
                best["probs"][mask], val_labels[mask]
            )
    (out_dir / "metrics_val.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # submission over test splits
    all_uids, all_probs = [], []
    for split in args.splits:
        test_ds = IWildCamChallengeDataset(args.data_root, split, default_eval_transform())
        uids = test_ds.uids
        z_test = apply_pipeline(pipeline, args.corner_root, split, uids, device)
        w = torch.softmax(torch.tensor(-cdist(z_test, centers), dtype=torch.float64) / best["tau"],
                          dim=1).numpy()
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        chunks = []
        for k, model in enumerate(expert_models):
            if model is None:
                chunks.append(np.ones((len(uids), test_ds.num_classes)) / test_ds.num_classes)
            else:
                chunks.append(collect_probs(model, loader, device, expert_Ts[k]))
        test_probs = gated_probs(np.stack(chunks, axis=0), w)
        all_uids.extend(uids)
        all_probs.append(test_probs)

    submission = out_dir / "submission.csv"
    write_submission(all_uids, np.concatenate(all_probs, axis=0), submission)
    print(f"\nwrote {submission} (tau={best['tau']})")


if __name__ == "__main__":
    main()
