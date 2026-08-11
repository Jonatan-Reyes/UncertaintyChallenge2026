"""Train a gated deep ensemble of K DINOv3+LoRA cluster-expert models.

Same pipeline as ``train_ensemble``, but:
- The experts use a frozen DINOv3 backbone (facebook/dinov3-base via timm
  ``vit_base_patch16_dinov3``) with LoRA adapters on the attention
  ``qkv``/``proj`` projections, so only ~0.5% of the backbone params train.
- Clustering runs on 64x64 corner crops stitched into 128x128 images
  (``challenge_data_prep_corners64``), instead of 32x32 corners.

Usage:
    python train_ensemble_dinov3.py [options]
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
from peft import LoraConfig, inject_adapter_in_model
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Subset

from student.data import (
    IWildCamChallengeDataset,
    default_eval_transform,
    default_train_transform,
)
from student.metrics import compute_all_metrics
from student.predict import write_submission
from student.train import Trainer, save_checkpoint, temperature_scale
from train_ensemble import (
    GATE_TAUS,
    apply_pipeline,
    collect_probs,
    fit_cluster_candidate,
    gated_probs,
    load_corner_arrays,
    pretrained_embeddings,
)

DEFAULT_DATA_ROOT = Path("/home/alice/work/dtu_ss_26/challenge_data_prep")
DEFAULT_CORNER_ROOT = Path("/home/alice/work/dtu_ss_26/challenge_data_prep_corners64")
DEFAULT_OUTPUT_DIR = Path("/home/alice/work/dtu_ss_26/runs/ensemble_dinov3")

DINOV3_BACKBONE = "vit_base_patch16_dinov3"


class DinoV3LoraExpert(nn.Module):
    """Frozen DINOv3 backbone + LoRA adapters + linear head."""

    def __init__(
        self,
        num_classes: int,
        backbone_id: str = DINOV3_BACKBONE,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        backbone = timm.create_model(backbone_id, pretrained=True, num_classes=0)
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.backbone_id = backbone_id
        self.embed_dim = int(backbone.num_features)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["qkv", "proj"],
            bias="none",
        )
        self.backbone = inject_adapter_in_model(lora_cfg, backbone)
        self.head = nn.Linear(self.embed_dim, int(num_classes))
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


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

    model = DinoV3LoraExpert(
        train_ds.num_classes, backbone_id=cfg.backbone,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=nn.CrossEntropyLoss(),
        device=device, patience=cfg.patience, early_stop_metric="accuracy",
    )

    print(f"\n=== expert {k}: {len(train_idx)} train / {len(val_idx)} val "
          f"({sum(p.numel() for p in trainable):,} trainable params) ===")
    trainer.fit(cfg.epochs)

    T = temperature_scale(model, val_loader, device) if len(es_val_idx) >= 8 else 1.0
    ckpt = out_dir / "experts" / f"expert_{k}.pt"
    save_checkpoint(
        model, train_ds.num_classes, T, ckpt,
        hyperparameters={**hyper, "cluster": int(k), "val_subset": int(len(val_idx))},
    )
    print(f"  saved {ckpt} (T={T:.4f})")
    return model, T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corner-root", type=Path, default=DEFAULT_CORNER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--backbone", type=str, default=DINOV3_BACKBONE)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
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
    print("\n[stage 1] clustering val corner images (64x64 corners)...")
    cand_pix = fit_cluster_candidate(
        load_corner_arrays(args.corner_root, "val", val_uids), args.n_clusters, args.seed
    )
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

    # ---- Stage 3: train K DINOv3+LoRA experts ----
    print("\n[stage 3] training DINOv3+LoRA experts...")
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
