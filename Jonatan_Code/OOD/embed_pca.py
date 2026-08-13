"""Feature-space visualization for OOD detection.

Loads a ConvNeXt-Base backbone pretrained with DINOv3 self-supervised
weights (timm id ``convnext_base.dinov3_lvd1689m``), embeds every image in
``challenge_data/train`` and ``challenge_data/val``, fits a 2-component PCA
on the pooled embeddings, and scatter-plots the projection — train in blue,
val in red. If train and val occupy different regions of the PCA plane,
that's a visual signal of covariate/domain shift between the splits.

Usage (run as a plain script — ``code/OOD`` isn't a Python package, so it
can't be ``-m``-imported):
    python student/code/OOD/embed_pca.py \
        --data-root /path/to/challenge_data \
        --output pca.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from student.data import IWildCamChallengeDataset, default_eval_transform

BACKBONE = "convnext_base.dinov3_lvd1689m"


def load_backbone(device: torch.device) -> torch.nn.Module:
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.eval()
    return model.to(device)


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for imgs, _ in tqdm(loader, desc=desc):
        imgs = imgs.to(device)
        feats = model(imgs)
        chunks.append(feats.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Project ``X`` (N, D) onto its top-2 principal components via SVD."""
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results" / "OOD" / "images" / "pca_train_val.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_backbone(device)
    transform = default_eval_transform()

    train_ds = IWildCamChallengeDataset(args.data_root, "train", transform)
    val_ds = IWildCamChallengeDataset(args.data_root, "val", transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    train_feats = collect_embeddings(model, train_loader, device, "train")
    val_feats = collect_embeddings(model, val_loader, device, "val")
    print(f"train: {train_feats.shape}, val: {val_feats.shape}")

    all_feats = np.concatenate([train_feats, val_feats], axis=0)
    proj = pca_2d(all_feats)
    train_proj = proj[: len(train_feats)]
    val_proj = proj[len(train_feats):]

    np.savez(
        args.output.with_suffix(".npz"),
        train_proj=train_proj, val_proj=val_proj,
        train_feats=train_feats, val_feats=val_feats,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(train_proj[:, 0], train_proj[:, 1], s=6, c="tab:blue",
               alpha=0.5, label=f"train (n={len(train_proj)})")
    ax.scatter(val_proj[:, 0], val_proj[:, 1], s=6, c="tab:red",
               alpha=0.5, label=f"val (n={len(val_proj)})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of {BACKBONE} embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
