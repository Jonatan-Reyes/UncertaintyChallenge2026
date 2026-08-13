"""Re-plot the cached train/val PCA projection, colored by domain instead
of just train-vs-val: train (always 'id'), val 'id', and val 'ood' each
get their own color, so in- and out-of-distribution points are visually
distinguishable.

Reuses the embeddings/projection cached by embed_pca.py in
pca_train_val.npz — no re-embedding needed.

Usage:
    python student/code/evaluation/plot_pca_domain.py --data-root challenge_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Okabe-Ito colorblind-safe categorical colors.
COLOR_TRAIN = "#0072B2"     # blue    — train (always 'id')
COLOR_VAL_ID = "#009E73"    # green   — val, in-distribution
COLOR_VAL_OOD = "#D55E00"   # vermillion — val, out-of-distribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=HERE.parents[1] / "results" / "OOD" / "images" / "pca_train_val.npz")
    parser.add_argument("--output", type=Path,
                        default=HERE.parents[1] / "results" / "evaluation" / "images" / "pca_train_val.png")
    args = parser.parse_args()

    d = np.load(args.cache)
    train_proj, val_proj = d["train_proj"], d["val_proj"]

    va = pd.read_csv(args.data_root / "val" / "labels.csv")
    assert len(va) == len(val_proj), "val labels.csv row count doesn't match cached val_proj"
    domain = va["domain"].to_numpy()
    is_ood = domain == "ood"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(train_proj[:, 0], train_proj[:, 1], s=6, c=COLOR_TRAIN,
               alpha=0.4, label=f"train, id (n={len(train_proj)})")
    ax.scatter(val_proj[~is_ood, 0], val_proj[~is_ood, 1], s=10, c=COLOR_VAL_ID,
               alpha=0.8, label=f"val, id (n={(~is_ood).sum()})")
    ax.scatter(val_proj[is_ood, 0], val_proj[is_ood, 1], s=10, c=COLOR_VAL_OOD,
               alpha=0.8, label=f"val, ood (n={is_ood.sum()})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of convnext_base.dinov3_lvd1689m embeddings, by domain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
