"""Distance-based OOD detection in DINOv3 ConvNeXt embedding space.

PCA (embed_pca.py) showed val sitting inside the train cloud in the top-2
components — that's expected, since PCA keeps the directions of highest
*train* variance, which need not be the directions that separate id/ood.
Full-dimensional distance to the train distribution is a stronger signal.

Reuses the embeddings cached by embed_pca.py (``pca_train_val.npz``) and
scores every val point by three distances to the *train* feature cloud:

  - centroid  : Euclidean distance to the train mean
  - mahalanobis: Mahalanobis distance using the train covariance (accounts
                 for feature correlations/anisotropy)
  - knn       : mean Euclidean distance to its k nearest train neighbors
                 (Sun et al. 2022 — usually the strongest of the three)

``val/labels.csv`` carries a ground-truth ``domain`` column (``id``/``ood``)
that is not used anywhere in embedding computation — it's held out purely to
check whether these distances actually separate the two groups, via AUROC
(ood = positive class) and score histograms.

Usage:
    python student/code/OOD/ood_distance.py --data-root challenge_data
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from student.data import IWildCamChallengeDataset

HERE = Path(__file__).resolve().parent


def _load_embed_pca_module():
    """embed_pca.py can't be ``import``-ed normally: ``code/OOD`` isn't a
    Python package. Load it by file path instead."""
    spec = importlib.util.spec_from_file_location("embed_pca", HERE / "embed_pca.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_embeddings(data_root: Path, cache: Path, batch_size: int, num_workers: int):
    if cache.exists():
        print(f"loading cached embeddings from {cache}")
        d = np.load(cache)
        return d["train_feats"], d["val_feats"]

    print(f"{cache} not found, computing embeddings")
    embed_pca = _load_embed_pca_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = embed_pca.load_backbone(device)
    transform = embed_pca.default_eval_transform()
    from torch.utils.data import DataLoader

    train_ds = IWildCamChallengeDataset(data_root, "train", transform)
    val_ds = IWildCamChallengeDataset(data_root, "val", transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_feats = embed_pca.collect_embeddings(model, train_loader, device, "train")
    val_feats = embed_pca.collect_embeddings(model, val_loader, device, "val")
    return train_feats, val_feats


def auroc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """AUROC via Mann-Whitney U (average-rank ties) — same formulation as
    ``student.metrics.misclassification_auroc``, generalized to any score."""
    n_pos = int(is_positive.sum())
    n_neg = int(len(is_positive) - n_pos)
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    pos_rank_sum = float(ranks[is_positive == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def centroid_distance(z_train: torch.Tensor, z_val: torch.Tensor) -> np.ndarray:
    mean = z_train.mean(dim=0, keepdim=True)
    return torch.linalg.norm(z_val - mean, dim=1).cpu().numpy()


def mahalanobis_distance(z_train: torch.Tensor, z_val: torch.Tensor, eps: float = 1e-3) -> np.ndarray:
    mean = z_train.mean(dim=0, keepdim=True)
    Xc = z_train - mean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    cov += eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    L = torch.linalg.cholesky(cov)
    diff = (z_val - mean).T  # (D, N_val)
    y = torch.linalg.solve_triangular(L, diff, upper=False)
    return torch.sqrt((y ** 2).sum(dim=0)).cpu().numpy()


def knn_distance(z_train: torch.Tensor, z_val: torch.Tensor, k: int = 5, chunk: int = 256) -> np.ndarray:
    out = []
    for i in range(0, z_val.shape[0], chunk):
        d = torch.cdist(z_val[i:i + chunk], z_train)  # (chunk, N_train)
        knn = torch.topk(d, k, dim=1, largest=False).values
        out.append(knn.mean(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=HERE.parents[1] / "results" / "OOD" / "images" / "pca_train_val.npz")
    parser.add_argument("--output", type=Path,
                        default=HERE.parents[1] / "results" / "OOD" / "images" / "ood_distance_hist.png")
    parser.add_argument("--k", type=int, default=5, help="k for kNN distance")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    train_feats, val_feats = get_embeddings(args.data_root, args.cache, args.batch_size, args.num_workers)
    print(f"train: {train_feats.shape}, val: {val_feats.shape}")

    val_ds = IWildCamChallengeDataset(args.data_root, "val", transform=None)
    assert val_ds.domains is not None, "val/labels.csv has no domain column"
    domain = np.array(val_ds.domains)
    is_ood = (domain == "ood").astype(int)
    print(f"val domain counts: {pd.Series(domain).value_counts().to_dict()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train_feats.mean(axis=0, keepdims=True)
    std = train_feats.std(axis=0, keepdims=True) + 1e-6
    z_train = torch.from_numpy((train_feats - mean) / std).to(device)
    z_val = torch.from_numpy((val_feats - mean) / std).to(device)

    scores = {
        "centroid": centroid_distance(z_train, z_val),
        "mahalanobis": mahalanobis_distance(z_train, z_val),
        f"knn(k={args.k})": knn_distance(z_train, z_val, k=args.k),
    }

    print("\nOOD-vs-ID separation (higher distance -> predicted OOD):")
    print(f"{'method':<14} {'AUROC':>8}   {'mean_id':>10} {'mean_ood':>10}")
    for name, s in scores.items():
        a = auroc(s, is_ood)
        print(f"{name:<14} {a:>8.4f}   {s[is_ood == 0].mean():>10.3f} {s[is_ood == 1].mean():>10.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scores), figsize=(5 * len(scores), 4))
    for ax, (name, s) in zip(axes, scores.items()):
        ax.hist(s[is_ood == 0], bins=40, alpha=0.6, color="tab:blue", label="id", density=True)
        ax.hist(s[is_ood == 1], bins=40, alpha=0.6, color="tab:red", label="ood", density=True)
        ax.set_title(f"{name}\nAUROC={auroc(s, is_ood):.3f}")
        ax.set_xlabel("distance to train")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
