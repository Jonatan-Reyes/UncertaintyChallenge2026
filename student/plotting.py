"""Reliability diagram (calibration plot).

Bins match ``student.metrics.ece`` exactly (equal-width bins on max-softmax
confidence, last bin's upper edge inclusive) so the plot is a visual of the
same numbers ECE summarizes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from student.metrics import ECE_BINS, ece


def plot_reliability_diagram(
    probs: np.ndarray, labels: np.ndarray, path: Path, n_bins: int = ECE_BINS
) -> None:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if mask.any():
            bin_acc[i] = correct[mask].mean()
            bin_conf[i] = confs[mask].mean()

    width = 1.0 / n_bins
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(centers, bin_acc, width=width, edgecolor="black", color="steelblue",
           label="Accuracy", zorder=2)
    gap = bin_conf - bin_acc
    ax.bar(centers, gap, bottom=bin_acc, width=width, edgecolor="red",
           color="red", alpha=0.3, label="Gap to confidence", zorder=1)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece(probs, labels, n_bins):.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def energy_score(member_logits: np.ndarray) -> np.ndarray:
    """Ensemble-averaged energy score, shape ``(N,)``, from per-member raw
    logits ``(K, N, C)`` (i.e. ``Classifier.forward_members`` output, not
    ``forward``'s probability-averaged output — that sums to 1 per row, which
    would make ``logsumexp`` trivially 0 for every sample).

    Energy = ``-logsumexp(logits)`` (Liu et al., 2020): lower for inputs the
    model finds familiar, higher for OOD/unfamiliar ones. Computed per member
    then averaged, matching how the rest of the ensemble is combined.
    """
    per_member = -_logsumexp(member_logits, axis=-1)  # (K, N)
    return per_member.mean(axis=0)


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``labels`` is 1 for the positive class. Returns ``(fpr, tpr)``."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    return fpr, tpr


def plot_energy_ood_roc(id_energy: np.ndarray, ood_energy: np.ndarray, path: Path) -> None:
    """ROC curve for OOD detection: score = energy, positive class = OOD."""
    scores = np.concatenate([id_energy, ood_energy])
    labels = np.concatenate([np.zeros(len(id_energy)), np.ones(len(ood_energy))])
    fpr, tpr = _roc_curve(scores, labels)
    auc = float(np.trapz(tpr, fpr))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", label=f"Energy (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD Detection via Energy Score (val: id vs. ood)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
