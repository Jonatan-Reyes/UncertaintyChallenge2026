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
