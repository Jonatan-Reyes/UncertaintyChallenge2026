"""Class-distribution comparison between id and ood locations.

Distance-to-train in embedding space (ood_distance.py) barely separated
val's id/ood domains (AUROC ~0.5). A different hypothesis: ood locations
are different *places*, which may simply host a different mix of species
than the training locations. This plots the per-class count distribution
for val's id vs ood rows (from the ``domain`` column in val/labels.csv),
overlaid, both as absolute counts and as within-domain proportions.

Usage:
    python student/code/evaluation/class_distribution.py --data-root challenge_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

HERE = Path(__file__).resolve().parent

ID_COLOR = "tab:blue"
OOD_COLOR = "tab:red"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=HERE.parents[1] / "results" / "evaluation" / "images" / "class_distribution.png")
    args = parser.parse_args()

    with (args.data_root / "class_mapping.json").open() as f:
        num_classes = int(json.load(f)["num_classes"])

    df = pd.read_csv(args.data_root / "val" / "labels.csv")
    id_labels = df.loc[df["domain"] == "id", "y"].to_numpy()
    ood_labels = df.loc[df["domain"] == "ood", "y"].to_numpy()
    print(f"val: {len(id_labels)} id rows, {len(ood_labels)} ood rows, {num_classes} classes")

    classes = np.arange(num_classes)
    id_counts = pd.Series(id_labels).value_counts().reindex(classes, fill_value=0).to_numpy()
    ood_counts = pd.Series(ood_labels).value_counts().reindex(classes, fill_value=0).to_numpy()
    id_prop = id_counts / id_counts.sum()
    ood_prop = ood_counts / ood_counts.sum()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_abs, ax_prop) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    width = 0.4
    ax_abs.bar(classes - width / 2, id_counts, width=width, color=ID_COLOR,
               label=f"id (n={len(id_labels)})")
    ax_abs.bar(classes + width / 2, ood_counts, width=width, color=OOD_COLOR,
               label=f"ood (n={len(ood_labels)})")
    ax_abs.set_ylabel("count")
    ax_abs.set_title("Class distribution in val — absolute counts")
    ax_abs.legend()

    ax_prop.bar(classes - width / 2, id_prop, width=width, color=ID_COLOR, label="id")
    ax_prop.bar(classes + width / 2, ood_prop, width=width, color=OOD_COLOR, label="ood")
    ax_prop.set_ylabel("proportion within domain")
    ax_prop.set_xlabel("class")
    ax_prop.set_title("Class distribution in val — proportional (normalized within domain)")
    ax_prop.legend()
    ax_prop.set_xticks(classes)
    ax_prop.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")

    # Quick numeric summary: classes present in one domain but not the other,
    # and total variation distance between the two proportional distributions.
    id_only = classes[(id_counts > 0) & (ood_counts == 0)]
    ood_only = classes[(ood_counts > 0) & (id_counts == 0)]
    tv_distance = 0.5 * np.abs(id_prop - ood_prop).sum()
    print(f"classes present in id only: {list(id_only)}")
    print(f"classes present in ood only: {list(ood_only)}")
    print(f"total variation distance between id/ood class distributions: {tv_distance:.4f}")


if __name__ == "__main__":
    main()
