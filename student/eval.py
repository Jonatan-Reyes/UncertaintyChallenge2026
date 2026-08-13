"""Local evaluator: load a checkpoint, run on val, report the four metrics + accuracy.

Output matches what the master evaluator computes on the server, so use this
to iterate locally before submitting.

Checkpoint format expected from ``student.train``::

    torch.save({
        "state_dict":     model.state_dict(),
        "num_classes":    K,
        "temperatures":   [T_0, ..., T_4],  # one per backbone, 1.0 means no scaling
        "backbone_names": [...],            # timm model id per ensemble member
    }, path)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from student.data import IWildCamChallengeDataset, default_eval_transform
from student.metrics import compute_all_metrics
from student.model import DEFAULT_BACKBONES, Classifier
from student.plotting import energy_score, plot_energy_ood_roc, plot_reliability_diagram


def load_checkpoint(ckpt_path: Path, device) -> tuple[nn.Module, list]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone_names = ckpt.get("backbone_names", DEFAULT_BACKBONES)
    hparams = ckpt.get("hyperparameters", {})
    model = Classifier(
        int(ckpt["num_classes"]),
        backbone_names=backbone_names,
        heads_per_backbone=hparams.get("heads_per_backbone", 2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    temperatures = ckpt.get("temperatures", [1.0] * len(backbone_names))
    return model, [float(t) for t in temperatures]


def tta_predict(model: nn.Module, imgs: torch.Tensor, temperatures: list | None = None) -> torch.Tensor:
    """Test-time augmentation: average softmax over several views (original,
    flip, rotations, greyscale, solarize)."""
    views = (
        imgs,
        torch.flip(imgs, dims=[3]),
        TF.rotate(imgs, 15),
        TF.rotate(imgs, -15),
        TF.rgb_to_grayscale(imgs, num_output_channels=3),
        TF.solarize(imgs, threshold=0.5),
    )
    probs = sum(torch.softmax(model(v, temperatures=temperatures), dim=1) for v in views)
    return probs / len(views)


def collect_predictions(
    model: nn.Module, loader: DataLoader, device, temperatures: list | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` (with TTA, and each backbone's own
    temperature applied before combining, if given) and return
    ``(probs, labels)`` as np arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = tta_predict(model, imgs, temperatures)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(np.asarray(labels))
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def evaluate(
    model: nn.Module, loader: DataLoader, device, temperatures: list | None = None
) -> dict:
    probs, labels = collect_predictions(model, loader, device, temperatures)
    return compute_all_metrics(probs, labels)


def collect_energy(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    """Ensemble-averaged energy score per sample (see ``student.plotting.energy_score``).

    Uses ``forward_members`` for raw per-member logits — ``model.forward``'s
    probability-averaged output always sums to 1, which would make the energy
    score trivially constant.
    """
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            member_logits = model.forward_members(imgs).cpu().numpy()
            chunks.append(energy_score(member_logits))
    return np.concatenate(chunks, axis=0)


def evaluate_val_by_domain(
    model: nn.Module, val_ds: IWildCamChallengeDataset, device,
    temperatures: list | None = None, batch_size: int = 32, num_workers: int = 4,
    output_dir: Path | None = None,
) -> dict:
    """Evaluate the val split as a whole, plus split by domain (id vs. ood).

    If ``output_dir`` is given, also saves a reliability diagram and an
    energy-based OOD-detection ROC curve (id vs. ood) to ``output_dir``.
    """
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, labels = collect_predictions(model, loader, device, temperatures)

    domains = np.asarray(val_ds.domains)
    id_mask = domains == "id"
    ood_mask = domains == "ood"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability_diagram(probs, labels, output_dir / "reliability_diagram.png")
        energy = collect_energy(model, loader, device)
        plot_energy_ood_roc(energy[id_mask], energy[ood_mask], output_dir / "energy_ood_roc.png")

    return {
        "overall": compute_all_metrics(probs, labels),
        "id": compute_all_metrics(probs[id_mask], labels[id_mask]),
        "ood": compute_all_metrics(probs[ood_mask], labels[ood_mask]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the val split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="If set, write val_metrics.json here (e.g. the training run's output dir).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, temperatures = load_checkpoint(args.checkpoint, device)
    val_ds = IWildCamChallengeDataset(args.data_root, "val", default_eval_transform())
    metrics = evaluate_val_by_domain(
        model, val_ds, device, temperatures,
        batch_size=args.batch_size, num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.output_dir / 'val_metrics.json'}")


if __name__ == "__main__":
    main()
