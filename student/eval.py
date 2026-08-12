"""Local evaluator: load a checkpoint, run on val, report the four metrics + accuracy.

Output matches what the master evaluator computes on the server, so use this
to iterate locally before submitting.

Checkpoint format expected from ``student.train``::

    torch.save({
        "state_dict":       model.state_dict(),
        "num_classes":      K,
        "temperature":      T,            # 1.0 means no scaling
        "backbone":         backbone_name,  # timm model id
        "optimizer_states": [head_0_ivon_state, head_1_ivon_state, ...],
    }, path)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ivon
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from student.data import IWildCamChallengeDataset, default_eval_transform
from student.metrics import compute_all_metrics
from student.model import DEFAULT_BACKBONE, Classifier
from student.plotting import energy_score, plot_energy_ood_roc, plot_reliability_diagram


def load_checkpoint(ckpt_path: Path, device) -> tuple[nn.Module, list, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", DEFAULT_BACKBONE)
    hparams = ckpt.get("hyperparameters", {})
    model = Classifier(
        int(ckpt["num_classes"]),
        backbone_name=backbone_name,
        num_heads=hparams.get("num_heads", 4),
        head_hidden_dim=hparams.get("head_hidden_dim", 256),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    optimizers = [
        ivon.IVON(
            head.parameters(),
            lr=hparams.get("head_lr", 1e-3),
            ess=hparams.get("ess", 1),
            weight_decay=hparams.get("weight_decay", 1e-4),
        )
        for head in model.head
    ]
    opt_states = ckpt.get("optimizer_states")
    if opt_states is not None:
        for optimizer, state in zip(optimizers, opt_states):
            optimizer.load_state_dict(state)

    return model, optimizers, float(ckpt.get("temperature", 1.0))


def ivon_predict(
    model: nn.Module, optimizers: list, imgs: torch.Tensor,
    n_samples: int = 5, temperature: float = 1.0,
) -> torch.Tensor:
    """Average softmax over ``n_samples`` IVON posterior draws per head,
    *then* average across heads — posterior sampling happens before, and
    separately from, the ensemble combination.
    """
    feats = model.embed(imgs)
    member_probs = []
    for head, optimizer in zip(model.head, optimizers):
        probs_sum = 0.0
        for _ in range(n_samples):
            with optimizer.sampled_params(train=False):
                logits = head(feats)
            probs_sum = probs_sum + torch.softmax(logits / temperature, dim=1)
        member_probs.append(probs_sum / n_samples)
    return torch.stack(member_probs, dim=0).mean(dim=0)


def collect_predictions(
    model: nn.Module, optimizers: list, loader: DataLoader, device,
    temperature: float = 1.0, n_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` (with IVON posterior sampling) and
    return ``(probs, labels)`` as np arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = ivon_predict(model, optimizers, imgs, n_samples, temperature)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(np.asarray(labels))
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def evaluate(
    model: nn.Module, optimizers: list, loader: DataLoader, device,
    temperature: float = 1.0, n_samples: int = 5,
) -> dict:
    probs, labels = collect_predictions(model, optimizers, loader, device, temperature, n_samples)
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
    model: nn.Module, optimizers: list, val_ds: IWildCamChallengeDataset, device,
    temperature: float = 1.0, batch_size: int = 32, num_workers: int = 4,
    n_samples: int = 5, output_dir: Path | None = None,
) -> dict:
    """Evaluate the val split as a whole, plus split by domain (id vs. ood).

    If ``output_dir`` is given, also saves a reliability diagram and an
    energy-based OOD-detection ROC curve (id vs. ood) to ``output_dir``.
    """
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, labels = collect_predictions(model, optimizers, loader, device, temperature, n_samples)

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
    parser.add_argument("--posterior-samples", type=int, default=5,
                        help="IVON posterior weight samples drawn per head at eval time.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="If set, write val_metrics.json here (e.g. the training run's output dir).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizers, temperature = load_checkpoint(args.checkpoint, device)
    val_ds = IWildCamChallengeDataset(args.data_root, "val", default_eval_transform())
    metrics = evaluate_val_by_domain(
        model, optimizers, val_ds, device, temperature,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_samples=args.posterior_samples, output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.output_dir / 'val_metrics.json'}")


if __name__ == "__main__":
    main()
