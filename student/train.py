"""Trainer for a 5-foundation-model ensemble, with early stopping and
post-hoc temperature scaling.

Two checkpoints are written:

- ``model.pt``              — best weights (early-stopped on val NLL), ``T = 1.0``
- ``model_temp_scaled.pt``  — same weights, with scalar ``T`` learned on val

Things to modify:
- ``Trainer.train_epoch``    — your optimization step / loss
- ``Trainer.evaluate_val``   — your validation metric (default: NLL)
- ``fit_temperature``        — swap NLL for another calibration objective
- argparse defaults once you've validated the pipeline
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from student import metrics as M
from student.data import (
    IWildCamChallengeDataset,
    default_eval_transform,
    default_train_transform,
)
from student.eval import evaluate_val_by_domain
from student.model import DEFAULT_BACKBONES, BrierLoss, Classifier, CombinedLoss, SoftECELoss
from student.plotting import energy_score, plot_energy_ood_roc, plot_reliability_diagram


def make_optimizer(
    model: Classifier, lr_backbone: float, lr_head: float, weight_decay: float
) -> optim.Optimizer:
    """AdamW with two param groups: each backbone's unfrozen last-2-layers
    (lower ``lr_backbone``, since they're pretrained) and the heads (higher
    ``lr_head``, freshly initialized)."""
    backbone_params = [p for bb in model.backbones for p in bb.parameters() if p.requires_grad]
    return optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
    ])


HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({"accuracy"})


@dataclass
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Optimizer
    criteria: list  # cycled to match Classifier.iter_member_logits' per-head order
    device: torch.device
    patience: int = 3
    early_stop_metric: str = "accuracy"

    def __post_init__(self) -> None:
        if self.early_stop_metric in HIGHER_IS_BETTER_METRICS:
            self.best_val_metric: float = float("-inf")
        else:
            self.best_val_metric = float("inf")
        self.best_state_dict: dict | None = None
        self.epochs_no_improve: int = 0

    def _is_improved(self, value: float) -> bool:
        if self.early_stop_metric in HIGHER_IS_BETTER_METRICS:
            return value > self.best_val_metric
        return value < self.best_val_metric

    def train_epoch(self) -> tuple[float, float]:
        """Each member (one backbone's one head) is trained on its own loss,
        not on the loss of the ensemble's averaged prediction — so members
        fit the data independently instead of being pulled toward agreement.
        Within each backbone, its ``heads_per_backbone`` heads each get a
        *different* loss from ``self.criteria`` (NLL, Brier, ECE), cycling
        to match ``iter_member_logits``'s per-backbone-then-per-head order.
        Each member's loss is backpropagated immediately (gradients
        accumulate across all ``backward()`` calls before one
        ``optimizer.step()``), so only one member's activations are ever in
        memory at once. Gradients land on that member's own head and its
        backbone's unfrozen last-2-layers; the rest of each backbone stays
        fixed.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for imgs, labels in tqdm(self.train_loader, desc="train", leave=False):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            loss_sum = 0.0
            member_logits = []
            for logits_k, criterion_k in zip(self.model.iter_member_logits(imgs), itertools.cycle(self.criteria)):
                loss_k = criterion_k(logits_k, labels)
                loss_k.backward()
                loss_sum += loss_k.item()
                member_logits.append(logits_k.detach())
            self.optimizer.step()
            ensemble_logits = torch.stack(member_logits, dim=0).mean(dim=0)
            total_loss += (loss_sum / len(member_logits)) * imgs.size(0)
            total_correct += int((ensemble_logits.argmax(dim=1) == labels).sum().item())
            total += imgs.size(0)
        return total_loss / total, total_correct / total

    def evaluate_val(self, epoch: int, output_dir: Path | None = None) -> dict[str, dict[str, float]]:
        """Return ``{overall, id, ood}`` each with ``{nll, accuracy, brier}``.

        Uses ``student.metrics`` so the numbers students see during training
        are computed the same way as the master evaluator's scoring. If
        ``output_dir`` is given, also plots the reliability diagram and
        energy-based OOD ROC for this epoch.
        """
        self.model.eval()
        probs_chunks: list[np.ndarray] = []
        labels_chunks: list[np.ndarray] = []
        energy_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                probs = torch.softmax(logits, dim=1)
                probs_chunks.append(probs.cpu().numpy())
                labels_chunks.append(np.asarray(labels))
                if output_dir is not None:
                    member_logits = self.model.forward_members(imgs).cpu().numpy()
                    energy_chunks.append(energy_score(member_logits))
        probs = np.concatenate(probs_chunks)
        labels = np.concatenate(labels_chunks)
        domains = np.asarray(self.val_loader.dataset.domains)
        id_mask = domains == "id"
        ood_mask = domains == "ood"

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_reliability_diagram(probs, labels, output_dir / f"reliability_diagram_epoch{epoch:03d}.png")
            energy = np.concatenate(energy_chunks)
            plot_energy_ood_roc(energy[id_mask], energy[ood_mask], output_dir / f"energy_ood_roc_epoch{epoch:03d}.png")

        def _subset(mask: np.ndarray) -> dict[str, float]:
            return {
                "nll": M.nll(probs[mask], labels[mask]),
                "accuracy": M.accuracy(probs[mask], labels[mask]),
                "brier": M.brier(probs[mask], labels[mask]),
                "ece": M.ece(probs[mask], labels[mask]),
            }

        return {
            "overall": _subset(np.ones(len(labels), dtype=bool)),
            "id": _subset(id_mask),
            "ood": _subset(ood_mask),
        }

    def fit(self, epochs: int, output_dir: Path | None = None) -> None:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val = self.evaluate_val(epoch, output_dir)
            print(
                f"epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_nll={val['overall']['nll']:.4f} val_acc={val['overall']['accuracy']:.4f} "
                f"val_brier={val['overall']['brier']:.4f} val_ece={val['overall']['ece']:.4f} | "
                f"id_acc={val['id']['accuracy']:.4f} id_nll={val['id']['nll']:.4f} "
                f"id_brier={val['id']['brier']:.4f} id_ece={val['id']['ece']:.4f} | "
                f"ood_acc={val['ood']['accuracy']:.4f} ood_nll={val['ood']['nll']:.4f} "
                f"ood_brier={val['ood']['brier']:.4f} ood_ece={val['ood']['ece']:.4f}"
            )
            current = val["overall"][self.early_stop_metric]
            if self._is_improved(current):
                self.best_val_metric = current
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"early stopping at epoch {epoch} "
                          f"(no improvement in val/{self.early_stop_metric} for {self.patience} epochs)")
                    break

            if output_dir is not None:
                save_checkpoint(self.model, self.model.num_classes, 1.0, Path(output_dir) / "model.pt")

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Optimize a single scalar ``T`` to minimize NLL on ``(logits, labels)``.

    Parameterized as ``T = exp(log_T)`` so the optimizer stays in (0, ∞)
    without bounds constraints. Falls back to ``T = 1.0`` if LBFGS diverges
    (e.g. on a tiny val set against a barely-trained model).
    """
    log_T = nn.Parameter(torch.zeros(1, device=logits.device))
    optimizer = optim.LBFGS([log_T], lr=0.1, max_iter=100)
    criterion = BrierLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(log_T.exp().detach().cpu())
    if not (T > 0 and T < float("inf")):
        return 1.0
    return T


def temperature_scale(model: nn.Module, val_loader: DataLoader, device) -> float:
    """Collect val logits, then fit a single scalar T against NLL."""
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits_list.append(model(imgs))
            labels_list.append(labels.to(device))
    return fit_temperature(torch.cat(logits_list), torch.cat(labels_list))


def save_checkpoint(
    model: nn.Module,
    num_classes: int,
    temperature: float,
    path: Path,
    hyperparameters: dict | None = None,
) -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "num_classes": int(num_classes),
        "temperature": float(temperature),
        "backbone_names": getattr(model, "backbone_names", DEFAULT_BACKBONES),
    }
    if hyperparameters is not None:
        ckpt["hyperparameters"] = dict(hyperparameters)
    torch.save(ckpt, path)


def train(
    data_root: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    num_workers: int = 4,
    backbones: list[str] | None = None,
    pretrained: bool = False,
    num_unfrozen_layers: int = 2,
    alpha: float = 0.5,
    early_stop_metric: str = "accuracy",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(device)

    backbones = list(backbones) if backbones else list(DEFAULT_BACKBONES)
    # 2 heads per backbone: both cross-entropy, each plus an alpha-weighted
    # secondary calibration term -- order must match this list.
    criteria = [
        CombinedLoss(nn.CrossEntropyLoss(), BrierLoss(), alpha),
        CombinedLoss(nn.CrossEntropyLoss(), SoftECELoss(), alpha),
    ]

    hparams = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "head_lr": float(head_lr),
        "weight_decay": float(weight_decay),
        "patience": int(patience),
        "num_workers": int(num_workers),
        "backbones": backbones,
        "pretrained": bool(pretrained),
        "num_unfrozen_layers": int(num_unfrozen_layers),
        "heads_per_backbone": len(criteria),
        "alpha": float(alpha),
        "early_stop_metric": str(early_stop_metric),
        "data_root": str(data_root),
    }
    (output_dir / "config.json").write_text(json.dumps(hparams, indent=2))

    train_ds = IWildCamChallengeDataset(data_root, "train", default_train_transform())
    val_ds = IWildCamChallengeDataset(data_root, "val", default_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    model = Classifier(
        train_ds.num_classes, backbone_names=backbones, pretrained=pretrained,
        num_unfrozen_layers=num_unfrozen_layers, heads_per_backbone=len(criteria),
    ).to(device)
    optimizer = make_optimizer(model, lr_backbone=lr, lr_head=head_lr, weight_decay=weight_decay)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criteria=criteria, device=device,
        patience=patience, early_stop_metric=early_stop_metric,
    )
    trainer.fit(epochs, output_dir=output_dir)

    save_checkpoint(model, train_ds.num_classes, 1.0, output_dir / "model.pt", hyperparameters=hparams)
    print("saved model.pt")

    T = temperature_scale(model, val_loader, device)
    save_checkpoint(model, train_ds.num_classes, T, output_dir / "model_temp_scaled.pt", hyperparameters=hparams)
    print(f"learned T={T:.4f} → saved model_temp_scaled.pt")

    val_metrics = evaluate_val_by_domain(
        model, val_ds, device, T, batch_size=batch_size, num_workers=num_workers,
        output_dir=output_dir,
    )
    (output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    print(f"wrote {output_dir / 'val_metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 5-foundation-model ensemble.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save model.pt and model_temp_scaled.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for each backbone's unfrozen last-2-layers.")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="Learning rate for the linear heads.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stop after this many epochs without val-metric improvement.")
    parser.add_argument("--early-stop-metric", type=str, default="accuracy",
                        choices=["accuracy", "nll", "brier"],
                        help="Which val metric drives early stopping. Default is "
                             "accuracy: we teach accuracy-first, calibration-second.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbones", type=str, nargs="+", default=list(DEFAULT_BACKBONES),
                        help="timm model ids, one per ensemble member (default: 5 different foundation models).")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialize every backbone from timm's pretrained weights.")
    parser.add_argument("--num-unfrozen-layers", type=int, default=2,
                        help="Number of layers to unfreeze/fine-tune at the end of each backbone.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on each head's secondary calibration loss "
                             "(cross-entropy + alpha * Brier/ECE).")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
