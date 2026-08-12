"""ResNet-50 baseline trainer with early stopping and post-hoc temperature scaling.

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
from student.eval import evaluate_val_by_domain, tta_predict
from student.model import DEFAULT_BACKBONE, Classifier, SoftECELoss
from student.plotting import energy_score, plot_energy_ood_roc, plot_reliability_diagram


def _split_decay(named_params) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split (name, param) pairs into (decay, no_decay).

    Convention: 1-D parameters (biases, BatchNorm scale/shift) skip weight
    decay; 2-D+ weight matrices get it. Matches the Karpathy / nanoGPT recipe.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def make_optimizer(
    model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float
) -> optim.Optimizer:
    """AdamW with four param groups: backbone/head × decay/no_decay.

    - Backbone gets the lower ``lr_backbone`` (pretrained weights).
    - Head gets the higher ``lr_head`` (new classification layer).
    - Biases and BatchNorm scale/shift parameters are excluded from weight
      decay; weight matrices receive it.
    """
    head_param_ids = {id(p) for p in model.head.parameters()}
    backbone_named = [(n, p) for n, p in model.named_parameters() if id(p) not in head_param_ids]
    head_named = [(n, p) for n, p in model.named_parameters() if id(p) in head_param_ids]

    bb_decay, bb_no_decay = _split_decay(backbone_named)
    hd_decay, hd_no_decay = _split_decay(head_named)

    return optim.AdamW([
        {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay},
        {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0},
    ])


HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({"accuracy"})


@dataclass
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    criterion: nn.Module
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
        """Each ensemble member is trained on its own cross-entropy loss, not on
        the loss of their averaged prediction — so members fit the data
        independently instead of being pulled toward agreement. Each member's
        loss is backpropagated immediately (gradients accumulate across the
        K ``backward()`` calls before one ``optimizer.step()``), so only one
        member's activations are ever in memory at once — no need to hold all
        K forward passes (and no backbone duplication) simultaneously. With a
        single member (no LoRA ensemble) this reduces to plain single-model
        training.
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
            for logits_k in self.model.iter_member_logits(imgs):
                loss_k = self.criterion(logits_k, labels)
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
                probs = tta_predict(self.model, imgs)
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
            self.scheduler.step()
            print(
                f"epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_nll={val['overall']['nll']:.4f} val_acc={val['overall']['accuracy']:.4f} "
                f"val_brier={val['overall']['brier']:.4f} | "
                f"id_acc={val['id']['accuracy']:.4f} id_nll={val['id']['nll']:.4f} "
                f"id_brier={val['id']['brier']:.4f} | "
                f"ood_acc={val['ood']['accuracy']:.4f} ood_nll={val['ood']['nll']:.4f} "
                f"ood_brier={val['ood']['brier']:.4f}"
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
    criterion = SoftECELoss()

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
        "backbone": getattr(model, "backbone_name", DEFAULT_BACKBONE),
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
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    num_lora_members: int = 4,
    early_stop_metric: str = "accuracy",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(device)
    hparams = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "head_lr": float(head_lr),
        "weight_decay": float(weight_decay),
        "patience": int(patience),
        "num_workers": int(num_workers),
        "backbone": str(backbone),
        "pretrained": bool(pretrained),
        "lora_r": int(lora_r),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "num_lora_members": int(num_lora_members),
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
        train_ds.num_classes, backbone_name=backbone, pretrained=pretrained,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        num_lora_members=num_lora_members,
    ).to(device)
    optimizer = make_optimizer(model, lr_backbone=lr, lr_head=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, patience=patience, early_stop_metric=early_stop_metric,
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
    parser = argparse.ArgumentParser(description="Train a ResNet-50 baseline.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save model.pt and model_temp_scaled.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Backbone learning rate (lower LR for pretrained weights).")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="Head learning rate (higher LR for the new classification layer).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stop after this many epochs without val-metric improvement.")
    parser.add_argument("--early-stop-metric", type=str, default="accuracy",
                        choices=["accuracy", "nll", "brier"],
                        help="Which val metric drives early stopping. Default is "
                             "accuracy: we teach accuracy-first, calibration-second.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        help="timm model id (e.g. resnet50, resnet18, convnext_small, vit_base_patch16_224).")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialize the backbone from timm's pretrained weights.")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank for backbone adapters; 0 disables LoRA (full backbone fine-tuning).")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA scaling factor (applied update is scaled by alpha/r).")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="Dropout applied inside the LoRA adapters.")
    parser.add_argument("--num-lora-members", type=int, default=4,
                        help="Ensemble size: number of independent LoRA adapters (each with its own head).")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
