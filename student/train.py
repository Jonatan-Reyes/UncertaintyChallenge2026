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
import optuna
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
    get_num_classes,
)
from student.model import DEFAULT_BACKBONE, Classifier, LoRAClassifier


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
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        expected_hw = getattr(self.model, "input_size", None)
        for imgs, labels in tqdm(self.train_loader, desc="train", leave=False):
            if expected_hw is not None:
                h, w = int(imgs.shape[-2]), int(imgs.shape[-1])
                if (h, w) != tuple(expected_hw):
                    raise ValueError(
                        "Input batch size mismatch: got "
                        f"{(h, w)} but backbone expects {tuple(expected_hw)}. "
                        "Use backbone-aware transforms in student.data."
                    )
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += imgs.size(0)
        return total_loss / total, total_correct / total

    def evaluate_val(self) -> dict[str, float]:
        """Return ``{nll, accuracy, brier}`` on the val loader.

        Uses ``student.metrics`` so the numbers students see during training
        are computed the same way as the master evaluator's scoring.
        """
        self.model.eval()
        probs_chunks: list[np.ndarray] = []
        labels_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                probs = torch.softmax(logits, dim=1)
                probs_chunks.append(probs.cpu().numpy())
                labels_chunks.append(np.asarray(labels))
        probs = np.concatenate(probs_chunks)
        labels = np.concatenate(labels_chunks)
        return {
            "nll": M.nll(probs, labels),
            "accuracy": M.accuracy(probs, labels),
            "brier": M.brier(probs, labels),
        }

    def fit(self, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val = self.evaluate_val()
            self.scheduler.step()
            print(
                f"epoch {epoch:3d} | train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} | val_nll={val['nll']:.4f} "
                f"val_acc={val['accuracy']:.4f} val_brier={val['brier']:.4f}"
            )
            current = val[self.early_stop_metric]
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


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "nll",
    n_trials: int = 50,
    seed: int | None = None,
) -> float:
    """Fit a single scalar temperature on ``(logits, labels)``.

    ``metric`` can be ``"nll"`` (default, differentiable via LBFGS) or
    ``"ece"`` (non-differentiable, optimized with Optuna). ``n_trials`` is only
    used for the ECE path.
    """
    metric = metric.lower()
    if metric == "nll":
        log_T = nn.Parameter(torch.zeros(1, device=logits.device))
        optimizer = optim.LBFGS([log_T], lr=0.1, max_iter=100)
        criterion = nn.CrossEntropyLoss()

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

    if metric == "ece":
        if n_trials <= 0:
            n_trials = 1
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            T = trial.suggest_float("T", 0.1, 10.0, log=True)
            probs = torch.softmax(logits / T, dim=1).detach().cpu().numpy()
            return float(M.ece(probs, labels.cpu().numpy()))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        T = float(study.best_trial.params["T"])
        if not (T > 0 and T < float("inf")):
            return 1.0
        return T

    raise ValueError(f"Unsupported temperature metric: {metric!r}; choose 'nll' or 'ece'.")


def temperature_scale(
    model: nn.Module,
    val_loader: DataLoader,
    device,
    metric: str = "nll",
    n_trials: int = 50,
    seed: int | None = None,
) -> float:
    """Collect val logits, then fit a scalar temperature with the chosen objective."""
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits_list.append(model(imgs))
            labels_list.append(labels.to(device))
    return fit_temperature(
        torch.cat(logits_list),
        torch.cat(labels_list),
        metric=metric,
        n_trials=n_trials,
        seed=seed,
    )


def save_checkpoint(
    model: nn.Module,
    num_classes: int,
    temperature: float,
    path: Path,
    hyperparameters: dict | None = None,
) -> None:
    if isinstance(model, LoRAClassifier):
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        meta = {
            "num_classes": int(num_classes),
            "temperature": float(temperature),
            "backbone": getattr(model, "backbone_name", DEFAULT_BACKBONE),
            "lora": getattr(model, "lora_config", None),
        }
        if hyperparameters is not None:
            meta["hyperparameters"] = dict(hyperparameters)
        (path / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))
        return

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
    device_idx: int | None = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
    modules_to_save: tuple[str, ...] = ("head",),
    early_stop_metric: str = "accuracy",
    temperature_metric: str = "nll",
    temperature_n_trials: int = 50,
    temperature_seed: int | None = None,
) -> None:
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx is not None else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = get_num_classes(data_root)

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
        "use_lora": bool(use_lora),
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_target_modules": list(lora_target_modules),
        "modules_to_save": list(modules_to_save),
        "early_stop_metric": str(early_stop_metric),
        "temperature_metric": str(temperature_metric),
        "temperature_n_trials": int(temperature_n_trials),
        "temperature_seed": None if temperature_seed is None else int(temperature_seed),
        "data_root": str(data_root),
    }
    (output_dir / "config.json").write_text(json.dumps(hparams, indent=2))

    if use_lora:
        model = LoRAClassifier(
            num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=tuple(lora_target_modules),
            modules_to_save=tuple(modules_to_save),
        ).to(device)
    else:
        model = Classifier(
            num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
        ).to(device)

    expected_hw = model.input_size if model.input_size is not None else (224, 224)
    print(f"backbone={backbone} expected_input_size={expected_hw}")

    train_ds = IWildCamChallengeDataset(data_root, "train", default_train_transform(expected_hw))
    val_ds = IWildCamChallengeDataset(data_root, "val", default_eval_transform(expected_hw))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    sample_img, _ = train_ds[0]
    sample_hw = tuple(sample_img.shape[-2:])
    if sample_hw != tuple(expected_hw):
        raise ValueError(
            "Transform/model mismatch before training: got sample size "
            f"{sample_hw}, expected {tuple(expected_hw)}"
        )
    optimizer = make_optimizer(model, lr_backbone=lr, lr_head=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, patience=patience, early_stop_metric=early_stop_metric,
    )
    trainer.fit(epochs)

    if use_lora:
        save_checkpoint(model, num_classes, 1.0, output_dir / "model_lora", hyperparameters=hparams)
        print("saved model_lora/")
    else:
        save_checkpoint(model, num_classes, 1.0, output_dir / "model.pt", hyperparameters=hparams)
        print("saved model.pt")

    T = temperature_scale(
        model,
        val_loader,
        device,
        metric=temperature_metric,
        n_trials=temperature_n_trials,
        seed=temperature_seed,
    )
    if use_lora:
        save_checkpoint(model, num_classes, T, output_dir / "model_temp_scaled_lora", hyperparameters=hparams)
        print(f"learned T={T:.4f} via {temperature_metric} -> saved model_temp_scaled_lora/")
    else:
        save_checkpoint(model, num_classes, T, output_dir / "model_temp_scaled.pt", hyperparameters=hparams)
        print(f"learned T={T:.4f} via {temperature_metric} -> saved model_temp_scaled.pt")


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
    parser.add_argument("--temperature-metric", type=str, default="nll",
                        choices=["nll", "ece"],
                        help="Objective used to fit the temperature scaling scalar on validation logits.")
    parser.add_argument("--temperature-n-trials", type=int, default=50,
                        help="Number of Optuna trials used when --temperature-metric=ece.")
    parser.add_argument("--temperature-seed", type=int, default=None,
                        help="Optional seed for the Optuna study when fitting temperature.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        help="timm model id (e.g. resnet50, resnet18, convnext_small, vit_base_patch16_224).")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialize the backbone from timm's pretrained weights.")
    parser.add_argument("--device-idx", type=int, default=None,
                        help="CUDA device index to use (default: None, which uses the first available device). ")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRAClassifier instead of the base Classifier.")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank; set >0 to enable PEFT LoRA on the backbone.")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA scaling alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout.")
    parser.add_argument("--lora-target-modules", nargs="+",
                        default=["qkv", "proj", "fc1", "fc2"],
                        help="Module name fragments to target with LoRA adapters.")
    parser.add_argument("--modules-to-save", nargs="+", default=["head"],
                        help="Extra modules to keep trainable/saved in LoRA checkpoints (e.g. head).")
    args = parser.parse_args()
    args.lora_target_modules = tuple(args.lora_target_modules)
    args.modules_to_save = tuple(args.modules_to_save)
    train(**vars(args))


if __name__ == "__main__":
    main()
