"""Online trainer: frozen backbone, LoRA fine-tune, or full fine-tune,
with early stopping and post-hoc temperature scaling.

Two checkpoints are written:

- ``model.pt``              — best weights (early-stopped on val metric), ``T = 1.0``
- ``model_temp_scaled.pt``  — same weights, with scalar ``T`` learned on val

Things to modify:
- ``Trainer.train_epoch``    — your optimization step / loss
- ``Trainer.evaluate_val``   — your validation metric (default: accuracy)
- ``fit_temperature``        — swap NLL for another calibration objective
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from student import metrics as M
from student.data import (
    IMG_SIZE,
    IWildCamChallengeDataset,
    TRAIN_RECIPES,
    default_eval_transform,
    resolve_norm,
    train_transform,
)
from student.model import DEFAULT_BACKBONE, Classifier


def freeze_backbone(model: Classifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad_(False)


def _split_decay(named_params) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split (name, param) pairs into (decay, no_decay).

    Convention: 1-D parameters (biases, BatchNorm scale/shift, LayerNorm)
    skip weight decay; 2-D+ weight matrices get it. Matches the Karpathy /
    nanoGPT recipe.
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
    """AdamW with four param groups: backbone/head x decay/no_decay.

    Frozen params (``requires_grad=False``, e.g. a frozen probe or the base
    weights under a LoRA wrapper) are excluded by ``_split_decay``.
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


def soft_target_loss(
    logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module,
    weights: torch.Tensor | None = None,
    class_sim_target: torch.Tensor | None = None,
    class_smooth_alpha: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Plain CE, or a soft target built from up to two independent mixes:

    - ``weights`` (per-sample similarity-to-"nothing" above the soft-relabel
      gate): ``target = (1-w)*onehot(y) + w*onehot(0)``. ``w == 0``
      (unflagged rows) leaves onehot(y) untouched.
    - ``class_smooth_alpha`` (CLIP class-similarity smoothing, see
      preprocessing/build_class_similarity.py): blends a further
      ``class_smooth_alpha`` of the target's mass into
      ``class_sim_target[y]`` (a softmax-normalized similarity row, most of
      its own mass already on class ``y``), so confusions between
      semantically close classes are penalized less than confusions
      between unrelated ones. ``alpha == 0`` leaves the target untouched.

    Both are off by default, so this reduces exactly to ``criterion``
    (plain/label-smoothed CE) — a strict superset, not a separate regime.
    Shared between the single-GPU ``Trainer`` and DDP ``DDPTrainer`` so the
    two training paths can't drift apart on loss semantics."""
    if weights is None and class_smooth_alpha == 0.0:
        return criterion(logits, labels)
    num_classes = logits.size(1)
    target = F.one_hot(labels, num_classes).float()
    if weights is not None:
        target = target * (1.0 - weights).unsqueeze(1)
        target[:, 0] += weights
    if class_smooth_alpha > 0.0:
        sim_target = class_sim_target[labels]
        target = (1.0 - class_smooth_alpha) * target + class_smooth_alpha * sim_target
    if label_smoothing > 0:
        target = target * (1.0 - label_smoothing) + label_smoothing / num_classes
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


@dataclass
class Trainer:
    model: Classifier
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    criterion: nn.Module
    device: torch.device
    patience: int = 3
    early_stop_metric: str = "accuracy"
    label_smoothing: float = 0.0
    class_sim_target: torch.Tensor | None = None
    class_smooth_alpha: float = 0.0

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

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor,
              weights: torch.Tensor | None) -> torch.Tensor:
        return soft_target_loss(
            logits, labels, self.criterion, weights,
            self.class_sim_target, self.class_smooth_alpha, self.label_smoothing,
        )

    def _batch_loss(self, imgs: torch.Tensor, labels: torch.Tensor,
                     weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, logits) where logits are the mean-over-heads
        prediction used for the accuracy readout. With ``num_heads > 1`` each
        head gets its own CE term (jointly trained, own init + dropout mask)
        rather than training on the pre-averaged logits.

        Summed, not averaged, across heads: dividing by num_heads scales
        every head's own gradient down by 1/K relative to training it alone,
        starving each head of signal and making early stopping trigger
        before any head is actually converged. The per-head terms are
        independent (disjoint parameters), so summing instead of averaging
        doesn't change what each head learns -- only how hard it's pushed
        per step, which should match the single-head case.
        """
        if self.model.num_heads == 1:
            logits = self.model(imgs)
            return self._loss(logits, labels, weights), logits
        head_logits = self.model.forward_heads(imgs)  # (N, K, C)
        loss = sum(self._loss(head_logits[:, k], labels, weights) for k in range(self.model.num_heads))
        return loss, head_logits.mean(dim=1)

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        amp_enabled = self.device.type == "cuda"
        for batch in tqdm(self.train_loader, desc="train", leave=False):
            if len(batch) == 3:
                imgs, labels, weights = batch
                weights = weights.to(self.device).float()
            else:
                imgs, labels = batch
                weights = None
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                loss, logits = self._batch_loss(imgs, labels, weights)
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
        amp_enabled = self.device.type == "cuda"
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logits = self.model(imgs)
                probs = torch.softmax(logits.float(), dim=1)
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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor,
    t_min: float = 0.5, t_max: float = 5.0, n_grid: int = 91,
) -> float:
    """Grid-search a single scalar ``T``, scored by summed rank over
    ECE + NLL + Brier (a Borda vote restricted to the metrics ``T`` can
    actually move).

    ``T`` is a monotonic rescaling of the logits, so it never changes the
    argmax or the confidence ranking — accuracy and misclassification AUROC
    are invariant to it. That leaves ECE/NLL/Brier as the only Borda
    components a temperature search can affect, so ranking the grid on
    those three (rather than NLL alone, or an arbitrarily-weighted
    NLL+ECE combination) is the whole sub-problem, not an approximation of
    it. ECE is non-differentiable (binned), which rules out LBFGS in favor
    of a grid in the first place.
    """
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    grid = np.geomspace(t_min, t_max, n_grid)

    ece_vals = np.empty(n_grid)
    nll_vals = np.empty(n_grid)
    brier_vals = np.empty(n_grid)
    for i, T in enumerate(grid):
        probs = _softmax_np(logits_np / T)
        ece_vals[i] = M.ece(probs, labels_np)
        nll_vals[i] = M.nll(probs, labels_np)
        brier_vals[i] = M.brier(probs, labels_np)

    def ranks(x: np.ndarray) -> np.ndarray:
        order = x.argsort()
        r = np.empty_like(order)
        r[order] = np.arange(len(x))
        return r

    total_rank = ranks(ece_vals) + ranks(nll_vals) + ranks(brier_vals)
    return float(grid[int(total_rank.argmin())])


def temperature_scale(model: nn.Module, val_loader: DataLoader, device) -> float:
    """Collect val logits, then fit a single scalar T against NLL."""
    model.eval()
    amp_enabled = device.type == "cuda"
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(imgs)
            logits_list.append(logits.float())
            labels_list.append(labels.to(device))
    return fit_temperature(torch.cat(logits_list), torch.cat(labels_list))


def save_checkpoint(
    model: Classifier,
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
    pooling = getattr(model, "pooling", None)
    if pooling is not None:
        ckpt["pooling"] = pooling
    ckpt["num_heads"] = int(getattr(model, "num_heads", 1))
    ckpt["lora"] = getattr(model, "lora_cfg", None)
    patch_embed = getattr(model.backbone, "patch_embed", None)
    img_size = getattr(patch_embed, "img_size", None)
    if img_size is not None:
        # (H, W) tuple; models used here are square.
        ckpt["img_size"] = int(img_size[0]) if isinstance(img_size, (tuple, list)) else int(img_size)
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
    early_stop_metric: str = "accuracy",
    pooling: str | None = None,
    img_size: int | None = None,
    aug: str = "light",
    label_smoothing: float = 0.0,
    init_head_from: Path | None = None,
    freeze_backbone_flag: bool = False,
    num_heads: int = 1,
    head_dropout: float = 0.0,
    lora_r: int | None = None,
    lora_alpha: int = 16,
    lora_target_blocks: int = 8,
    grad_checkpointing: bool = False,
    soft_relabel_threshold: float | None = None,
    class_smooth_alpha: float = 0.0,
    class_smooth_temp: float = 0.07,
    class_smooth_path: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_img_size = img_size if img_size is not None else IMG_SIZE
    lora_cfg = {"r": lora_r, "alpha": lora_alpha, "target_blocks": lora_target_blocks} if lora_r else None

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
        "early_stop_metric": str(early_stop_metric),
        "pooling": pooling,
        "img_size": resolved_img_size,
        "aug": aug,
        "label_smoothing": float(label_smoothing),
        "data_root": str(data_root),
        "freeze_backbone": bool(freeze_backbone_flag),
        "num_heads": int(num_heads),
        "head_dropout": float(head_dropout),
        "lora": lora_cfg,
        "grad_checkpointing": bool(grad_checkpointing),
        "soft_relabel_threshold": soft_relabel_threshold,
        "class_smooth_alpha": float(class_smooth_alpha),
        "class_smooth_temp": float(class_smooth_temp),
    }
    (output_dir / "config.json").write_text(json.dumps(hparams, indent=2))

    mean, std = resolve_norm(backbone)
    eval_transform = default_eval_transform(resolved_img_size, mean, std)
    train_tf = train_transform(aug, resolved_img_size, mean, std)

    train_labels_filename = "labels_flagged.csv" if soft_relabel_threshold is not None else "labels.csv"
    train_ds = IWildCamChallengeDataset(
        data_root, "train", train_tf,
        labels_filename=train_labels_filename, soft_relabel_threshold=soft_relabel_threshold,
    )
    val_ds = IWildCamChallengeDataset(data_root, "val", eval_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    model = Classifier(
        train_ds.num_classes, backbone_name=backbone, pretrained=pretrained,
        pooling=pooling, img_size=img_size, num_heads=num_heads, head_dropout=head_dropout,
        lora=lora_cfg, grad_checkpointing=grad_checkpointing,
    ).to(device)
    if freeze_backbone_flag and lora_cfg is None:
        freeze_backbone(model)
    if init_head_from is not None:
        init_ckpt = torch.load(init_head_from, map_location=device, weights_only=False)
        head_state = {
            k[len("head."):]: v for k, v in init_ckpt["state_dict"].items() if k.startswith("head.")
        }
        model.head.load_state_dict(head_state)
        print(f"initialized head from {init_head_from}")
    optimizer = make_optimizer(model, lr_backbone=lr, lr_head=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    class_sim_target = None
    if class_smooth_alpha > 0.0:
        sim_path = class_smooth_path if class_smooth_path is not None else data_root / "class_similarity.npy"
        sim = np.load(sim_path)
        assert sim.shape == (train_ds.num_classes, train_ds.num_classes), \
            f"class_similarity matrix shape {sim.shape} != num_classes {train_ds.num_classes}"
        class_sim_target = torch.softmax(torch.from_numpy(sim).float() / class_smooth_temp, dim=1).to(device)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, patience=patience, early_stop_metric=early_stop_metric,
        label_smoothing=label_smoothing,
        class_sim_target=class_sim_target, class_smooth_alpha=class_smooth_alpha,
    )
    trainer.fit(epochs)

    save_checkpoint(model, train_ds.num_classes, 1.0, output_dir / "model.pt", hyperparameters=hparams)
    print("saved model.pt")

    T = temperature_scale(model, val_loader, device)
    save_checkpoint(model, train_ds.num_classes, T, output_dir / "model_temp_scaled.pt", hyperparameters=hparams)
    print(f"learned T={T:.4f} -> saved model_temp_scaled.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier online (frozen / LoRA / full fine-tune).")
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
                        help="timm model id (e.g. resnet50, vit_giant_patch14_reg4_dinov2, eva02_large_patch14_448...).")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialize the backbone from timm's pretrained weights.")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone_flag", action="store_true",
                        help="Freeze all backbone params (head-only / linear-probe training). "
                             "Ignored if --lora-r is set (LoRA already freezes the non-adapter weights).")
    parser.add_argument("--pooling", type=str, default=None, choices=["cls", "avg", "cls_avg", "gem", "attn"])
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override student.data.IMG_SIZE (needed for fixed-input-size backbones like DINOv2/v3).")
    parser.add_argument("--aug", type=str, default="light", choices=sorted(TRAIN_RECIPES),
                        help="Named training-augmentation recipe (see student/data.py TRAIN_RECIPES).")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--init-head-from", type=Path, default=None,
                        help="Warm-start the head from a previous checkpoint's head weights (LP->FT).")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="K independently-initialized linear heads, jointly trained (Phase 4).")
    parser.add_argument("--head-dropout", type=float, default=0.0,
                        help="Per-head dropout on the pooled features (only meaningful with --num-heads > 1). "
                             "On a frozen backbone the per-head problem is convex, so random init alone tends to "
                             "converge heads to near-duplicate solutions -- a per-head dropout mask forces genuine "
                             "diversity by fitting each head against a different random feature subset.")
    parser.add_argument("--lora-r", type=int, default=None,
                        help="Enable LoRA on the last --lora-target-blocks transformer blocks with this rank.")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-blocks", type=int, default=8)
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="Mandatory in practice for DINOv2-G/EVA02-L at native resolution — see PLAN.md memory table.")
    parser.add_argument("--soft-relabel-threshold", type=float, default=None,
                        help="Gate for empty-frame soft relabeling (see preprocessing/find_empty_frames.ipynb "
                             "and preprocessing/build_flagged_labels.py). Train rows with cos_to_nothing_knn "
                             "above this mix their target toward one-hot(class 0), scaled by that similarity; "
                             "rows at or below it (or unflagged) train normally. Requires "
                             "challenge_data/train/labels_flagged.csv to exist. Unset by default (vanilla CE).")
    parser.add_argument("--class-smooth-alpha", type=float, default=0.0,
                        help="Blend this much of each sample's target toward its class's CLIP "
                             "text-embedding similarity row (see preprocessing/build_class_similarity.py), "
                             "so confusions between semantically close classes are penalized less than "
                             "confusions between unrelated ones. 0.0 disables (default).")
    parser.add_argument("--class-smooth-temp", type=float, default=0.07,
                        help="Softmax temperature applied to the cosine-similarity row before blending "
                             "(lower = sharper, closer to onehot; only matters if --class-smooth-alpha > 0).")
    parser.add_argument("--class-smooth-path", type=Path, default=None,
                        help="Path to class_similarity.npy (default: <data-root>/class_similarity.npy).")
    args = parser.parse_args()
    kwargs = vars(args)
    train(**kwargs)


if __name__ == "__main__":
    main()
