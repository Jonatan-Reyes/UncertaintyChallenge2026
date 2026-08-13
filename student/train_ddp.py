"""torchrun-launched DDP variant of student.train, for splitting one run's
batch across multiple GPUs instead of running independent single-GPU jobs.

Use this when a *specific* run's wall-clock is the bottleneck (e.g. a
DINOv2-G frozen-probe pass at ~45-75 min on one GPU); for sweeping many
independent configs, single-GPU student.train jobs in parallel are already
the better use of 8 GPUs (embarrassingly parallel, zero sync overhead).

Supports the same soft-target machinery as student.train (``--num-heads``,
``--soft-relabel-threshold``, ``--class-smooth-alpha``), via
``student.train.soft_target_loss`` shared between both trainers. K-heads
needs one DDP-specific trick: ``model.module.forward_heads()`` called
directly would bypass DDP's ``__call__`` hook (the hook that arms the
gradient-sync callbacks for the following ``backward()``) and silently
desync the ranks. Instead, ``Classifier._return_all_heads`` is toggled
around the ordinary ``model(imgs)`` call so the (N,K,C) per-head logits
still come back through ``DDP.forward`` -> ``self.module.forward(...)``,
the same call path DDP's hooks are armed against.

Launch:
    torchrun --nproc_per_node=<N> -m student.train_ddp \\
        --data-root challenge_data --output-dir runs/foo \\
        --backbone vit_giant_patch14_reg4_dinov2.lvd142m --pretrained \\
        --pooling cls --freeze-backbone --img-size 518 --aug light \\
        --epochs 15 --patience 4 --batch-size 96 --num-workers 8

``--batch-size`` is the per-GPU batch size (matches torchrun/DDP convention),
so the effective global batch is ``batch_size * nproc_per_node``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

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
from student.train import fit_temperature, freeze_backbone, make_optimizer, save_checkpoint, soft_target_loss

HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({"accuracy"})


def is_main_process() -> bool:
    return dist.get_rank() == 0


@dataclass
class DDPTrainer:
    model: DDP
    train_loader: DataLoader
    train_sampler: DistributedSampler
    val_loader: DataLoader
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    device: torch.device
    patience: int = 3
    early_stop_metric: str = "accuracy"
    label_smoothing: float = 0.0
    class_sim_target: torch.Tensor | None = None
    class_smooth_alpha: float = 0.0

    def __post_init__(self) -> None:
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
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
                     weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, logits) where logits are the mean-over-heads
        prediction. See module docstring for why K-heads goes through
        ``model(imgs)`` with ``_return_all_heads`` toggled rather than
        ``model.module.forward_heads(imgs)`` directly (as student.train
        does on a single GPU) -- the latter bypasses DDP's gradient-sync
        hooks and silently desyncs the ranks."""
        num_heads = self.model.module.num_heads
        if num_heads == 1:
            logits = self.model(imgs)
            return self._loss(logits, labels, weights), logits
        self.model.module._return_all_heads = True
        try:
            head_logits = self.model(imgs)  # (N, K, C), through DDP.__call__
        finally:
            self.model.module._return_all_heads = False
        loss = sum(self._loss(head_logits[:, k], labels, weights) for k in range(num_heads))
        return loss, head_logits.mean(dim=1)

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        self.train_sampler.set_epoch(epoch)  # reshuffles differently per epoch across ranks
        total_loss = 0.0
        total_correct = 0
        total = 0
        amp_enabled = self.device.type == "cuda"
        for batch in self.train_loader:
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
            loss.backward()  # DDP all-reduces gradients across ranks here
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += imgs.size(0)

        # each rank only trained on its own shard; all-reduce for an honest logged number
        stats = torch.tensor([total_loss, total_correct, total], device=self.device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total = stats.tolist()
        return total_loss / total, total_correct / total

    def evaluate_val(self) -> dict[str, float]:
        """Every rank evaluates the full (un-sharded) val set redundantly.

        Cheap here (918 images) and sidesteps needing an all-gather for
        ECE/AUROC, which aren't simple batch-averages of per-rank shards.
        Ranks stay bit-identical anyway, since DDP keeps gradients (and
        therefore weights) synced after every step -- so this redundant
        compute gives every rank the same number, not an approximation.
        """
        self.model.eval()
        self.model.module._return_all_heads = False  # model(imgs) -> mean-over-heads (N,C)
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
            train_loss, train_acc = self.train_epoch(epoch)
            val = self.evaluate_val()
            self.scheduler.step()
            if is_main_process():
                print(
                    f"epoch {epoch:3d} | train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} | val_nll={val['nll']:.4f} "
                    f"val_acc={val['accuracy']:.4f} val_brier={val['brier']:.4f}"
                )
            current = val[self.early_stop_metric]
            if self._is_improved(current):
                self.best_val_metric = current
                self.best_state_dict = copy.deepcopy(self.model.module.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    if is_main_process():
                        print(f"early stopping at epoch {epoch} "
                              f"(no improvement in val/{self.early_stop_metric} for {self.patience} epochs)")
                    break

        if self.best_state_dict is not None:
            self.model.module.load_state_dict(self.best_state_dict)


def temperature_scale_ddp(model: DDP, val_loader: DataLoader, device) -> float:
    model.eval()
    model.module._return_all_heads = False  # model(imgs) -> mean-over-heads (N,C)
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
    freeze_backbone_flag: bool = False,
    lora_r: int | None = None,
    lora_alpha: int = 16,
    lora_target_blocks: int = 8,
    grad_checkpointing: bool = False,
    num_heads: int = 1,
    head_dropout: float = 0.0,
    soft_relabel_threshold: float | None = None,
    class_smooth_alpha: float = 0.0,
    class_smooth_temp: float = 0.07,
    class_smooth_path: Path | None = None,
) -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    output_dir = Path(output_dir)
    resolved_img_size = img_size if img_size is not None else IMG_SIZE
    lora_cfg = {"r": lora_r, "alpha": lora_alpha, "target_blocks": lora_target_blocks} if lora_r else None

    hparams = {
        "epochs": int(epochs), "batch_size_per_gpu": int(batch_size),
        "world_size": dist.get_world_size(), "lr": float(lr), "head_lr": float(head_lr),
        "weight_decay": float(weight_decay), "patience": int(patience), "num_workers": int(num_workers),
        "backbone": str(backbone), "pretrained": bool(pretrained), "early_stop_metric": str(early_stop_metric),
        "pooling": pooling, "img_size": resolved_img_size, "aug": aug,
        "label_smoothing": float(label_smoothing), "data_root": str(data_root),
        "freeze_backbone": bool(freeze_backbone_flag), "lora": lora_cfg,
        "grad_checkpointing": bool(grad_checkpointing),
        "num_heads": int(num_heads), "head_dropout": float(head_dropout),
        "soft_relabel_threshold": soft_relabel_threshold,
        "class_smooth_alpha": float(class_smooth_alpha), "class_smooth_temp": float(class_smooth_temp),
    }
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(hparams, indent=2))
    dist.barrier()

    mean, std = resolve_norm(backbone)
    eval_transform = default_eval_transform(resolved_img_size, mean, std)
    train_tf = train_transform(aug, resolved_img_size, mean, std)

    train_labels_filename = "labels_flagged.csv" if soft_relabel_threshold is not None else "labels.csv"
    train_ds = IWildCamChallengeDataset(
        data_root, "train", train_tf,
        labels_filename=train_labels_filename, soft_relabel_threshold=soft_relabel_threshold,
    )
    val_ds = IWildCamChallengeDataset(data_root, "val", eval_transform)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = Classifier(
        train_ds.num_classes, backbone_name=backbone, pretrained=pretrained,
        pooling=pooling, img_size=img_size, num_heads=num_heads, head_dropout=head_dropout,
        lora=lora_cfg, grad_checkpointing=grad_checkpointing,
    ).to(device)
    if freeze_backbone_flag and lora_cfg is None:
        freeze_backbone(model)
    model = DDP(model, device_ids=[local_rank])

    optimizer = make_optimizer(model.module, lr_backbone=lr, lr_head=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    class_sim_target = None
    if class_smooth_alpha > 0.0:
        sim_path = class_smooth_path if class_smooth_path is not None else data_root / "class_similarity.npy"
        sim = np.load(sim_path)
        assert sim.shape == (train_ds.num_classes, train_ds.num_classes), \
            f"class_similarity matrix shape {sim.shape} != num_classes {train_ds.num_classes}"
        class_sim_target = torch.softmax(torch.from_numpy(sim).float() / class_smooth_temp, dim=1).to(device)

    trainer = DDPTrainer(
        model=model, train_loader=train_loader, train_sampler=train_sampler, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device, patience=patience,
        early_stop_metric=early_stop_metric, label_smoothing=label_smoothing,
        class_sim_target=class_sim_target, class_smooth_alpha=class_smooth_alpha,
    )
    trainer.fit(epochs)

    if is_main_process():
        save_checkpoint(model.module, train_ds.num_classes, 1.0, output_dir / "model.pt", hyperparameters=hparams)
        print("saved model.pt")
    dist.barrier()

    T = temperature_scale_ddp(model, val_loader, device)
    if is_main_process():
        save_checkpoint(model.module, train_ds.num_classes, T, output_dir / "model_temp_scaled.pt", hyperparameters=hparams)
        print(f"learned T={T:.4f} -> saved model_temp_scaled.pt")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP training via torchrun.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early-stop-metric", type=str, default="accuracy", choices=["accuracy", "nll", "brier"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone_flag", action="store_true")
    parser.add_argument("--pooling", type=str, default=None, choices=["cls", "avg", "cls_avg", "gem", "attn"])
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--aug", type=str, default="light", choices=sorted(TRAIN_RECIPES))
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-blocks", type=int, default=8)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="K independently-initialized linear heads, jointly trained (see module docstring "
                             "for the DDP-specific _return_all_heads plumbing).")
    parser.add_argument("--head-dropout", type=float, default=0.0,
                        help="Per-head dropout on the pooled features (only meaningful with --num-heads > 1).")
    parser.add_argument("--soft-relabel-threshold", type=float, default=None,
                        help="Gate for empty-frame soft relabeling (see preprocessing/find_empty_frames.ipynb "
                             "and preprocessing/build_flagged_labels.py). Requires "
                             "challenge_data/train/labels_flagged.csv to exist. Unset by default (vanilla CE).")
    parser.add_argument("--class-smooth-alpha", type=float, default=0.0,
                        help="Blend this much of each sample's target toward its class's CLIP "
                             "text-embedding similarity row (see preprocessing/build_class_similarity.py). "
                             "0.0 disables (default).")
    parser.add_argument("--class-smooth-temp", type=float, default=0.07,
                        help="Softmax temperature applied to the cosine-similarity row before blending.")
    parser.add_argument("--class-smooth-path", type=Path, default=None,
                        help="Path to class_similarity.npy (default: <data-root>/class_similarity.npy).")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
