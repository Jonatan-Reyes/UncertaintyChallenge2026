"""Run a checkpoint over one or more views of a split, saving raw logits.

Separates "produce predictions" from "combine/calibrate them" (``combine.py``,
``calibrate.py``): this script never applies temperature or averages
anything, so its output is reusable for every downstream combination without
re-running the (expensive) backbone forward pass.

One ``.npz`` per view, written to ``<output-dir>/<view>.npz``:

    uids       (N,)   str      — row order is stable across views/checkpoints
                                  for the same (data_root, split)
    logits     (N, K) float32  — raw, pre-temperature
    labels     (N,)   int64    — only for train/val
    domains    (N,)   str      — only for val ('id'/'ood')
    temperature        float   — the checkpoint's own fit T, for reference
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from student.data import IMG_SIZE, IWildCamChallengeDataset, TTA_VIEWS, default_eval_transform, resolve_norm
from student.eval import load_checkpoint
from student.model import DEFAULT_BACKBONE

VIEW_NAMES = ("eval",) + tuple(TTA_VIEWS)


def build_view_transform(view: str, backbone: str, img_size: int):
    mean, std = resolve_norm(backbone)
    if view == "eval":
        return default_eval_transform(img_size, mean, std)
    if view not in TTA_VIEWS:
        raise ValueError(f"unknown view {view!r}, choose from {VIEW_NAMES}")
    return TTA_VIEWS[view](img_size, mean, std)


def collect_logits(model: nn.Module, loader: DataLoader, device, has_labels: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns ``(logits, labels)``; ``labels`` is ``None`` for splits
    without ground truth (test_public/test_private). ``uids`` come from
    ``dataset.uids`` directly (row order matches, since ``shuffle=False``)."""
    model.eval()
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(imgs)
            logits_chunks.append(logits.float().cpu().numpy())
            if has_labels:
                labels_chunks.append(np.asarray(targets))
    logits = np.concatenate(logits_chunks, axis=0).astype(np.float32)
    labels = np.concatenate(labels_chunks) if has_labels else None
    return logits, labels


def run_infer(
    checkpoint: Path, data_root: Path, split: str, views: tuple[str, ...],
    output_dir: Path, batch_size: int = 64, num_workers: int = 8,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T = load_checkpoint(checkpoint, device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", DEFAULT_BACKBONE)
    img_size = ckpt.get("img_size") or IMG_SIZE
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for view in views:
        transform = build_view_transform(view, backbone, img_size)
        ds = IWildCamChallengeDataset(data_root, split, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logits, labels = collect_logits(model, loader, device, has_labels=ds.labels is not None)

        out = {"uids": np.array(ds.uids, dtype=object), "logits": logits, "temperature": np.float32(T)}
        if labels is not None:
            out["labels"] = labels
        if ds.domains is not None:
            out["domains"] = np.array(ds.domains, dtype=object)

        out_path = output_dir / f"{view}.npz"
        np.savez(out_path, **out)
        print(f"wrote {out_path} ({logits.shape[0]} rows, {logits.shape[1]} classes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a checkpoint over one or more views, save raw logits.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test_public", "test_private"])
    parser.add_argument("--views", nargs="+", default=["eval"], choices=list(VIEW_NAMES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    run_infer(
        checkpoint=args.checkpoint, data_root=args.data_root, split=args.split,
        views=tuple(args.views), output_dir=args.output_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
