"""Local evaluator: load a checkpoint, run on val, report the four metrics + accuracy.

Output matches what the master evaluator computes on the server, so use this
to iterate locally before submitting.

Checkpoint format expected from ``student.train``::

    torch.save({
        "state_dict":   model.state_dict(),
        "num_classes":  K,
        "temperature":  T,            # 1.0 means no scaling
        "backbone":     backbone_name,  # timm model id, e.g. "resnet50"
    }, path)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - only needed for LoRA checkpoints
    PeftModel = None

from student.data import IWildCamChallengeDataset, default_eval_transform
from student.metrics import compute_all_metrics
from student.model import DEFAULT_BACKBONE, Classifier


def load_checkpoint(ckpt_path: Path, device) -> tuple[nn.Module, float]:
    if ckpt_path.is_dir():
        meta_path = ckpt_path / "checkpoint_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing LoRA metadata: {meta_path}")
        meta = json.loads(meta_path.read_text())
        backbone_name = meta.get("backbone", DEFAULT_BACKBONE)
        model = Classifier(int(meta["num_classes"]), backbone_name=backbone_name)
        if PeftModel is None:
            raise ImportError(
                "peft is required to load LoRA checkpoints. Install with: pip install peft"
            )
        model.backbone = PeftModel.from_pretrained(
            model.backbone,
            ckpt_path,
            is_trainable=False,
        )
        model.to(device).eval()
        return model, float(meta.get("temperature", 1.0))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", DEFAULT_BACKBONE)
    model = Classifier(int(ckpt["num_classes"]), backbone_name=backbone_name)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, float(ckpt.get("temperature", 1.0))


def collect_predictions(
    model: nn.Module, loader: DataLoader, device, temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and return ``(probs, labels)`` as np arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits / temperature, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(np.asarray(labels))
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def evaluate(
    model: nn.Module, loader: DataLoader, device, temperature: float = 1.0
) -> dict:
    probs, labels = collect_predictions(model, loader, device, temperature)
    return compute_all_metrics(probs, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the val split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, temperature = load_checkpoint(args.checkpoint, device)
    val_ds = IWildCamChallengeDataset(args.data_root, "val", default_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    metrics = evaluate(model, val_loader, device, temperature)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
