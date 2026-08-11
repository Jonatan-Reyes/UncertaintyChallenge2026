"""Cache pooled backbone features for all four splits.

Frozen-backbone probes (linear head, kNN, multi-head ensembles, Mahalanobis
OOD scores, ...) all read from the same cached ``.npz`` files instead of
re-running the backbone forward pass every time. One run per backbone.

Loads each backbone with its default (pretrained-checkpoint-matching)
architecture — do NOT override ``global_pool`` at model-creation time, since
some pretrained checkpoints (e.g. DINOv2) only ship a ``norm`` layer and
break strict loading if timm's ``avg`` pooling head (which expects a
separate ``fc_norm``) is requested instead of the checkpoint's native
``token`` pooling. Instead we pull the full post-norm token sequence via
``forward_features`` and pool it ourselves, so a single forward pass yields
both the CLS token and the mean-patch embedding regardless of the
backbone's default pooling.

Usage:
    python -m student.extract_features --data-root challenge_data \
        --backbone vit_base_patch14_reg4_dinov2 --img-size 518 \
        --out-dir features/dinov2_vitb14_518 --gpu 0

Output: ``<out-dir>/<split>.npz`` with keys ``cls_embeddings`` (N, D),
``avg_embeddings`` (N, D), ``uids`` (N,), and ``labels`` / ``domains`` (N,)
when the split has a labels.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from student.data import (
    IMAGENET_MEAN, IMAGENET_STD, IWildCamChallengeDataset, aspect_preserving_eval_transform,
)

SPLITS = ("train", "val", "test_public", "test_private")


def make_transform(img_size: int, kind: str = "squash"):
    if kind == "squash":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if kind == "aspect":
        return aspect_preserving_eval_transform(img_size)
    raise ValueError(f"unknown kind {kind!r}")


@torch.no_grad()
def extract_split(model, data_root: Path, split: str, transform, device, batch_size, num_workers):
    ds = IWildCamChallengeDataset(data_root, split, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # CNNs (e.g. ConvNeXt) have no token sequence -- forward_features returns a
    # spatial (N, C, H, W) map, not (N, L, D). timm's num_classes=0 head already
    # does global pooling, so for those we just call the model directly and
    # cache a single 'embeddings' array (the legacy/CNN-agnostic key).
    is_vit = hasattr(model, "num_prefix_tokens")
    num_prefix = model.num_prefix_tokens if is_vit else None

    cls_feats, avg_feats, feats = [], [], []
    for imgs, _ in tqdm(loader, desc=split, leave=False):
        imgs = imgs.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if is_vit:
                tokens = model.forward_features(imgs)
            else:
                pooled = model(imgs)
        if is_vit:
            cls_feats.append(tokens[:, 0].float().cpu().numpy())
            avg_feats.append(tokens[:, num_prefix:].mean(dim=1).float().cpu().numpy())
        else:
            feats.append(pooled.float().cpu().numpy())

    out = {"uids": np.array(ds.uids, dtype=object)}
    if is_vit:
        out["cls_embeddings"] = np.concatenate(cls_feats).astype(np.float32)
        out["avg_embeddings"] = np.concatenate(avg_feats).astype(np.float32)
    else:
        out["embeddings"] = np.concatenate(feats).astype(np.float32)
    if ds.labels is not None:
        out["labels"] = np.array(ds.labels, dtype=np.int64)
    if ds.domains is not None:
        out["domains"] = np.array(ds.domains, dtype=object)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache pooled backbone features for all splits.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=str, required=True, help="timm model id")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--kind", type=str, default="squash", choices=["squash", "aspect"],
                        help="'squash' = plain square resize (default, matches the existing caches). "
                             "'aspect' = resize-short-side + center-crop, for training a probe that's "
                             "invariant to the crop TTA uses at inference.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    try:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=0, img_size=args.img_size)
    except TypeError:
        # fully-convolutional backbones (e.g. ConvNeXt) don't take img_size --
        # they're resolution-agnostic and infer it from the input tensor.
        model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    model.eval().to(device)

    transform = make_transform(args.img_size, args.kind)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        out = extract_split(model, args.data_root, split, transform, device, args.batch_size, args.num_workers)
        path = args.out_dir / f"{split}.npz"
        np.savez(path, **out)
        shapes = ", ".join(f"{k}={v.shape}" for k, v in out.items() if k.endswith("embeddings"))
        print(f"{split}: {shapes} -> {path}")


if __name__ == "__main__":
    main()
