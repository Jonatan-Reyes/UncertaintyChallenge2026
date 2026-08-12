"""Dataset and dataloaders for the iWildCam summer-school challenge.

Reads a prepared ``challenge_data/`` directory:

    challenge_data/
        train/{images/<uid>.<ext>, labels.csv}
        val/{images/<uid>.<ext>, labels.csv}
        test_public/images/<uid>.<ext>       (no labels)
        class_mapping.json

Common things to tweak:
- ``IMG_SIZE`` — image side. Lower it to speed up training.
- ``default_train_transform`` — your augmentations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_SPLITS = {"train", "val", "test_public", "test_private"}


class IWildCamChallengeDataset(Dataset):
    """One row per image. Returns ``(image_tensor, label_int)`` for train/val,
    ``(image_tensor, uid_str)`` for test_public (which has no labels).

    Exposes:
        self.num_classes : int
        self.uids        : list[str]
        self.labels      : list[int] | None
        self.domains     : list[str] | None        (val only: 'id'/'ood')
    """

    def __init__(self, root, split: str, transform: Optional[Callable] = None):
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.images_dir = self.root / split / "images"

        with (self.root / "class_mapping.json").open() as f:
            self.num_classes = int(json.load(f)["num_classes"])

        labels_path = self.root / split / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            self.uids = df["uid"].astype(str).tolist()
            self.labels = df["y"].astype(int).tolist()
            self.domains = df["domain"].astype(str).tolist() if "domain" in df.columns else None
        else:
            # test_public: images only.
            self.uids = sorted(p.stem for p in self.images_dir.iterdir() if p.is_file())
            self.labels = None
            self.domains = None

    def __len__(self) -> int:
        return len(self.uids)

    def _find_image(self, uid: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.images_dir / f"{uid}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no image for uid {uid} in {self.images_dir}")

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        img = Image.open(self._find_image(uid)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img, uid
        return img, int(self.labels[idx])


def default_train_transform() -> Callable:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_eval_transform() -> Callable:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    root, batch_size: int = 32, num_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)`` with sensible defaults."""
    train_ds = IWildCamChallengeDataset(root, "train", default_train_transform())
    val_ds = IWildCamChallengeDataset(root, "val", default_eval_transform())
    test_ds = IWildCamChallengeDataset(root, "test_public", default_eval_transform())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
