"""Dataset, per-backbone normalization, and named transform recipes.

Reads a prepared ``challenge_data/`` directory:

    challenge_data/
        train/{images/<uid>.<ext>, labels.csv}
        val/{images/<uid>.<ext>, labels.csv}
        test_public/images/<uid>.<ext>       (no labels)
        class_mapping.json

Normalization is resolved per backbone via ``timm.data.resolve_data_config`` —
never hardcode ImageNet stats. EVA02 (CLIP-pretrained) uses CLIP mean/std;
DINOv2/v3 use ImageNet mean/std. Getting this wrong doesn't crash, it just
silently costs accuracy and decalibrates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import timm
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

    def __init__(self, root, split: str, transform: Optional[Callable] = None,
                 labels_filename: str = "labels.csv",
                 soft_relabel_threshold: Optional[float] = None):
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.images_dir = self.root / split / "images"
        self.soft_relabel_threshold = soft_relabel_threshold

        with (self.root / "class_mapping.json").open() as f:
            self.num_classes = int(json.load(f)["num_classes"])

        labels_path = self.root / split / labels_filename
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            self.uids = df["uid"].astype(str).tolist()
            self.labels = df["y"].astype(int).tolist()
            self.domains = df["domain"].astype(str).tolist() if "domain" in df.columns else None
            if soft_relabel_threshold is not None:
                if "cos_to_nothing_knn" not in df.columns:
                    raise ValueError(
                        f"{labels_path} has no cos_to_nothing_knn column — run "
                        "preprocessing/build_flagged_labels.py first, or pass "
                        "labels_filename='labels_flagged.csv'"
                    )
                sim = df["cos_to_nothing_knn"].fillna(0.0)
                self.soft_weights = [float(s) if s > soft_relabel_threshold else 0.0 for s in sim]
            else:
                self.soft_weights = None
        else:
            # test_public / test_private: images only.
            self.uids = sorted(p.stem for p in self.images_dir.iterdir() if p.is_file())
            self.labels = None
            self.domains = None
            self.soft_weights = None

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
        if self.soft_weights is not None:
            return img, int(self.labels[idx]), self.soft_weights[idx]
        return img, int(self.labels[idx])


def resolve_norm(backbone_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(mean, std) timm ships for this backbone's pretrained checkpoint.

    Falls back to ImageNet stats if the backbone has no pretrained cfg
    (e.g. a from-scratch smoke-test model) rather than raising.
    """
    cfg = timm.get_pretrained_cfg(backbone_name)
    if cfg is None or cfg.mean is None:
        return IMAGENET_MEAN, IMAGENET_STD
    return tuple(cfg.mean), tuple(cfg.std)


# ---------------------------------------------------------------------------
# Named training-augmentation recipes (Phase 1 sweep). Each takes
# (img_size, mean, std) and returns a torchvision transform. Rationale is in
# PLAN.md — briefly: camera traps swap between colour daylight and IR night
# exposure, so brightness/contrast/grayscale are real domain variation, not
# synthetic nuisance; random erasing targets the uncertainty metrics
# (forces calls on partial evidence) rather than accuracy.
# ---------------------------------------------------------------------------

def _light(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _geom(img_size, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _photo(img_size, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _mask(img_size, mean, std):
    t = _photo(img_size, mean, std)
    t.transforms.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))
    return t


def _heavy(img_size, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.2)),
    ])


TRAIN_RECIPES: dict[str, Callable] = {
    "light": _light,
    "geom": _geom,
    "photo": _photo,
    "mask": _mask,
    "heavy": _heavy,
}


def train_transform(recipe: str = "light", img_size: int = IMG_SIZE,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Callable:
    if recipe not in TRAIN_RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}, choose from {sorted(TRAIN_RECIPES)}")
    return TRAIN_RECIPES[recipe](img_size, mean, std)


def default_eval_transform(img_size: int = IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Callable:
    """Squash-resize eval view — matches training's plain-resize baseline."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def aspect_preserving_eval_transform(img_size: int = IMG_SIZE, mean=IMAGENET_MEAN,
                                      std=IMAGENET_STD) -> Callable:
    """Resize the short side to ``img_size`` then center-crop, instead of
    squashing to a square. Source images are 1.25-1.8 aspect, so this is a
    genuinely different view — used as a TTA member."""
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def hflip_eval_transform(img_size: int = IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Callable:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# Named TTA views, shared between train.py's val-time checks and infer.py.
# Each is a zero-arg-except-(img_size, mean, std) callable, same signature as
# the train recipes, so infer.py can iterate a fixed list of view builders.
TTA_VIEWS: dict[str, Callable] = {
    "squash": default_eval_transform,
    "crop": aspect_preserving_eval_transform,
    "hflip": hflip_eval_transform,
}


def get_dataloaders(
    root, backbone_name: str, batch_size: int = 32, num_workers: int = 4,
    img_size: int = IMG_SIZE, recipe: str = "light",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)`` normalized for
    ``backbone_name`` via its timm pretrained cfg."""
    mean, std = resolve_norm(backbone_name)
    train_ds = IWildCamChallengeDataset(root, "train", train_transform(recipe, img_size, mean, std))
    val_ds = IWildCamChallengeDataset(root, "val", default_eval_transform(img_size, mean, std))
    test_ds = IWildCamChallengeDataset(root, "test_public", default_eval_transform(img_size, mean, std))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
