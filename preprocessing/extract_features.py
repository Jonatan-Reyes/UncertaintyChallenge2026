# %% [markdown]
# Extract pooled DINOv2-giant features for every train/val image, so
# `find_empty_frames.py` can check whether animal-labeled images actually
# cluster away from the label-0 ("nothing") class. Run cell by cell.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from student.data import IWildCamChallengeDataset, default_eval_transform, resolve_norm

BACKBONE = "vit_giant_patch14_reg4_dinov2"
IMG_SIZE = 518  # native DINOv2 res; patch14 requires multiples of 14
BATCH_SIZE = 16
NUM_WORKERS = 16
DEVICE = "cuda"

DATA_ROOT = REPO_ROOT / "challenge_data"
OUT_DIR = Path(__file__).resolve().parent / "features"
OUT_DIR.mkdir(exist_ok=True)

# %%
model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=IMG_SIZE)
model.eval().to(DEVICE)
mean, std = resolve_norm(BACKBONE)
transform = default_eval_transform(img_size=IMG_SIZE, mean=mean, std=std)

# %%
def extract(split: str):
    ds = IWildCamChallengeDataset(DATA_ROOT, split, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    feats = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for imgs, _ in tqdm(loader, desc=split):
            imgs = imgs.to(DEVICE, non_blocking=True)
            feats.append(model(imgs).float().cpu())
    return torch.cat(feats), ds.uids, ds.labels, ds.domains

# %%
train_feats, train_uids, train_labels, train_domains = extract("train")
torch.save(
    {"features": train_feats, "uids": train_uids, "labels": train_labels, "domains": train_domains},
    OUT_DIR / f"train_{BACKBONE}.pt",
)

# %%
val_feats, val_uids, val_labels, val_domains = extract("val")
torch.save(
    {"features": val_feats, "uids": val_uids, "labels": val_labels, "domains": val_domains},
    OUT_DIR / f"val_{BACKBONE}.pt",
)

# %%
print(train_feats.shape, val_feats.shape)
