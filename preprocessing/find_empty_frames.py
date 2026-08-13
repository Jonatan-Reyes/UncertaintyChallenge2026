# %% [markdown]
# Flag animal-labeled train images whose DINOv2-giant features sit
# suspiciously close to the label-0 ("nothing") cluster — candidates for
# video frames where the animal isn't actually in this particular frame.
# Requires `extract_features.py` to have been run first. Run cell by cell.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

BACKBONE = "vit_giant_patch14_reg4_dinov2"
FEATURES_DIR = Path(__file__).resolve().parent / "features"
DATA_ROOT = REPO_ROOT / "challenge_data"
K_NEIGHBORS = 5

# %%
data = torch.load(FEATURES_DIR / f"train_{BACKBONE}.pt")
feats = F.normalize(data["features"], dim=1)
uids = data["uids"]
labels = torch.tensor(data["labels"])
domains = data["domains"]

nothing_mask = labels == 0
nothing_feats = feats[nothing_mask]
animal_feats = feats[~nothing_mask]

# %%
# Prototype similarity: cosine to the mean "nothing" embedding. Cheap, but
# only meaningful if the nothing class is roughly unimodal.
prototype = F.normalize(nothing_feats.mean(dim=0), dim=0)
cos_to_prototype = animal_feats @ prototype

# %%
# kNN similarity: mean cosine to the K closest "nothing" images. Robust to
# the nothing class actually being multi-modal (day vs. night empty frames,
# different camera locations, etc).
sim_matrix = animal_feats @ nothing_feats.T  # (num_animal, num_nothing)
knn_sim, knn_idx = sim_matrix.topk(K_NEIGHBORS, dim=1)
cos_to_nothing_knn = knn_sim.mean(dim=1)

# %%
animal_uids = [u for u, m in zip(uids, nothing_mask.tolist()) if not m]
animal_labels = labels[~nothing_mask]
animal_domains = [d for d, m in zip(domains, nothing_mask.tolist()) if not m] if domains else None

nothing_uids_arr = [u for u, m in zip(uids, nothing_mask.tolist()) if m]
nearest_nothing_uid = [nothing_uids_arr[i] for i in knn_idx[:, 0].tolist()]

df = pd.DataFrame({
    "uid": animal_uids,
    "label": animal_labels.tolist(),
    "domain": animal_domains,
    "cos_to_nothing_prototype": cos_to_prototype.tolist(),
    "cos_to_nothing_knn": cos_to_nothing_knn.tolist(),
    "nearest_nothing_uid": nearest_nothing_uid,
})
df = df.sort_values("cos_to_nothing_knn", ascending=False).reset_index(drop=True)
df.to_csv(FEATURES_DIR / "suspicious_empty_frames.csv", index=False)
df.head(30)

# %%
df["cos_to_nothing_knn"].describe()

# %%
# per-class share of suspicious images at a chosen threshold
THRESHOLD = 0.9
flagged = df[df["cos_to_nothing_knn"] > THRESHOLD]
print(f"{len(flagged)} / {len(df)} animal-labeled images above threshold {THRESHOLD}")
flagged["label"].value_counts()

# %%
import matplotlib.pyplot as plt

def _find_image(uid: str, split: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        p = DATA_ROOT / split / "images" / f"{uid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(uid)

top = df.head(16)
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for ax, (_, row) in zip(axes.flat, top.iterrows()):
    ax.imshow(Image.open(_find_image(row.uid, "train")).convert("RGB"))
    ax.set_title(f"y={row.label} sim={row.cos_to_nothing_knn:.3f}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(FEATURES_DIR / "top_suspicious_thumbnails.png", dpi=120)
plt.show()
