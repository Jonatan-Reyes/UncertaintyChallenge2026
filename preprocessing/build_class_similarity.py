# %% [markdown]
# Semantic class-similarity matrix from CLIP text embeddings of `class_names.json`.
#
# Motivation: a softmax classifier spends its full probability mass over the known
# classes regardless of input, so confusions between visually/taxonomically close
# species (e.g. the three `equus` classes, or the two `mazama` brockets) shouldn't be
# penalized as harshly as confusions between unrelated classes. This builds a 57x57
# cosine-similarity matrix from CLIP text embeddings of each class's name, for use as
# a class-similarity soft-label target in `student/train.py` (`--class-smooth-alpha`).
#
# Bare names, not "a photo of a {name}" templates: CLIP's prompt-engineering advice
# (from the original paper) is about zero-shot *image*-text matching, where the text
# side needs to look like the caption-style alt-text the image encoder was aligned
# against. We only ever use the text tower here (text-to-text similarity, no image
# encoder involved), and empirically the shared template tokens inflate a constant
# ~0.55 mean off-diagonal similarity floor across every class pair versus ~0.47 bare
# -- the template dilutes the actual per-class signal rather than clarifying it.
# `empty`/`motorcycle` keep short disambiguating phrases since the bare words are
# genuinely ambiguous out of context (e.g. "empty" alone could mean anything).
#
# Output: `challenge_data/class_similarity.npy` (float32, C x C, row i = cosine sim of
# class i to every class j), `challenge_data/class_embeddings.npy` (float32, C x D, the
# L2-normalized embeddings the similarity matrix is the Gram matrix of -- kept around for
# `preprocessing/inspect_class_embeddings.ipynb`'s 3D projection, since a similarity
# matrix alone can't be projected to a point cloud) plus `class_similarity_meta.json`
# (prompts used, per class, for provenance/debugging).

# %%
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import open_clip
import torch

DATA_ROOT = REPO_ROOT / "challenge_data"
CLASS_NAMES_PATH = REPO_ROOT / "class_names.json"
MODEL_NAME = "ViT-L-14-quickgelu"  # matches OpenAI's actual QuickGELU checkpoint (avoids activation mismatch)
PRETRAINED = "openai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class_names = json.loads(CLASS_NAMES_PATH.read_text())
species = sorted(class_names["species"], key=lambda s: s["class_id"])
assert [s["class_id"] for s in species] == list(range(len(species))), \
    "species list must be dense 0..C-1, in order"


def prompts_for(entry: dict) -> list[str]:
    sci = entry["scientific_name"]
    if sci == "empty":
        return ["empty frame, no animal in the picture", "no animal present"]
    if sci == "motorcycle":
        return ["motorcycle", "dirt bike"]
    return [(entry.get("common_name") or sci).split(" (")[0]]


# %%
model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(DEVICE).eval()

# %%
embeddings = torch.empty(len(species), model.text_projection.shape[1], device=DEVICE)
meta = []
with torch.no_grad():
    for entry in species:
        prompts = prompts_for(entry)
        tokens = tokenizer(prompts).to(DEVICE)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[entry["class_id"]] = feats.mean(dim=0)
        meta.append({"class_id": entry["class_id"], "scientific_name": entry["scientific_name"], "prompts": prompts})
embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

# %%
sim = (embeddings @ embeddings.T).cpu().numpy().astype(np.float32)
np.save(DATA_ROOT / "class_similarity.npy", sim)
np.save(DATA_ROOT / "class_embeddings.npy", embeddings.cpu().numpy().astype(np.float32))
(DATA_ROOT / "class_similarity_meta.json").write_text(json.dumps(
    {"model": MODEL_NAME, "pretrained": PRETRAINED, "classes": meta}, indent=2,
))

# %%
print(f"saved {sim.shape} similarity matrix -> {DATA_ROOT / 'class_similarity.npy'}")
diag = np.diag(sim)
print(f"self-similarity: min={diag.min():.4f} max={diag.max():.4f} (should be ~1.0)")
off_diag_top = []
for i in range(len(species)):
    j = np.argsort(-sim[i])[1]  # best non-self match
    off_diag_top.append((species[i]["scientific_name"], species[j]["scientific_name"], float(sim[i, j])))
off_diag_top.sort(key=lambda t: -t[2])
print("top-5 closest cross-class pairs:")
for a, b, s in off_diag_top[:5]:
    print(f"  {a} <-> {b}: {s:.4f}")
