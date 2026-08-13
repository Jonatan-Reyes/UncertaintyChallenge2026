# %% [markdown]
# Turn `suspicious_empty_frames.csv` (from `find_empty_frames.py`) into
# label files that sit alongside the original `train/labels.csv` without
# touching it. Writes two derived files into `challenge_data/train/`:
#
# - `labels_flagged.csv`   — identical rows/order to `labels.csv`, plus
#   `cos_to_nothing_knn` and `suspicious_empty` columns. Nothing removed —
#   pass this to `IWildCamChallengeDataset(..., labels_filename=...)` and
#   filter/reweight in-place however you like.
# - `labels_excl_suspicious.csv` — same schema as the original
#   (uid, y, domain), with flagged rows dropped. A ready-to-use "remove it"
#   variant; the flagged rows are still recoverable from `labels_flagged.csv`.
#
# `labels.csv` itself is never modified.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

DATA_ROOT = REPO_ROOT / "challenge_data"
FEATURES_DIR = Path(__file__).resolve().parent / "features"
THRESHOLD = 0.9

# %%
labels = pd.read_csv(DATA_ROOT / "train" / "labels.csv")
suspicious = pd.read_csv(FEATURES_DIR / "suspicious_empty_frames.csv")[["uid", "cos_to_nothing_knn"]]

flagged = labels.merge(suspicious, on="uid", how="left")
flagged["suspicious_empty"] = flagged["cos_to_nothing_knn"] > THRESHOLD
flagged["suspicious_empty"] = flagged["suspicious_empty"].fillna(False)
assert len(flagged) == len(labels), "merge changed row count — labels.csv and suspicious csv uid sets diverged"

flagged.to_csv(DATA_ROOT / "train" / "labels_flagged.csv", index=False)

# %%
excluded = flagged.loc[~flagged["suspicious_empty"], ["uid", "y", "domain"]]
excluded.to_csv(DATA_ROOT / "train" / "labels_excl_suspicious.csv", index=False)

# %%
print(f"threshold: {THRESHOLD}")
print(f"total train rows: {len(labels)}")
print(f"flagged suspicious_empty: {int(flagged['suspicious_empty'].sum())}")
print(f"labels_excl_suspicious.csv rows: {len(excluded)}")
