"""Audit the class-0 ("empty") reference pool itself for mislabeled frames.

empty_frame_match.py found two references that actually contain an animal
(a zebra, and a frame with an animal's ears + eyeshine facing the camera)
just from eyeballing review grids. A bad *reference* is worse than a bad
candidate: it can pull other candidates' scores up by matching against it,
so it's worth actively hunting for more.

Applies the same leave-one-out idea to the reference pool itself: for every
class-0 image, find its own best match among the *other* class-0 images
(same-resolution bucket, embedding cosine similarity — the more reliable of
empty_frame_match.py's two signals). A very low score means "doesn't look
like any other empty frame we have," which is ambiguous on its own — could
be a mislabel, or could just be a one-off location with no repeat visits —
so this only ranks references for review, worst-first. Add confirmed
mislabels to KNOWN_MISLABELED_EMPTY_REFS in empty_frame_match.py.

Usage:
    python student/code/data-exploration/audit_empty_refs.py --data-root challenge_data
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent


def _load_matcher_module():
    """empty_frame_match.py can't be ``import``-ed normally: ``code/data-exploration``
    isn't a Python package. Load it by file path instead, mirroring ood_distance.py."""
    spec = importlib.util.spec_from_file_location("empty_frame_match", HERE / "empty_frame_match.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ref-splits", default="train,val")
    parser.add_argument("--exclude-ref-uids", default="",
                         help="comma-separated uids to also drop, on top of KNOWN_MISLABELED_EMPTY_REFS "
                              "(no point re-flagging refs already confirmed bad)")
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--crop-frac", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embed-cache", type=Path,
                         default=HERE.parents[1] / "results" / "OOD" / "images" / "pca_train_val.npz")
    parser.add_argument("--output-csv", type=Path,
                         default=HERE.parents[1] / "results" / "data-exploration" / "ref_audit.csv")
    parser.add_argument("--output-grid", type=Path,
                         default=HERE.parents[1] / "results" / "data-exploration" / "images" / "ref_audit_grid.png")
    parser.add_argument("--grid-n", type=int, default=40, help="worst-N references to render for review")
    args = parser.parse_args()

    mod = _load_matcher_module()
    ref_splits = args.ref_splits.split(",")

    print("loading empty (class-0) references...")
    refs = mod.collect(args.data_root, ref_splits, want_empty=True, max_side=args.max_side, crop_frac=args.crop_frac)
    exclude = mod.KNOWN_MISLABELED_EMPTY_REFS | {u for u in args.exclude_ref_uids.split(",") if u}
    removed = exclude & {r["uid"] for r in refs}
    if removed:
        refs = [r for r in refs if r["uid"] not in removed]
        print(f"dropped {len(removed)} already-known-mislabeled reference(s)")
    print(f"auditing {len(refs)} reference frames")

    ssim_buckets = mod.group_by_bucket(refs)

    print("\nscoring SSIM leave-one-out...")
    ssim_loo = {}
    for r in tqdm(refs, desc="ssim self-match"):
        m = mod.best_match(r["grey"], r["uid"], ssim_buckets.get(r["bucket"], []), args.top_k)
        ssim_loo[r["uid"]] = (m[0], m[1]["uid"]) if m is not None else (np.nan, None)

    print("\nscoring embedding leave-one-out...")
    embed_index = mod.load_embedding_index(args.data_root, args.embed_cache, ref_splits)
    size_buckets: dict = {}
    for r in refs:
        if r["uid"] in embed_index:
            size_buckets.setdefault(r["size"], []).append(r["uid"])
    size_arrs = {
        size: (np.array(uids), np.stack([embed_index[u] for u in uids]))
        for size, uids in size_buckets.items()
    }
    ref_uid_to_split = {r["uid"]: r["split"] for r in refs}
    embed_loo = {}
    for r in tqdm(refs, desc="embed self-match"):
        if r["uid"] not in embed_index:
            embed_loo[r["uid"]] = (np.nan, None)
            continue
        ref_uid_arr, ref_feat_arr = size_arrs.get(r["size"], (np.array([]), np.empty((0, 0))))
        m = mod.embed_best_match(r["uid"], embed_index[r["uid"]], ref_uid_arr, ref_feat_arr, args.top_k)
        embed_loo[r["uid"]] = (m[0], m[1]) if m is not None else (np.nan, None)

    rows = []
    for r in refs:
        ssim_best, ssim_ref = ssim_loo[r["uid"]]
        embed_best, embed_ref = embed_loo[r["uid"]]
        rows.append({
            "uid": r["uid"], "split": r["split"], "y": 0,
            "width": r["size"][0], "height": r["size"][1], "night": r["night"],
            "ssim_best": ssim_best, "ssim_ref_uid": ssim_ref,
            "embed_best": embed_best, "embed_ref_uid": embed_ref,
            "embed_ref_split": ref_uid_to_split.get(embed_ref) if embed_ref else None,
        })
    result = pd.DataFrame(rows).sort_values("embed_best", ascending=True, na_position="first")
    result.to_csv(args.output_csv, index=False)
    print(f"\nwrote {args.output_csv}")

    percentiles = {p: float(np.nanpercentile(result["embed_best"], p)) for p in (1, 5, 10, 25, 50)}
    print("embed leave-one-out percentiles (lowest = least like any other empty frame):")
    print("  " + "  ".join(f"p{p}={v:.3f}" for p, v in percentiles.items()))

    worst = result.dropna(subset=["embed_best", "embed_ref_uid"]).head(args.grid_n).reset_index(drop=True)
    if len(worst):
        mod.make_grid(args.data_root, worst, args.output_grid)
        print(f"wrote {args.output_grid} — review worst-{len(worst)} refs, add real mislabels to "
              f"KNOWN_MISLABELED_EMPTY_REFS in empty_frame_match.py")


if __name__ == "__main__":
    main()
