"""Screen animal-labeled frames for hidden "actually empty" frames.

iWildCam labels come from video-level annotations, so a still frame
extracted from a clip can inherit the species label of the clip even when
the extracted frame itself shows no animal (the animal was there a second
earlier/later, just not in this frame). Class 0 ("empty") is the one label
guaranteed to have no animal in the frame, so those images are a trustworthy
set of *reference backgrounds*.

For every animal-labeled (y != 0) frame, this greyscales it and computes
SSIM against every class-0 frame that plausibly shares its camera: same
original image resolution (a proxy for camera model — this dataset only
ever emits one of three fixed resolutions) and same day/night lighting
(color daytime vs IR-flash night frames are never comparable, even from the
same physical camera). A high SSIM against *some* empty reference means the
frame's content is basically indistinguishable from a known-empty
background, i.e. a plausible mislabel.

Two of the three camera models burn a fixed black info bar (timestamp,
temperature, model name) into the top and bottom of every frame. That bar
is nearly identical in *structure* across any two images from the same
model regardless of scene content, and dominates SSIM on low-texture night
frames — an early run without cropping matched several genuinely
animal-containing night frames to an unrelated empty reference purely
because both had the same bar. Top/bottom ``--crop-frac`` of each frame is
stripped before SSIM to remove this confound.

Even after cropping the info bar, plain greyscale SSIM turned out to be a
weak signal on its own: two frames from *unrelated* cameras — a sunlit
road and a pitch-black forest interior — still scored ~0.72 SSIM near-native
resolution. Camera-trap night frames are mostly large, low-texture regions
(uniform dark background, sensor grain, smooth flash falloff), and SSIM's
structure term is stabilized by a constant added to the denominator
specifically so it doesn't blow up on flat patches — which means flat
patches trivially score ~1 regardless of *what* is flat, inflating the
whole-frame average for any two images that are each mostly flat. SSIM
alone is not trustworthy here.

So this also scores every candidate by cosine similarity to the nearest
empty reference in DINOv3 ConvNeXt-Base embedding space (reusing the cache
`embed_pca.py` already writes for the whole dataset — no recomputation).
Deep self-supervised features aren't fooled by shared flatness the way raw
pixel SSIM is, so they're treated as the primary ranking signal; SSIM is
kept as a secondary, independent corroborating score. Run `embed_pca.py`
first if `pca_train_val.npz` doesn't exist yet.

Like SSIM, the embedding match is restricted to same-resolution references
(the camera-model proxy): a different resolution can only mean a different
physical camera, so it can never be the same background, and letting those
matches through just made review harder (e.g. the grid's diff panel can't
even be drawn when the two frames aren't the same shape).

Both signals are calibrated against themselves: for every class-0 image,
its best leave-one-out score to *other* class-0 images in the same bucket
(SSIM) or overall (embedding) gives the score distribution genuinely empty
frames produce against each other. That distribution is bimodal (single-shot
locations score low against everything, repeat-shot locations score high
against each other), so a low percentile (e.g. p5) sits in the low mode and
is too permissive — the median is used as the default suggested cutoff;
inspect the histograms and override if needed.

This is a screening tool, not an auto-relabeler. A high score means "looks
like a plausible background", not "definitely animal-free" — e.g. two
different but visually similar empty locations can also score high. Inspect
`empty_frame_grid.png` (candidate | matched reference | abs greyscale diff,
ranked by embedding similarity) before trusting any threshold.

Usage:
    python student/code/OOD/embed_pca.py --data-root challenge_data   # once, if no cache yet
    python student/code/data-exploration/empty_frame_match.py --data-root challenge_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from student.data import IWildCamChallengeDataset  # noqa: E402

HERE = Path(__file__).resolve().parent

# Class-0 ("empty") frames that turned out to actually contain an animal, caught by
# eyeballing the review grids — a bad *reference* is worse than a bad candidate, since
# it can pull other candidates' scores up by matching against it. Found so far:
#   9052296b13e747d7d737b574  zebra crossing the frame
#   6f8856f5b76a5b783c71e8c1  animal (ears + eyeshine) facing the camera
#   3655de6270a12b5b86ccbf3b  motion-blurred animal (tail/hindquarters) close to the lens
#                             — found by audit_empty_refs.py, the least like any other
#                             empty frame (embed leave-one-out = 0.33)
# Add more here as they turn up; pass --exclude-ref-uids for one-off additions.
KNOWN_MISLABELED_EMPTY_REFS = {
    "9052296b13e747d7d737b574",
    "6f8856f5b76a5b783c71e8c1",
    "3655de6270a12b5b86ccbf3b",
}


def find_image(images_dir: Path, uid: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{uid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for uid {uid} in {images_dir}")


def is_night(rgb: np.ndarray) -> bool:
    """Near-zero cross-channel difference => IR-flash frame shot as greyscale-in-RGB."""
    diff = np.abs(rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int)).mean()
    diff += np.abs(rgb[:, :, 1].astype(int) - rgb[:, :, 2].astype(int)).mean()
    return bool(diff < 2.0)


def load_record(images_dir: Path, uid: str, split: str, y: int, max_side: int, crop_frac: float) -> dict:
    path = find_image(images_dir, uid)
    img = Image.open(path).convert("RGB")
    night = is_night(np.asarray(img))
    w, h = img.size
    top = int(h * crop_frac)
    grey = img.crop((0, top, w, h - top)).convert("L")
    grey.thumbnail((max_side, max_side), Image.BILINEAR)
    return {
        "uid": uid,
        "split": split,
        "y": y,
        "size": img.size,  # (w, h) original resolution, camera-model proxy
        "night": night,
        "grey": np.asarray(grey),
        "bucket": (img.size, night),
    }


def collect(data_root: Path, splits: list[str], want_empty: bool, max_side: int,
            crop_frac: float, limit: int | None = None) -> list[dict]:
    out = []
    for split in splits:
        df = pd.read_csv(data_root / split / "labels.csv")
        images_dir = data_root / split / "images"
        sub = df[df["y"] == 0] if want_empty else df[df["y"] != 0]
        pairs = list(zip(sub["uid"].astype(str), sub["y"].astype(int)))
        if limit is not None:
            pairs = pairs[: max(0, limit - len(out))]
        desc = f"{split} ({'empty refs' if want_empty else 'candidates'})"
        for uid, y in tqdm(pairs, desc=desc):
            out.append(load_record(images_dir, uid, split, y, max_side, crop_frac))
        if limit is not None and len(out) >= limit:
            break
    return out


def group_by_bucket(records: list[dict]) -> dict:
    buckets: dict = {}
    for r in records:
        buckets.setdefault(r["bucket"], []).append(r)
    return buckets


def best_match(grey: np.ndarray, own_uid: str, ref_list: list[dict], top_k: int):
    """Best (and top-k mean) SSIM of `grey` against `ref_list`, excluding `own_uid`."""
    scores = [(ssim(grey, ref["grey"], data_range=255), ref) for ref in ref_list if ref["uid"] != own_uid]
    if not scores:
        return None
    scores.sort(key=lambda t: -t[0])
    best_score, best_ref = scores[0]
    topk_mean = float(np.mean([s for s, _ in scores[:top_k]]))
    return best_score, best_ref, topk_mean


def load_embedding_index(data_root: Path, cache_path: Path, splits: list[str]) -> dict[str, np.ndarray]:
    """uid -> L2-normalized DINOv3 ConvNeXt embedding, for every uid in `splits`."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Run `python student/code/OOD/embed_pca.py "
            f"--data-root {data_root}` once to build it (or pass --no-embed to skip this signal)."
        )
    cache = np.load(cache_path)
    index: dict[str, np.ndarray] = {}
    for split, feats_key in (("train", "train_feats"), ("val", "val_feats")):
        if split not in splits or feats_key not in cache:
            continue
        uids = IWildCamChallengeDataset(data_root, split, transform=None).uids
        feats = cache[feats_key]
        assert len(uids) == len(feats), f"{split}: {len(uids)} uids vs {len(feats)} cached embeddings — stale cache?"
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-8, None)
        index.update(zip(uids, feats))
    return index


def embed_best_match(uid: str, feat: np.ndarray, ref_uids: np.ndarray, ref_feats: np.ndarray, top_k: int):
    """Best (and top-k mean) cosine similarity of `feat` against `ref_feats`, excluding `uid` itself."""
    if ref_uids.size == 0:
        return None
    sims = ref_feats @ feat
    mask = ref_uids != uid
    sims, ref_uids = sims[mask], ref_uids[mask]
    if len(sims) == 0:
        return None
    order = np.argsort(-sims)
    best_idx = order[0]
    topk_mean = float(sims[order[:top_k]].mean())
    return float(sims[best_idx]), str(ref_uids[best_idx]), topk_mean


def make_grid(data_root: Path, top: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(top)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, row in enumerate(top.itertuples()):
        cand_path = find_image(data_root / row.split / "images", row.uid)
        ref_path = find_image(data_root / row.embed_ref_split / "images", row.embed_ref_uid)
        cand_img = Image.open(cand_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")
        cand_grey = np.asarray(cand_img.convert("L"), dtype=np.int16)
        ref_grey = np.asarray(ref_img.convert("L"), dtype=np.int16)

        ssim_str = f"{row.ssim_best:.3f}" if pd.notna(row.ssim_best) else "n/a"
        axes[i, 0].imshow(cand_img)
        axes[i, 0].set_title(f"{row.uid}\ny={row.y}  embed={row.embed_best:.3f}  ssim={ssim_str}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(ref_img)
        axes[i, 1].set_title(f"nearest empty ref (embedding)\n{row.embed_ref_uid}", fontsize=8)
        axes[i, 1].axis("off")

        if cand_grey.shape == ref_grey.shape:
            axes[i, 2].imshow(np.abs(cand_grey - ref_grey), cmap="inferno", vmin=0, vmax=255)
        axes[i, 2].set_title("abs greyscale diff", fontsize=8)
        axes[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ref-splits", default="train,val", help="comma-separated splits to draw class-0 references from")
    parser.add_argument("--exclude-ref-uids", default="",
                         help="comma-separated uids to drop from the reference pool on top of "
                              "KNOWN_MISLABELED_EMPTY_REFS, e.g. after spotting a bad match in the review grid")
    parser.add_argument("--candidate-splits", default="train,val", help="comma-separated splits to screen y!=0 frames from")
    parser.add_argument("--max-side", type=int, default=128, help="greyscale thumbnail size used for SSIM")
    parser.add_argument("--crop-frac", type=float, default=0.10,
                         help="fraction of height stripped from top and bottom before SSIM, to drop burned-in "
                              "camera info bars (present on 2 of 3 camera models here) that would otherwise "
                              "dominate the score on low-texture night frames")
    parser.add_argument("--top-k", type=int, default=3, help="k for the top-k mean SSIM companion score")
    parser.add_argument("--threshold-percentile", type=float, default=50.0,
                         help="percentile of the true-empty leave-one-out SSIM distribution used as the "
                              "suggested cutoff. That distribution is bimodal (single-shot locations score low "
                              "against everything, repeat-shot locations score high against each other), so p5 "
                              "sits in the low mode and is too permissive — the median is a more defensible "
                              "default; inspect the histogram and override if needed")
    parser.add_argument("--limit", type=int, default=None, help="cap number of candidates loaded, for a quick test run")
    parser.add_argument("--embed-cache", type=Path,
                         default=HERE.parents[1] / "results" / "OOD" / "images" / "pca_train_val.npz",
                         help="cache written by embed_pca.py; run that script first if it doesn't exist")
    parser.add_argument("--no-embed", action="store_true", help="skip the embedding signal, SSIM only")
    parser.add_argument("--output-csv", type=Path,
                         default=HERE.parents[1] / "results" / "data-exploration" / "empty_frame_matches.csv")
    parser.add_argument("--output-hist", type=Path,
                         default=HERE.parents[1] / "results" / "data-exploration" / "images" / "empty_frame_ssim_hist.png")
    parser.add_argument("--output-grid", type=Path,
                         default=HERE.parents[1] / "results" / "data-exploration" / "images" / "empty_frame_grid.png")
    parser.add_argument("--grid-n", type=int, default=16, help="top-N candidates to render in the review grid")
    args = parser.parse_args()

    ref_splits = args.ref_splits.split(",")
    cand_splits = args.candidate_splits.split(",")

    print("loading empty (class-0) references...")
    refs = collect(args.data_root, ref_splits, want_empty=True, max_side=args.max_side, crop_frac=args.crop_frac)
    exclude = KNOWN_MISLABELED_EMPTY_REFS | {u for u in args.exclude_ref_uids.split(",") if u}
    removed = exclude & {r["uid"] for r in refs}
    if removed:
        refs = [r for r in refs if r["uid"] not in removed]
        print(f"dropped {len(removed)} known-mislabeled reference(s): {sorted(removed)}")
    print(f"{len(refs)} reference frames")

    print("loading candidate (animal-labeled) frames...")
    candidates = collect(args.data_root, cand_splits, want_empty=False, max_side=args.max_side,
                          crop_frac=args.crop_frac, limit=args.limit)
    print(f"{len(candidates)} candidate frames")

    ref_buckets = group_by_bucket(refs)
    print("reference buckets (resolution, is_night) -> count:")
    for k, v in sorted(ref_buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k}: {len(v)}")

    print("\ncalibrating: leave-one-out best SSIM among true-empty refs...")
    calib_scores = []
    for r in tqdm(refs, desc="calibration"):
        m = best_match(r["grey"], r["uid"], ref_buckets.get(r["bucket"], []), args.top_k)
        if m is not None:
            calib_scores.append(m[0])
    calib_scores = np.array(calib_scores)
    if len(calib_scores):
        percentiles = {p: float(np.percentile(calib_scores, p)) for p in (5, 25, 50, 75, 90)}
        threshold = float(np.percentile(calib_scores, args.threshold_percentile))
    else:
        percentiles, threshold = {}, float("nan")
    print(f"calibration: n={len(calib_scores)}  mean={calib_scores.mean():.3f}")
    print("  percentiles: " + "  ".join(f"p{p}={v:.3f}" for p, v in percentiles.items()))
    print(f"  using p{args.threshold_percentile:g}={threshold:.3f} as the suggested cutoff")

    print("\nmatching candidates against reference backgrounds (SSIM)...")
    rows = []
    for c in tqdm(candidates, desc="matching"):
        m = best_match(c["grey"], c["uid"], ref_buckets.get(c["bucket"], []), args.top_k)
        row = {
            "uid": c["uid"], "split": c["split"], "y": c["y"],
            "width": c["size"][0], "height": c["size"][1], "night": c["night"],
        }
        if m is None:
            row.update(ssim_ref_uid=None, ssim_ref_split=None, ssim_best=np.nan, ssim_topk_mean=np.nan)
        else:
            best_score, best_ref, topk_mean = m
            row.update(ssim_ref_uid=best_ref["uid"], ssim_ref_split=best_ref["split"],
                       ssim_best=best_score, ssim_topk_mean=topk_mean)
        rows.append(row)
    result = pd.DataFrame(rows)

    embed_calib = np.array([])
    embed_threshold = float("nan")
    if not args.no_embed:
        print("\nscoring against reference backgrounds (DINOv3 ConvNeXt embedding cosine similarity)...")
        splits_needed = sorted(set(ref_splits) | set(cand_splits))
        embed_index = load_embedding_index(args.data_root, args.embed_cache, splits_needed)

        # Bucket by resolution only (not day/night — embeddings are already robust to
        # lighting). Cross-resolution matches come from different camera models, which
        # can never be the same physical background, so they're not meaningful matches
        # and only make the reviewed pairs harder to sanity-check (e.g. the diff panel
        # in the grid can't even be drawn when the two frames aren't the same shape).
        embed_size_buckets: dict = {}
        for r in refs:
            if r["uid"] in embed_index:
                embed_size_buckets.setdefault(r["size"], []).append(r["uid"])
        embed_size_arrs = {
            size: (np.array(uids), np.stack([embed_index[u] for u in uids]))
            for size, uids in embed_size_buckets.items()
        }
        n_with_embed = sum(len(v) for v in embed_size_buckets.values())
        print(f"{n_with_embed}/{len(refs)} reference frames have cached embeddings")

        embed_calib_list = []
        for r in tqdm(refs, desc="embed calibration"):
            if r["uid"] not in embed_index:
                continue
            ref_uid_arr, ref_feat_arr = embed_size_arrs.get(r["size"], (np.array([]), np.empty((0, 0))))
            m = embed_best_match(r["uid"], embed_index[r["uid"]], ref_uid_arr, ref_feat_arr, args.top_k)
            if m is not None:
                embed_calib_list.append(m[0])
        embed_calib = np.array(embed_calib_list)
        embed_threshold = float(np.percentile(embed_calib, args.threshold_percentile)) if len(embed_calib) else float("nan")
        if len(embed_calib):
            embed_percentiles = {p: float(np.percentile(embed_calib, p)) for p in (5, 25, 50, 75, 90)}
            print(f"embed calibration: n={len(embed_calib)}  mean={embed_calib.mean():.3f}")
            print("  percentiles: " + "  ".join(f"p{p}={v:.3f}" for p, v in embed_percentiles.items()))
            print(f"  using p{args.threshold_percentile:g}={embed_threshold:.3f} as the suggested cutoff")

        ref_uid_to_split = {r["uid"]: r["split"] for r in refs}
        embed_cols = []
        for c in tqdm(candidates, desc="embed matching"):
            if c["uid"] not in embed_index:
                embed_cols.append((np.nan, np.nan, None, None))
                continue
            ref_uid_arr, ref_feat_arr = embed_size_arrs.get(c["size"], (np.array([]), np.empty((0, 0))))
            m = embed_best_match(c["uid"], embed_index[c["uid"]], ref_uid_arr, ref_feat_arr, args.top_k)
            if m is None:
                embed_cols.append((np.nan, np.nan, None, None))
                continue
            best_score, best_uid, topk_mean = m
            embed_cols.append((best_score, topk_mean, best_uid, ref_uid_to_split[best_uid]))
        result["embed_best"], result["embed_topk_mean"], result["embed_ref_uid"], result["embed_ref_split"] = zip(*embed_cols)

    sort_col = "embed_best" if "embed_best" in result.columns else "ssim_best"
    result = result.sort_values(sort_col, ascending=False, na_position="last")
    result.to_csv(args.output_csv, index=False)
    print(f"\nwrote {args.output_csv}")

    n_unmatched = int(result["ssim_best"].isna().sum())
    n_ssim_above = int((result["ssim_best"] >= threshold).sum())
    print(f"{n_unmatched}/{len(result)} candidates had no same-bucket SSIM reference (no score)")
    print(f"{n_ssim_above}/{len(result)} candidates score >= SSIM calibration p{args.threshold_percentile:g} ({threshold:.3f})")
    if "embed_best" in result.columns:
        n_embed_above = int((result["embed_best"] >= embed_threshold).sum())
        n_both_above = int(((result["embed_best"] >= embed_threshold) & (result["ssim_best"] >= threshold)).sum())
        print(f"{n_embed_above}/{len(result)} candidates score >= embedding calibration p{args.threshold_percentile:g} ({embed_threshold:.3f})")
        print(f"{n_both_above}/{len(result)} candidates score above threshold on BOTH signals (highest confidence)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have_embed = "embed_best" in result.columns
    fig, axes = plt.subplots(1, 2 if have_embed else 1, figsize=(14 if have_embed else 7, 5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.hist(calib_scores, bins=40, density=True, alpha=0.6, color="tab:green",
            label=f"true-empty leave-one-out (n={len(calib_scores)})")
    ax.hist(result["ssim_best"].dropna(), bins=40, density=True, alpha=0.6, color="tab:orange",
            label=f"animal-labeled candidates (n={int(result['ssim_best'].notna().sum())})")
    ax.axvline(threshold, color="k", linestyle="--",
               label=f"p{args.threshold_percentile:g} of true-empty ({threshold:.3f})")
    ax.set_xlabel("best SSIM vs a same-camera, same-lighting empty reference\n(unreliable alone — see docstring)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    if have_embed:
        ax = axes[1]
        ax.hist(embed_calib, bins=40, density=True, alpha=0.6, color="tab:green",
                label=f"true-empty leave-one-out (n={len(embed_calib)})")
        ax.hist(result["embed_best"].dropna(), bins=40, density=True, alpha=0.6, color="tab:orange",
                label=f"animal-labeled candidates (n={int(result['embed_best'].notna().sum())})")
        ax.axvline(embed_threshold, color="k", linestyle="--",
                   label=f"p{args.threshold_percentile:g} of true-empty ({embed_threshold:.3f})")
        ax.set_xlabel("best DINOv3 embedding cosine similarity vs an empty reference\n(primary signal)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output_hist, dpi=150)
    print(f"wrote {args.output_hist}")

    top = result.dropna(subset=[sort_col]).head(args.grid_n)
    if len(top) and have_embed:
        make_grid(args.data_root, top, args.output_grid)
        print(f"wrote {args.output_grid}")
    elif len(top):
        print("skipping review grid: needs the embedding signal (pass without --no-embed)")


if __name__ == "__main__":
    main()
