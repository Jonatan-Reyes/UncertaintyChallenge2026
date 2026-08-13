"""Phase 6 submission builder: TTA-average -> cross-backbone combine ->
OOF-validated calibration ladder -> submission.csv.

Input: per-checkpoint, per-split ``infer.py`` output directories, each holding
one ``.npz`` per TTA view (``squash``/``crop``/``hflip``). Pipeline:

1. **TTA-average** each checkpoint's views (arithmetic mean of probabilities,
   each view scaled by the checkpoint's own already-fit ``T`` -- no per-view
   refit, since these are augmented copies of one model, not independent
   members; refitting per view on 918 val rows would add fit-variance with
   no real miscalibration behind it to correct).
2. **Combine** the two TTA-averaged backbones on val: grid-search mode
   (arithmetic/geometric) x weight, scored by full Borda rank over all five
   metrics (not the greedy NLL+ECE proxy ``combine.py``'s own CLI picks --
   see its docstring). Apply the winning (mode, weight) to val/test alike.
3. **Calibrate**: ``student.calibrate.run_ladder`` on the combined val probs
   reports OOF (5-fold, domain-stratified) metrics for baseline-T /
   top-label-beta / per-class-beta + floor, and picks a winner under an
   accuracy-preservation guard. That OOF winner is a *rung choice*, not a
   reusable transform (it fits and discards on each fold) -- so the winning
   rung's calibrator is refit once on the *full* val set and applied to
   test_public/test_private, which is what a submission actually needs.
4. **Write** submission.csv (test_public + test_private combined, matching
   ``predict.py``'s format), with row/uid/probability sanity checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from student import metrics as M
from student.calibrate import apply_floor, fit_floor, run_ladder
from student.calibration import (
    apply_beta_calibration,
    apply_toplabel_calibration,
    fit_beta_calibration,
    fit_toplabel_calibration,
)
from student.combine import borda_rank, mix
from student.predict import write_submission

TTA_VIEW_NAMES: tuple[str, ...] = ("squash", "crop", "hflip")

RUNG_FIT_APPLY = {
    "baseline (T only)": (None, None),
    "+ top-label beta": (fit_toplabel_calibration, apply_toplabel_calibration),
    "+ per-class beta (instead of top-label)": (fit_beta_calibration, apply_beta_calibration),
}


def _softmax(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_view(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    out = {"uids": data["uids"], "logits": data["logits"], "temperature": float(data["temperature"])}
    if "labels" in data:
        out["labels"] = data["labels"]
    if "domains" in data:
        out["domains"] = data["domains"]
    return out


def tta_average(view_dir: Path, views: tuple[str, ...] = TTA_VIEW_NAMES) -> dict:
    members = [_load_view(view_dir / f"{v}.npz") for v in views]
    ref_uids = list(members[0]["uids"])
    for v, m in zip(views, members):
        assert list(m["uids"]) == ref_uids, f"{view_dir / (v + '.npz')}: uid order diverged from {views[0]}"
    avg = sum(_softmax(m["logits"], m["temperature"]) for m in members) / len(members)
    out = {"uids": members[0]["uids"], "probs": avg}
    if "labels" in members[0]:
        out["labels"] = members[0]["labels"]
    if "domains" in members[0]:
        out["domains"] = members[0]["domains"]
    return out


def align_to(ref_uids: list, other: dict) -> dict:
    ref_index = {u: i for i, u in enumerate(ref_uids)}
    order = [ref_index[u] for u in other["uids"]]
    inv = np.empty(len(order), dtype=int)
    for pos, orig in enumerate(order):
        inv[orig] = pos
    out = dict(other)
    out["probs"] = other["probs"][inv]
    out["uids"] = other["uids"][inv]
    if "labels" in other:
        out["labels"] = other["labels"][inv]
    if "domains" in other:
        out["domains"] = other["domains"][inv]
    return out


def sweep_combine_config(probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray,
                          n_grid: int = 41) -> tuple[str, float, dict]:
    """Full-Borda grid search over (mode, weight-on-a), not the cheap
    NLL+ECE proxy ``combine.py``'s CLI uses for its own pick."""
    candidates: dict[str, np.ndarray] = {}
    meta: dict[str, tuple[str, float]] = {}
    for mode in ("arithmetic", "geometric"):
        for w in np.linspace(0.0, 1.0, n_grid):
            key = f"{mode}_{w:.3f}"
            candidates[key] = mix([probs_a, probs_b], np.array([w, 1 - w]), mode)
            meta[key] = (mode, float(w))
    metrics_by_name = {k: M.compute_all_metrics(v, labels) for k, v in candidates.items()}
    ranks = borda_rank(metrics_by_name)
    best_key = min(ranks, key=ranks.get)
    mode, w = meta[best_key]
    return mode, w, metrics_by_name[best_key]


def refit_and_apply(rung_name: str, val_probs: np.ndarray, val_labels: np.ndarray,
                     test_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refit the OOF-chosen rung's calibrator on the *full* val set (more
    data than any single fold) and apply to both val (for a final honest
    report) and test. Returns (val_calibrated, test_calibrated)."""
    fit_fn, apply_fn = RUNG_FIT_APPLY[rung_name]
    if fit_fn is None:
        return val_probs, test_probs
    params = fit_fn(val_probs, val_labels)
    return apply_fn(val_probs, params), apply_fn(test_probs, params)


def build(dino_dir: Path, eva_dir: Path, output: Path) -> None:
    splits = {}
    for split in ("val", "test_public", "test_private"):
        dino = tta_average(dino_dir / split)
        eva = tta_average(eva_dir / split)
        eva = align_to(list(dino["uids"]), eva)
        splits[split] = (dino, eva)

    val_dino, val_eva = splits["val"]
    labels, domains = val_dino["labels"], val_dino["domains"]
    mode, weight, val_combine_metrics = sweep_combine_config(val_dino["probs"], val_eva["probs"], labels)
    print(f"combine: mode={mode} weight_dino={weight:.3f} -> val metrics {val_combine_metrics}")

    combined = {}
    for split, (a, b) in splits.items():
        combined[split] = mix([a["probs"], b["probs"]], np.array([weight, 1 - weight]), mode)

    ladder = run_ladder(combined["val"], labels, domains)
    print("\n--- OOF calibration ladder (val, 5-fold domain-stratified) ---")
    for name, m in ladder["results"].items():
        rank = ladder["ranks"].get(name, "-")
        print(f"{name}: {m} (borda={rank})")
    rung = ladder["best_pre_floor"]
    print(f"\nwinning rung: {rung!r}")

    val_calibrated, test_public_calibrated = refit_and_apply(rung, combined["val"], labels, combined["test_public"])
    _, test_private_calibrated = refit_and_apply(rung, combined["val"], labels, combined["test_private"])

    eps = fit_floor(val_calibrated, labels, np.linspace(0.0, 0.02, 21))
    val_final = apply_floor(val_calibrated, eps)
    test_public_final = apply_floor(test_public_calibrated, eps)
    test_private_final = apply_floor(test_private_calibrated, eps)
    print(f"floor eps (refit on full val, post-rung): {eps:.4f}")

    print("\n--- final (full-val-refit rung + floor), honest check on val ---")
    for dom in ("id", "ood"):
        mask = domains == dom
        print(f"{dom}:", M.compute_all_metrics(val_final[mask], labels[mask]))
    print("pooled:", M.compute_all_metrics(val_final, labels))

    all_uids = list(splits["test_public"][0]["uids"]) + list(splits["test_private"][0]["uids"])
    all_probs = np.concatenate([test_public_final, test_private_final], axis=0)
    assert len(all_uids) == len(set(all_uids)), "duplicate uid across test_public/test_private"
    assert np.all(all_probs >= 0), "negative probability in final output"
    row_sums = all_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3), f"row sums drifted from 1.0: min={row_sums.min()} max={row_sums.max()}"

    write_submission(all_uids, all_probs, output)
    print(f"\nwrote {output} ({len(all_uids)} rows, {all_probs.shape[1]} classes)")

    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "combine_mode": mode, "combine_weight_dino": weight,
        "calibration_rung": rung, "floor_eps": float(eps),
        "val_combine_metrics": val_combine_metrics,
    }, indent=2))
    print(f"wrote {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TTA+combine+calibrate submission.csv.")
    parser.add_argument("--dino-infer-dir", type=Path, required=True,
                        help="infer.py output root with val/test_public/test_private subdirs.")
    parser.add_argument("--eva-infer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.dino_infer_dir, args.eva_infer_dir, args.output)


if __name__ == "__main__":
    main()
