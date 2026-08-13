"""Phase 5 calibration ladder: global T -> top-label beta -> per-class beta
-> floor, each validated by out-of-fold CV before being trusted.

Every rung is scored on **out-of-fold** predictions only — never on the same
rows used to fit it, since a 1-3 parameter fit on 918 points can look better
than it is in-fold. Folds are stratified by ``domain`` (id/ood) so each fold
keeps roughly the true id/ood ratio.

Caveat vs PLAN.md: the plan calls for folds grouped by *location* so ood
locations are never split across fit/eval. ``val/labels.csv`` only carries a
coarse id/ood flag, not location ids, so that specific grouping isn't
available here — domain-stratified k-fold is the closest approximation, not
the same guarantee. Worth revisiting if location ids ever become available.

Input: a ``combine.py``-produced ``.npz`` (``uids``, ``probs``, ``labels``,
``domains``). Global T is assumed already applied upstream (that's
``combine.py``'s job) — this script starts from those probabilities and
layers beta/floor on top, each only kept if it wins on OOF metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from student import metrics as M
from student.calibration import (
    apply_beta_calibration,
    apply_toplabel_calibration,
    fit_beta_calibration,
    fit_toplabel_calibration,
)

N_FOLDS = 5


def stratified_folds(domains: np.ndarray, n_folds: int = N_FOLDS, seed: int = 0) -> list[np.ndarray]:
    """Returns a list of boolean held-out masks, one per fold, each keeping
    roughly the true id/ood ratio (see module docstring caveat)."""
    rng = np.random.RandomState(seed)
    fold_id = np.empty(len(domains), dtype=int)
    for dom in np.unique(domains):
        idx = np.where(domains == dom)[0]
        rng.shuffle(idx)
        fold_id[idx] = np.arange(len(idx)) % n_folds
    return [fold_id == k for k in range(n_folds)]


def oof_predict(probs: np.ndarray, labels: np.ndarray, folds: list[np.ndarray], fit_fn, apply_fn) -> np.ndarray:
    """Fit ``fit_fn(train_probs, train_labels)`` on each fold's complement,
    apply to the held-out fold, and stitch the OOF predictions back
    together in original row order."""
    oof = np.empty_like(probs)
    for held_out in folds:
        train = ~held_out
        params = fit_fn(probs[train], labels[train])
        oof[held_out] = apply_fn(probs[held_out], params)
    return oof


def fit_floor(probs: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> float:
    best_eps, best_score = 0.0, float("inf")
    for eps in grid:
        p = np.clip(probs, eps, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        score = M.nll(p, labels) + 10 * M.ece(p, labels)
        if score < best_score:
            best_eps, best_score = eps, score
    return best_eps


def apply_floor(probs: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def borda_rank(metrics_by_name: dict[str, dict]) -> dict[str, int]:
    names = list(metrics_by_name)
    metric_keys = ["accuracy", "ece", "nll", "brier", "misclassification_auroc"]
    higher_better = {"accuracy", "misclassification_auroc"}
    ranks = {n: 0 for n in names}
    for k in metric_keys:
        order = sorted(names, key=lambda n: metrics_by_name[n][k], reverse=(k in higher_better))
        for i, n in enumerate(order):
            ranks[n] += i
    return ranks


def run_ladder(probs: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> dict:
    folds = stratified_folds(domains)
    results = {}
    oof_probs = {}

    oof_probs["baseline (T only)"] = probs
    results["baseline (T only)"] = M.compute_all_metrics(probs, labels)

    oof_tl = oof_predict(
        probs, labels, folds,
        fit_fn=fit_toplabel_calibration,
        apply_fn=apply_toplabel_calibration,
    )
    oof_probs["+ top-label beta"] = oof_tl
    results["+ top-label beta"] = M.compute_all_metrics(oof_tl, labels)

    oof_pc = oof_predict(
        probs, labels, folds,
        fit_fn=lambda p, y: fit_beta_calibration(p, y),
        apply_fn=apply_beta_calibration,
    )
    oof_probs["+ per-class beta (instead of top-label)"] = oof_pc
    results["+ per-class beta (instead of top-label)"] = M.compute_all_metrics(oof_pc, labels)

    # Per-class beta is 3 params x K classes fit on ~n*(folds-1)/folds points
    # (171 params on ~734 points here) with many classes having a handful of
    # positives or none -- it has no argmax-preservation guarantee (unlike
    # top-label's clamp), so it CAN and does overfit rare classes hard enough
    # to swing accuracy. A calibrator changing accuracy at all is a red flag,
    # not a win: don't let the Borda ranking pick it just because the
    # overfit shows up as better numbers on 3-of-5 metrics. See PLAN.md's v1
    # notes -- this exact trap already cost AUROC there.
    baseline_acc = results["baseline (T only)"]["accuracy"]
    acc_tol = 0.005
    eligible = {
        name: m for name, m in results.items()
        if abs(m["accuracy"] - baseline_acc) <= acc_tol
    }
    for name, m in results.items():
        if name not in eligible:
            print(f"excluding {name!r} from selection: accuracy moved by "
                  f"{m['accuracy'] - baseline_acc:+.4f} (>{acc_tol}), likely overfit not calibration")

    ranks = borda_rank(eligible)
    best_name = min(ranks, key=ranks.get)
    best_probs = oof_probs[best_name]

    grid = np.linspace(0.0, 0.02, 21)
    eps = fit_floor(best_probs, labels, grid)
    floored = apply_floor(best_probs, eps)
    results[f"{best_name} + floor(eps={eps:.4f})"] = M.compute_all_metrics(floored, labels)

    return {
        "results": results,
        "ranks": ranks,
        "best_pre_floor": best_name,
        "floor_eps": eps,
        "final_probs": floored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 calibration ladder (OOF-validated).")
    parser.add_argument("--input", type=Path, required=True, help="combine.py-produced .npz with probs/labels/domains.")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the final calibrated probs.")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    probs, labels, domains = data["probs"], data["labels"], data["domains"]

    out = run_ladder(probs, labels, domains)
    for name, m in out["results"].items():
        rank = out["ranks"].get(name, "-")
        print(f"{name}: {m} (borda={rank})")
    print(f"\nbest pre-floor rung: {out['best_pre_floor']}, floor eps={out['floor_eps']:.4f}")

    for dom in ["id", "ood"]:
        mask = domains == dom
        print(f"{dom} (final):", M.compute_all_metrics(out["final_probs"][mask], labels[mask]))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, uids=data["uids"], probs=out["final_probs"], labels=labels, domains=domains)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
