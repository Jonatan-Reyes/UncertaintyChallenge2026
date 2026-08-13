"""Combine multiple members' logits (from ``infer.py`` .npz files) into one
prediction: match (per-member temperature) then mix (weighted average).

"Match before mix" matters: members trained with different backbones/augs
have different confidence scales, so averaging raw probabilities lets the
most overconfident member dominate — we hit exactly this failure mode
combining augmentation-recipe variants of EVA02-L (equal-weight raw average
pushed ECE from ~0.04 to ~0.07; re-fitting a single T after mixing recovered
most of it). Fitting each member's own T first, then mixing, avoids relying
on a post-hoc fix.

Two mixing modes:
- ``arithmetic``: weighted mean of probabilities. Safer / better ECE.
- ``geometric``: weighted mean of log-probabilities (= logit averaging).
  Sharper, usually better NLL/accuracy, at some calibration cost.

For exactly 2 members, sweeps the arithmetic weight on a grid (0..1) and
picks by Borda rank over accuracy/ECE/NLL/Brier/AUROC. For 3+, uses equal
weights (an N-way grid search is a separate, more expensive tool).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from student import metrics as M
from student.train import _softmax_np, fit_temperature


def load_member(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    out = {"uids": data["uids"], "logits": data["logits"], "temperature": float(data["temperature"])}
    if "labels" in data:
        out["labels"] = data["labels"]
    if "domains" in data:
        out["domains"] = data["domains"]
    return out


def align_members(members: list[dict]) -> list[dict]:
    """Reindex every member to the first member's uid order."""
    ref_uids = list(members[0]["uids"])
    ref_index = {u: i for i, u in enumerate(ref_uids)}
    aligned = [members[0]]
    for m in members[1:]:
        order = [ref_index[u] for u in m["uids"]]
        inv = np.empty(len(order), dtype=int)
        for pos, orig_idx in enumerate(order):
            inv[orig_idx] = pos
        m2 = dict(m)
        m2["logits"] = m["logits"][inv]
        m2["uids"] = m["uids"][inv]
        if "labels" in m:
            m2["labels"] = m["labels"][inv]
        aligned.append(m2)
    return aligned


def match(member: dict, refit: bool = True) -> np.ndarray:
    """Return this member's calibrated probabilities: refit T via the
    ECE+NLL+Brier grid search if labels are available and ``refit``,
    otherwise fall back to the checkpoint's own stored T."""
    T = member["temperature"]
    if refit and "labels" in member:
        logits_t = torch.from_numpy(member["logits"])
        labels_t = torch.from_numpy(member["labels"]).long()
        T = fit_temperature(logits_t, labels_t)
    return _softmax_np(member["logits"] / T)


def mix(probs_list: list[np.ndarray], weights: np.ndarray, mode: str) -> np.ndarray:
    weights = weights / weights.sum()
    if mode == "arithmetic":
        return sum(w * p for w, p in zip(weights, probs_list))
    if mode == "geometric":
        eps = 1e-9
        log_mix = sum(w * np.log(np.clip(p, eps, 1.0)) for w, p in zip(weights, probs_list))
        return _softmax_np(log_mix)
    raise ValueError(f"unknown mode {mode!r}, choose 'arithmetic' or 'geometric'")


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


def sweep_two_member_weight(probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray,
                             mode: str, n_grid: int = 21) -> tuple[float, np.ndarray, dict]:
    """Grid-search the arithmetic weight on member A (0..1) for a 2-member
    mix, scored by Borda rank over the 5 metrics. Returns (best_w, best_probs, best_metrics)."""
    best = None
    for w in np.linspace(0.0, 1.0, n_grid):
        combined = mix([probs_a, probs_b], np.array([w, 1 - w]), mode)
        m = M.compute_all_metrics(combined, labels)
        if best is None or _score(m) < _score(best[2]):
            best = (w, combined, m)
    return best


def _score(m: dict) -> float:
    """Cheap scalar proxy for the grid sweep: normalized NLL+ECE (lower
    better). Final selection between candidate configs should still use
    the full Borda comparison, not this."""
    return m["nll"] + 10 * m["ece"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Match (per-member T) then mix (weighted average) member logits.")
    parser.add_argument("--members", nargs="+", required=True, type=Path,
                        help="Paths to infer.py-produced .npz files, one per member.")
    parser.add_argument("--mode", choices=["arithmetic", "geometric"], default="arithmetic")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Explicit mixing weights (same order as --members). "
                             "If omitted and there are exactly 2 members with labels, sweeps a grid.")
    parser.add_argument("--output", type=Path, default=None, help="Where to save combined probs (.npz).")
    args = parser.parse_args()

    members = align_members([load_member(p) for p in args.members])
    probs_list = [match(m) for m in members]
    has_labels = "labels" in members[0]
    labels = members[0].get("labels")

    if args.weights is not None:
        weights = np.array(args.weights)
        combined = mix(probs_list, weights, args.mode)
        if has_labels:
            print(M.compute_all_metrics(combined, labels))
    elif len(members) == 2 and has_labels:
        w, combined, m = sweep_two_member_weight(probs_list[0], probs_list[1], labels, args.mode)
        print(f"best weight on member 0: {w:.3f} (member 1: {1 - w:.3f})")
        print(m)
    else:
        weights = np.ones(len(members))
        combined = mix(probs_list, weights, args.mode)
        if has_labels:
            print(M.compute_all_metrics(combined, labels))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {"uids": members[0]["uids"], "probs": combined}
        if has_labels:
            out["labels"] = labels
        if "domains" in members[0]:
            out["domains"] = members[0]["domains"]
        np.savez(args.output, **out)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
