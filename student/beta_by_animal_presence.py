"""Fit and apply animal-presence-conditional beta confidence calibration.

This is intentionally lower-capacity than class-wise beta calibration. With
only ~918 validation rows, splitting by Grounding-DINO animal presence leaves
too little data for per-class beta parameters. Instead, this script calibrates
the top-class confidence with a three-parameter beta map per group, preserves
the predicted class, and redistributes the remaining probability mass
proportionally across the non-top classes.

The command shape mirrors ``student.temperature_by_animal_presence``:

Fit:
    python -m student.beta_by_animal_presence fit \
        --data-root challenge_data \
        --val-submission-csv val_predictions.csv \
        --animal-presence-csv animal_detection/animal_presence.csv \
        --threshold 0.35 \
        --output-json animal_detection/beta_by_animal_presence.json

Apply:
    python -m student.beta_by_animal_presence apply \
        --submission-csv submission.csv \
        --animal-presence-csv animal_detection/animal_presence.csv \
        --calibration-json animal_detection/beta_by_animal_presence.json \
        --output-csv submission_beta_animal_presence.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from student.metrics import compute_all_metrics


def _probability_columns(df: pd.DataFrame) -> list[str]:
    p_cols = [c for c in df.columns if c.startswith("p_")]
    if not p_cols:
        raise ValueError("CSV must contain probability columns p_0, p_1, ...")
    p_cols = sorted(p_cols, key=lambda c: int(c.split("_")[1]))
    expected = [f"p_{i}" for i in range(len(p_cols))]
    if p_cols != expected:
        raise ValueError(f"Probability columns must be contiguous and 0-indexed: expected {expected}, got {p_cols}")
    return p_cols


def _load_presence_groups(animal_presence_csv: Path, threshold: float) -> pd.DataFrame:
    if not animal_presence_csv.exists():
        raise FileNotFoundError(f"animal_presence CSV not found: {animal_presence_csv}")
    df = pd.read_csv(animal_presence_csv)
    required = {"uid", "max_confidence"}
    if not required.issubset(df.columns):
        raise ValueError(f"{animal_presence_csv} must contain columns {sorted(required)}")
    out = df[["uid", "max_confidence"]].copy()
    out["uid"] = out["uid"].astype(str)
    out["max_confidence"] = out["max_confidence"].astype(float)
    out["group"] = np.where(out["max_confidence"] >= float(threshold), "animal", "no_animal")
    if out["uid"].duplicated().any():
        raise ValueError("animal_presence CSV has duplicated uid values")
    return out[["uid", "group"]]


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _identity_beta() -> dict[str, Any]:
    return {
        "a": 1.0,
        "b": 1.0,
        "c": 0.0,
        "status": "identity",
    }


def _fit_beta_confidence(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    device: torch.device,
    max_iter: int,
    l2: float,
    eps: float,
) -> dict[str, Any]:
    if confidences.ndim != 1 or correct.ndim != 1 or len(confidences) != len(correct):
        raise ValueError("confidences and correct must be matching 1D arrays")
    if len(confidences) == 0:
        raise ValueError("cannot fit beta calibration on an empty set")

    p = torch.as_tensor(confidences, dtype=torch.float32, device=device).clamp(eps, 1.0 - eps)
    y = torch.as_tensor(correct, dtype=torch.float32, device=device)

    raw_ab = nn.Parameter(_inverse_softplus(torch.ones(2, device=device)))
    c = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([raw_ab, c], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    log_p = p.log()
    neg_log1m_p = -torch.log1p(-p)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        ab = F.softplus(raw_ab)
        logits = ab[0] * log_p + ab[1] * neg_log1m_p + c
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if l2 > 0:
            loss = loss + float(l2) * ((ab[0] - 1.0).pow(2) + (ab[1] - 1.0).pow(2) + c.pow(2))
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        out = _identity_beta()
        out["status"] = "optimizer_failed"
        return out

    with torch.no_grad():
        ab = F.softplus(raw_ab)
        a = float(ab[0].detach().cpu())
        b = float(ab[1].detach().cpu())
        c_value = float(c.detach().cpu())
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c_value)):
        out = _identity_beta()
        out["status"] = "non_finite"
        return out

    return {
        "a": a,
        "b": b,
        "c": c_value,
        "status": "fit",
    }


def _beta_map(confidences: np.ndarray, beta: dict[str, Any], eps: float) -> np.ndarray:
    p = np.clip(confidences.astype(np.float64), eps, 1.0 - eps)
    logits = (
        float(beta["a"]) * np.log(p)
        - float(beta["b"]) * np.log1p(-p)
        + float(beta["c"])
    )
    return 1.0 / (1.0 + np.exp(-logits))


def _apply_beta_to_probs(
    probs: np.ndarray,
    beta: dict[str, Any],
    *,
    eps: float,
    preserve_argmax: bool,
) -> np.ndarray:
    if probs.ndim != 2:
        raise ValueError(f"expected probs with shape (N,K), got {probs.shape}")
    probs = np.clip(probs.astype(np.float64), eps, 1.0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)

    n, num_classes = probs.shape
    top_idx = probs.argmax(axis=1)
    row_idx = np.arange(n)
    top_p = probs[row_idx, top_idx]
    calibrated_top = np.clip(_beta_map(top_p, beta, eps), eps, 1.0 - eps)

    if preserve_argmax and num_classes > 1:
        tmp = probs.copy()
        tmp[row_idx, top_idx] = -np.inf
        max_other = tmp.max(axis=1)
        # If non-top probabilities are scaled to sum to 1-q, keeping the
        # original top class on top requires q >= r / (1 + r), where
        # r = max_other / sum_other.
        sum_other = np.clip(1.0 - top_p, eps, None)
        ratio = max_other / sum_other
        lower_bound = ratio / (1.0 + ratio)
        calibrated_top = np.maximum(calibrated_top, lower_bound + eps)
        calibrated_top = np.clip(calibrated_top, eps, 1.0 - eps)

    out = np.zeros_like(probs)
    non_top_scale = (1.0 - calibrated_top) / np.clip(1.0 - top_p, eps, None)
    out[:] = probs * non_top_scale[:, None]
    out[row_idx, top_idx] = calibrated_top
    out = np.clip(out, eps, 1.0)
    return out / np.clip(out.sum(axis=1, keepdims=True), eps, None)


def _metrics_for(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    metrics = compute_all_metrics(probs, labels)
    return {k: (None if v is None else float(v)) for k, v in metrics.items()}


def _load_fit_frame(
    data_root: Path,
    val_submission_csv: Path,
    animal_presence_csv: Path,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    labels_path = data_root / "val" / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Validation labels not found: {labels_path}")
    if not val_submission_csv.exists():
        raise FileNotFoundError(f"Validation submission CSV not found: {val_submission_csv}")

    labels_df = pd.read_csv(labels_path)
    if not {"uid", "y"}.issubset(labels_df.columns):
        raise ValueError("labels.csv must contain uid and y columns")
    labels_df = labels_df[["uid", "y"]].copy()
    labels_df["uid"] = labels_df["uid"].astype(str)
    labels_df["y"] = labels_df["y"].astype(int)

    pred_df = pd.read_csv(val_submission_csv)
    if "uid" not in pred_df.columns:
        raise ValueError("Validation submission CSV must contain uid column")
    p_cols = _probability_columns(pred_df)
    pred_df = pred_df[["uid"] + p_cols].copy()
    pred_df["uid"] = pred_df["uid"].astype(str)

    groups_df = _load_presence_groups(animal_presence_csv, threshold)
    merged = labels_df.merge(pred_df, on="uid", how="inner", sort=False)
    merged = merged.merge(groups_df, on="uid", how="inner", sort=False)
    if merged.empty:
        raise ValueError("No overlapping uid rows among labels, val submission, and animal_presence CSV")
    return merged, p_cols


def _group_can_fit(correct: np.ndarray, min_group_size: int, min_correct: int, min_incorrect: int) -> tuple[bool, str]:
    n = int(len(correct))
    n_correct = int(correct.sum())
    n_incorrect = n - n_correct
    if n < min_group_size:
        return False, f"too_few_rows:{n}<{min_group_size}"
    if n_correct < min_correct:
        return False, f"too_few_correct:{n_correct}<{min_correct}"
    if n_incorrect < min_incorrect:
        return False, f"too_few_incorrect:{n_incorrect}<{min_incorrect}"
    return True, "ok"


def fit_group_betas(
    data_root: Path,
    val_submission_csv: Path,
    animal_presence_csv: Path,
    threshold: float,
    output_json: Path,
    *,
    min_group_size: int = 150,
    min_correct: int = 25,
    min_incorrect: int = 25,
    l2: float = 0.05,
    max_iter: int = 100,
    eps: float = 1e-6,
    preserve_argmax: bool = True,
) -> Path:
    merged, p_cols = _load_fit_frame(data_root, val_submission_csv, animal_presence_csv, threshold)
    probs = merged[p_cols].to_numpy(dtype=np.float64)
    labels = merged["y"].to_numpy(dtype=np.int64)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_beta = _fit_beta_confidence(
        confidences,
        correct,
        device=device,
        max_iter=max_iter,
        l2=l2,
        eps=eps,
    )

    groups: dict[str, dict[str, Any]] = {}
    calibrated = np.empty_like(probs)
    for group_name in ("animal", "no_animal"):
        mask = (merged["group"].to_numpy() == group_name)
        group_correct = correct[mask]
        can_fit, reason = _group_can_fit(group_correct, min_group_size, min_correct, min_incorrect)
        if can_fit:
            beta = _fit_beta_confidence(
                confidences[mask],
                group_correct,
                device=device,
                max_iter=max_iter,
                l2=l2,
                eps=eps,
            )
            fallback_used = False
        else:
            beta = dict(global_beta)
            beta["status"] = f"fallback_global:{reason}"
            fallback_used = True

        groups[group_name] = {
            "beta": beta,
            "rows": int(mask.sum()),
            "correct": int(group_correct.sum()),
            "incorrect": int(len(group_correct) - group_correct.sum()),
            "fallback_used": bool(fallback_used),
        }
        if mask.any():
            calibrated[mask] = _apply_beta_to_probs(
                probs[mask],
                beta,
                eps=eps,
                preserve_argmax=preserve_argmax,
            )

    before_metrics = _metrics_for(probs, labels)
    after_metrics = _metrics_for(calibrated, labels)
    calibration = {
        "method": "animal_presence_beta_confidence",
        "threshold": float(threshold),
        "eps": float(eps),
        "l2": float(l2),
        "max_iter": int(max_iter),
        "min_group_size": int(min_group_size),
        "min_correct": int(min_correct),
        "min_incorrect": int(min_incorrect),
        "preserve_argmax": bool(preserve_argmax),
        "fit_rows": int(len(merged)),
        "num_classes": int(len(p_cols)),
        "global_beta": global_beta,
        "groups": groups,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(calibration, indent=2))
    print(f"wrote {output_json}")
    print(json.dumps({
        "fit_rows": calibration["fit_rows"],
        "groups": groups,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
    }, indent=2))
    return output_json


def apply_group_betas(
    submission_csv: Path,
    animal_presence_csv: Path,
    calibration_json: Path,
    output_csv: Path,
) -> Path:
    if not submission_csv.exists():
        raise FileNotFoundError(f"Submission CSV not found: {submission_csv}")
    if not calibration_json.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {calibration_json}")

    calibration = json.loads(calibration_json.read_text())
    threshold = float(calibration["threshold"])
    eps = float(calibration.get("eps", 1e-6))
    preserve_argmax = bool(calibration.get("preserve_argmax", True))
    global_beta = calibration["global_beta"]
    group_betas = {
        "animal": calibration["groups"]["animal"]["beta"],
        "no_animal": calibration["groups"]["no_animal"]["beta"],
    }

    sub = pd.read_csv(submission_csv)
    if "uid" not in sub.columns:
        raise ValueError("Submission CSV must contain uid column")
    p_cols = _probability_columns(sub)
    sub = sub.copy()
    sub["uid"] = sub["uid"].astype(str)

    groups_df = _load_presence_groups(animal_presence_csv, threshold)
    merged = sub.merge(groups_df, on="uid", how="left", sort=False)
    merged["group"] = merged["group"].fillna("unknown")

    probs = merged[p_cols].to_numpy(dtype=np.float64)
    group_values = merged["group"].to_numpy()
    out_probs = probs.copy()
    counts: dict[str, int] = {}
    for group_name in ("animal", "no_animal", "unknown"):
        mask = group_values == group_name
        counts[group_name] = int(mask.sum())
        if not mask.any():
            continue
        beta = group_betas.get(group_name, global_beta)
        out_probs[mask] = _apply_beta_to_probs(
            probs[mask],
            beta,
            eps=eps,
            preserve_argmax=preserve_argmax,
        )

    out_df = sub.copy()
    out_df[p_cols] = out_probs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, float_format="%.10g")
    print(f"wrote {output_csv} (rows={len(out_df)}, counts={counts})")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit/apply animal-presence-conditional beta confidence calibration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit beta calibrators on validation predictions")
    fit_parser.add_argument("--data-root", type=Path, required=True, help="Path to challenge_data root")
    fit_parser.add_argument("--val-submission-csv", type=Path, required=True, help="Validation CSV with uid + p_*")
    fit_parser.add_argument("--animal-presence-csv", type=Path, required=True, help="animal_presence CSV for val")
    fit_parser.add_argument("--threshold", type=float, required=True, help="Threshold on max_confidence")
    fit_parser.add_argument("--output-json", type=Path, required=True, help="Where to save fitted beta JSON")
    fit_parser.add_argument("--min-group-size", type=int, default=150)
    fit_parser.add_argument("--min-correct", type=int, default=25)
    fit_parser.add_argument("--min-incorrect", type=int, default=25)
    fit_parser.add_argument("--l2", type=float, default=0.05, help="Regularize beta parameters toward identity")
    fit_parser.add_argument("--max-iter", type=int, default=100)
    fit_parser.add_argument("--eps", type=float, default=1e-6)
    fit_parser.add_argument("--allow-argmax-change", action="store_true",
                            help="Allow calibration to change the predicted class")

    apply_parser = subparsers.add_parser("apply", help="Apply fitted beta calibration to a submission CSV")
    apply_parser.add_argument("--submission-csv", type=Path, required=True)
    apply_parser.add_argument("--animal-presence-csv", type=Path, required=True)
    apply_parser.add_argument("--calibration-json", type=Path, required=True)
    apply_parser.add_argument("--output-csv", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "fit":
        fit_group_betas(
            data_root=args.data_root,
            val_submission_csv=args.val_submission_csv,
            animal_presence_csv=args.animal_presence_csv,
            threshold=args.threshold,
            output_json=args.output_json,
            min_group_size=args.min_group_size,
            min_correct=args.min_correct,
            min_incorrect=args.min_incorrect,
            l2=args.l2,
            max_iter=args.max_iter,
            eps=args.eps,
            preserve_argmax=not args.allow_argmax_change,
        )
    elif args.command == "apply":
        apply_group_betas(
            submission_csv=args.submission_csv,
            animal_presence_csv=args.animal_presence_csv,
            calibration_json=args.calibration_json,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()
