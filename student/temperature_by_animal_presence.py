"""Fit and apply group-wise temperature scaling from animal_presence.csv.

Groups are derived from animal_presence max_confidence and a chosen threshold:
- animal_present: max_confidence >= threshold
- no_animal:      max_confidence < threshold

Fit mode learns two temperatures on validation predictions (uid + p_* columns)
against <data-root>/val/labels.csv, then saves a JSON file.

Apply mode loads that JSON and rescales a submission CSV by uid, applying
the temperature corresponding to the uid's group from animal_presence.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from student.train import fit_temperature


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
        raise ValueError(f"{animal_presence_csv} must contain columns {required}")
    out = df[["uid", "max_confidence"]].copy()
    out["uid"] = out["uid"].astype(str)
    out["max_confidence"] = out["max_confidence"].astype(float)
    out["group"] = np.where(out["max_confidence"] >= float(threshold), "animal", "no_animal")
    if out["uid"].duplicated().any():
        raise ValueError("animal_presence CSV has duplicated uid values")
    return out[["uid", "group"]]


def _fit_group_temperature(probs: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    if probs.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected probs shape (N,K) and labels shape (N,)")
    logits = torch.log(torch.as_tensor(np.clip(probs, 1e-12, 1.0), dtype=torch.float32, device=device))
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    return float(fit_temperature(logits, labels_t))


def fit_group_temperatures(
    data_root: Path,
    val_submission_csv: Path,
    animal_presence_csv: Path,
    threshold: float,
    output_json: Path,
    fallback_temperature: float = 1.0,
) -> Path:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_probs = merged[p_cols].to_numpy(dtype=np.float64)
    all_labels = merged["y"].to_numpy(dtype=np.int64)
    t_global = _fit_group_temperature(all_probs, all_labels, device)

    out = {
        "threshold": float(threshold),
        "temperature_animal": float(fallback_temperature),
        "temperature_no_animal": float(fallback_temperature),
        "temperature_global": float(t_global),
        "fallback_temperature": float(fallback_temperature),
        "fit_rows": int(len(merged)),
        "fit_rows_animal": int((merged["group"] == "animal").sum()),
        "fit_rows_no_animal": int((merged["group"] == "no_animal").sum()),
    }

    for group_name, key in (("animal", "temperature_animal"), ("no_animal", "temperature_no_animal")):
        g = merged[merged["group"] == group_name]
        if len(g) < 2:
            # Too few samples for reliable fit; fallback to global T.
            out[key] = float(t_global)
            continue
        probs = g[p_cols].to_numpy(dtype=np.float64)
        labels = g["y"].to_numpy(dtype=np.int64)
        out[key] = _fit_group_temperature(probs, labels, device)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2))
    print(f"wrote {output_json}")
    print(json.dumps(out, indent=2))
    return output_json


def _apply_temperature_to_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    denom = np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)
    return exp_scaled / denom


def apply_group_temperatures(
    submission_csv: Path,
    animal_presence_csv: Path,
    calibration_json: Path,
    output_csv: Path,
) -> Path:
    if not submission_csv.exists():
        raise FileNotFoundError(f"Submission CSV not found: {submission_csv}")
    if not calibration_json.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {calibration_json}")

    calib = json.loads(calibration_json.read_text())
    threshold = float(calib["threshold"])
    t_animal = float(calib["temperature_animal"])
    t_no_animal = float(calib["temperature_no_animal"])
    t_fallback = float(calib.get("temperature_global", calib.get("fallback_temperature", 1.0)))

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

    mask_animal = group_values == "animal"
    mask_no_animal = group_values == "no_animal"
    mask_unknown = ~(mask_animal | mask_no_animal)

    out_probs = probs.copy()
    if mask_animal.any():
        out_probs[mask_animal] = _apply_temperature_to_probs(probs[mask_animal], t_animal)
    if mask_no_animal.any():
        out_probs[mask_no_animal] = _apply_temperature_to_probs(probs[mask_no_animal], t_no_animal)
    if mask_unknown.any():
        out_probs[mask_unknown] = _apply_temperature_to_probs(probs[mask_unknown], t_fallback)

    out_df = sub.copy()
    out_df[p_cols] = out_probs

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, float_format="%.10g")
    print(
        f"wrote {output_csv} (rows={len(out_df)}, animal={int(mask_animal.sum())}, "
        f"no_animal={int(mask_no_animal.sum())}, unknown={int(mask_unknown.sum())})"
    )
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and apply two temperatures based on animal presence groups from animal_presence.csv"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit two temperatures on validation predictions")
    fit_parser.add_argument("--data-root", type=Path, required=True, help="Path to challenge_data root")
    fit_parser.add_argument("--val-submission-csv", type=Path, required=True, help="Validation submission CSV (uid + p_*)")
    fit_parser.add_argument("--animal-presence-csv", type=Path, required=True, help="animal_presence CSV for val split")
    fit_parser.add_argument("--threshold", type=float, required=True, help="Threshold on max_confidence for animal-present group")
    fit_parser.add_argument("--output-json", type=Path, required=True, help="Where to save fitted temperatures JSON")
    fit_parser.add_argument("--fallback-temperature", type=float, default=1.0, help="Unused fallback kept for compatibility")

    apply_parser = subparsers.add_parser("apply", help="Apply fitted temperatures to a submission CSV")
    apply_parser.add_argument("--submission-csv", type=Path, required=True, help="Submission CSV to recalibrate")
    apply_parser.add_argument("--animal-presence-csv", type=Path, required=True, help="animal_presence CSV for target split")
    apply_parser.add_argument("--calibration-json", type=Path, required=True, help="JSON from fit subcommand")
    apply_parser.add_argument("--output-csv", type=Path, required=True, help="Where to write recalibrated submission CSV")

    args = parser.parse_args()

    if args.command == "fit":
        fit_group_temperatures(
            data_root=args.data_root,
            val_submission_csv=args.val_submission_csv,
            animal_presence_csv=args.animal_presence_csv,
            threshold=args.threshold,
            output_json=args.output_json,
            fallback_temperature=args.fallback_temperature,
        )
    elif args.command == "apply":
        apply_group_temperatures(
            submission_csv=args.submission_csv,
            animal_presence_csv=args.animal_presence_csv,
            calibration_json=args.calibration_json,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()
