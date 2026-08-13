"""Combine TTA (§5) with the remaining §4.1 calibration checklist items and the
per-sample uncertainty injection (§4.4), then write the final submission.

Pipeline, in the order the plan's checklist requires (probability floor last):

1. Load per-view TTA probabilities (student.tta output) for val/test_public/
   test_private and average them -> ``avg_probs``. Treat ``log(avg_probs)`` as
   pseudo-logits so a scalar temperature can still be fit/applied on top of an
   averaged distribution (same trick as "temperature-scale the ensemble
   average" in plan §3).
2. Diagnostics only, not applied unless they change the decision:
   - k-fold (k=5) temperature fitting on val, compared against fit-on-all-val.
   - T_id vs T_ood fit separately, to check how much a pooled T compromises.
3. Per-sample injection T(u) = T0*exp(a*u) on top of the TTA pseudo-logits,
   reusing the Mahalanobis + kNN-disagreement signal from
   student.uncertainty_injection (built from cached single-view CLS features).
4. Probability floor epsilon, grid-searched on val NLL, applied last.
5. Same recipe applied to test_public/test_private -> submission.csv.

Usage:
    python -m student.calibrate_and_submit \
        --tta-dir runs/dinov2_vitb14_reg4_518_cls_linear_probe/tta \
        --features-dir features/dinov2_vitb14_reg4_518 \
        --checkpoint runs/dinov2_vitb14_reg4_518_cls_linear_probe/model.pt \
        --out-dir runs/dinov2_vitb14_reg4_518_cls_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from student import metrics as M
from student.calibration import (
    apply_beta_calibration, apply_toplabel_calibration,
    fit_beta_calibration, fit_toplabel_calibration,
)
from student.predict import write_submission
from student.train import fit_temperature
from student.uncertainty_injection import (
    auroc_of_score, build_head, fit_injection, knn_disagreement,
    mahalanobis_whitener, nearest_class_mahalanobis, probs_with_injection, zscore,
)
from student.train_probe import load_split

EPS = 1e-12
FLOOR_GRID = (0.0, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2)
N_FOLDS = 5


def load_tta(tta_dir: Path, split: str):
    d = np.load(tta_dir / f"{split}.npz", allow_pickle=True)
    view_probs = d["view_probs"]
    avg_probs = view_probs.mean(axis=0)
    labels = d["labels"] if "labels" in d else None
    domains = d["domains"] if "domains" in d else None
    uids = d["uids"]
    return avg_probs, labels, domains, uids


def pseudo_logits(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, EPS, 1.0))


def fit_T(logits: np.ndarray, labels: np.ndarray, device) -> float:
    return fit_temperature(torch.from_numpy(logits).float().to(device),
                            torch.from_numpy(labels).long().to(device))


def apply_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kfold_temperature(logits: np.ndarray, labels: np.ndarray, device, k: int = N_FOLDS) -> dict:
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(labels))
    folds = np.array_split(idx, k)
    fold_Ts, oof_nlls = [], []
    for i in range(k):
        held_out = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        T = fit_T(logits[train_idx], labels[train_idx], device)
        probs_held = apply_T(logits[held_out], T)
        fold_Ts.append(T)
        oof_nlls.append(M.nll(probs_held, labels[held_out]))
    T_all = fit_T(logits, labels, device)
    return {
        "fold_Ts": fold_Ts,
        "mean_fold_T": float(np.mean(fold_Ts)),
        "oof_nll_per_fold": oof_nlls,
        "mean_oof_nll": float(np.mean(oof_nlls)),
        "T_fit_on_all_val": T_all,
        "val_nll_at_T_fit_on_all": M.nll(apply_T(logits, T_all), labels),
    }


def class_log_prior(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    return np.log(counts / counts.sum())


def apply_logit_adjustment(logits: np.ndarray, log_prior: np.ndarray, tau: float) -> np.ndarray:
    """Menon et al. 2021 test-time logit adjustment: subtract tau*log(prior) so
    frequent classes stop dominating rare classes' logits (the long-tail analog
    of DINO's teacher-output centering, applied post-hoc instead of at train
    time -- no retraining needed since it's just an additive per-class bias on
    already-computed pseudo-logits). tau=0 is a no-op."""
    return logits - tau * log_prior[None, :]


def kfold_beta_calibration(probs: np.ndarray, labels: np.ndarray, k: int = N_FOLDS) -> dict:
    """Beta calibration fits 57*3=171 parameters -- unlike the 1-2 parameter
    temperature scaling above, fitting and evaluating that on the same 918
    val samples is a real leakage risk, not a formality. This gives an
    honest out-of-fold estimate to compare against the naive fit=eval number
    before trusting it for the final submission."""
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(labels))
    folds = np.array_split(idx, k)
    oof_probs = np.empty_like(probs)
    for i in range(k):
        held_out = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        calibrators = fit_beta_calibration(probs[train_idx], labels[train_idx])
        oof_probs[held_out] = apply_beta_calibration(probs[held_out], calibrators)
    return {"oof_metrics": M.compute_all_metrics(oof_probs, labels), "oof_probs": oof_probs}


def kfold_toplabel_calibration(probs: np.ndarray, labels: np.ndarray, k: int = N_FOLDS) -> dict:
    """Only 2 fitted parameters, so much less leakage-prone than beta's 171 --
    but reported out-of-fold anyway so the two candidates are compared on the
    same footing."""
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(labels))
    folds = np.array_split(idx, k)
    oof_probs = np.empty_like(probs)
    for i in range(k):
        held_out = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        params = fit_toplabel_calibration(probs[train_idx], labels[train_idx])
        oof_probs[held_out] = apply_toplabel_calibration(probs[held_out], params)
    return {"oof_metrics": M.compute_all_metrics(oof_probs, labels), "oof_probs": oof_probs}


def kfold_toplabel_on_temperature(logits: np.ndarray, labels: np.ndarray, device,
                                   k: int = N_FOLDS) -> dict:
    """OOF for the stacked T -> top-label recipe, refitting *both* stages inside
    each fold.

    Refitting T per fold matters and is not pedantry: reusing a T fit on all of
    val while cross-validating only the top-label stage gives 0.0385 ECE here
    against 0.0330 for the honest version -- the leaked stage is fit to NLL
    while the metric in question is ECE, so the hybrid is not merely optimistic,
    it estimates a procedure we never run."""
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(labels))
    folds = np.array_split(idx, k)
    oof_probs = np.empty((len(labels), logits.shape[1]))
    for i in range(k):
        held_out = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        T = fit_T(logits[train_idx], labels[train_idx], device)
        probs_train = apply_T(logits[train_idx], T)
        params = fit_toplabel_calibration(probs_train, labels[train_idx])
        oof_probs[held_out] = apply_toplabel_calibration(apply_T(logits[held_out], T), params)
    return {"oof_metrics": M.compute_all_metrics(oof_probs, labels), "oof_probs": oof_probs}


def id_vs_ood_temperature(logits: np.ndarray, labels: np.ndarray, domains: np.ndarray, device) -> dict:
    out = {}
    for domain in ("id", "ood"):
        mask = domains == domain
        if mask.any():
            out[f"T_{domain}"] = fit_T(logits[mask], labels[mask], device)
    return out


def compute_signal(features_dir: Path, split: str, emb: np.ndarray, pred: np.ndarray,
                    train_emb_np: np.ndarray, train_labels_np: np.ndarray,
                    train_maha: np.ndarray, train_knn: np.ndarray,
                    W: np.ndarray, class_means: np.ndarray,
                    maha_sign: float, knn_sign: float, knn_weight: float, device) -> np.ndarray:
    maha = nearest_class_mahalanobis(emb, W, class_means)
    knn = knn_disagreement(train_emb_np, train_labels_np, emb, pred, device)
    return knn_weight * knn_sign * zscore(knn, train_knn) + (1 - knn_weight) * maha_sign * zscore(maha, train_maha)


def run(tta_dir: Path, features_dir: Path, checkpoint: Path, out_dir: Path, knn_weight: float,
        final_method: str, tau: float = 0.0, skip_test: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_labels_for_prior = load_split(features_dir, "train", "cls")[1].numpy()

    val_avg_raw, val_labels, val_domains, val_uids = load_tta(tta_dir, "val")
    num_classes = int(val_avg_raw.shape[1])
    log_prior = class_log_prior(train_labels_for_prior, num_classes)
    val_logits_raw = pseudo_logits(val_avg_raw)
    val_pred_raw = val_avg_raw.argmax(axis=1)

    # Logit adjustment only touches the raw/global_T/beta candidates -- the
    # per-sample injection signal (Mahalanobis + kNN disagreement) stays on
    # the unadjusted logits/predictions below since it's an independent
    # post-hoc mechanism, not something the class-prior bias should feed into.
    if tau != 0.0:
        val_logits = apply_logit_adjustment(val_logits_raw, log_prior, tau)
        val_avg = apply_T(val_logits, 1.0)
    else:
        val_logits, val_avg = val_logits_raw, val_avg_raw
    val_pred = val_avg.argmax(axis=1)

    # --- diagnostics: k-fold T, id-vs-ood T ---
    kfold = kfold_temperature(val_logits, val_labels, device)
    id_ood_T = id_vs_ood_temperature(val_logits, val_labels, val_domains, device)
    print("k-fold T diagnostic:", json.dumps(kfold, indent=2))
    print("id-vs-ood T diagnostic:", json.dumps(id_ood_T, indent=2))

    T_pooled = kfold["T_fit_on_all_val"]
    rel_div = None
    base_T = T_pooled
    if "T_id" in id_ood_T and "T_ood" in id_ood_T:
        rel_div = abs(id_ood_T["T_id"] - id_ood_T["T_ood"]) / max(id_ood_T["T_id"], id_ood_T["T_ood"])
        # test is location-shifted like val/ood; lean toward T_ood if the two
        # diverge meaningfully (plan §4.1), otherwise the pooled fit is fine.
        if rel_div > 0.15:
            base_T = id_ood_T["T_ood"]
    print(f"pooled T={T_pooled:.4f}, id/ood relative divergence={rel_div}, base_T chosen={base_T:.4f}")

    # --- per-sample injection signal, built once from cached features.
    # Deliberately uses the *unadjusted* val_logits_raw/val_pred_raw below --
    # logit adjustment is a class-prior correction, orthogonal to this
    # feature-space misclassification signal, and "injected" was never the
    # final_method logit adjustment was evaluated against anyway (beta is).
    train_emb, train_labels, _, _ = load_split(features_dir, "train", "cls")
    train_emb_np, train_labels_np = train_emb.numpy(), train_labels.numpy()
    head = build_head(checkpoint, device)
    with torch.no_grad():
        train_cached_logits = head(train_emb.to(device)).cpu().numpy()
    train_cached_pred = train_cached_logits.argmax(axis=1)

    print("fitting Mahalanobis whitener + kNN disagreement on cached train features...")
    W, class_means = mahalanobis_whitener(train_emb_np, train_labels_np, num_classes)
    train_maha = nearest_class_mahalanobis(train_emb_np, W, class_means)
    train_knn = knn_disagreement(train_emb_np, train_labels_np, train_emb_np, train_cached_pred,
                                  device, exclude_self=True)

    val_emb, _, _, _ = load_split(features_dir, "val", "cls")
    val_emb_np = val_emb.numpy()
    val_maha = nearest_class_mahalanobis(val_emb_np, W, class_means)
    val_knn = knn_disagreement(train_emb_np, train_labels_np, val_emb_np, val_pred_raw, device)
    val_incorrect = (val_pred_raw != val_labels).astype(int)
    auroc_maha = auroc_of_score(zscore(val_maha, train_maha), val_incorrect)
    auroc_knn = auroc_of_score(zscore(val_knn, train_knn), val_incorrect)
    maha_sign = 1.0 if auroc_maha >= 0.5 else -1.0
    knn_sign = 1.0 if auroc_knn >= 0.5 else -1.0
    val_u = knn_weight * knn_sign * zscore(val_knn, train_knn) + (1 - knn_weight) * maha_sign * zscore(val_maha, train_maha)

    print("fitting T(u) = T0 * exp(a*u) on TTA pseudo-logits...")
    T0, alpha = fit_injection(val_logits_raw, val_labels, val_u, device)
    print(f"T0={T0:.4f} alpha={alpha:.4f}")

    probs_raw = val_avg
    probs_global_T = apply_T(val_logits, base_T)
    probs_injected = probs_with_injection(val_logits_raw, val_u, T0, alpha)

    # beta calibration (student/calibration.py) fit directly on val -- val is
    # already held out from training, so no internal CV is needed here (unlike
    # train_tree.py's BetaCalibratedClassifier, which has to manufacture
    # out-of-fold probs from train itself). One more post-hoc candidate to
    # compare against global-T and per-sample injection, not a replacement.
    beta_calibrators = fit_beta_calibration(probs_raw, val_labels)
    probs_beta = apply_beta_calibration(probs_raw, beta_calibrators)
    beta_kfold = kfold_beta_calibration(probs_raw, val_labels)
    print(f"beta calibration: naive fit=eval-on-val NLL={M.nll(probs_beta, val_labels):.4f} "
          f"vs. honest out-of-fold NLL={beta_kfold['oof_metrics']['nll']:.4f} "
          f"(gap = overfitting of the 171 fitted parameters on 918 val samples)")

    # Two variants: on the raw ensemble average, and stacked on top of the
    # global temperature. Not redundant -- T reshapes the whole simplex
    # (including how mass is split among the non-top classes, which is what
    # NLL sees on a misclassified sample), while top-label only moves mass
    # between the top class and the rest. Measured over 10 fold seeds:
    # stacking is worth -0.0024 aggregate ECE, 10/10 seeds, with AUROC/NLL/
    # Brier unchanged within noise.
    toplabel_params = fit_toplabel_calibration(probs_raw, val_labels)
    probs_toplabel = apply_toplabel_calibration(probs_raw, toplabel_params)
    toplabel_kfold = kfold_toplabel_calibration(probs_raw, val_labels)

    toplabel_T_params = fit_toplabel_calibration(probs_global_T, val_labels)
    probs_toplabel_T = apply_toplabel_calibration(probs_global_T, toplabel_T_params)
    toplabel_T_kfold = kfold_toplabel_on_temperature(val_logits, val_labels, device)
    print(f"top-label calibration: a={toplabel_params[0]:.4f} b={toplabel_params[1]:+.4f}, "
          f"argmax flips vs raw={int((probs_toplabel.argmax(axis=1) != val_pred).sum())} "
          f"(must be 0 -- accuracy is meant to be invariant)")

    results = {
        "tau": tau,
        "kfold_T_diagnostic": kfold,
        "id_vs_ood_T_diagnostic": id_ood_T,
        "base_T_chosen": base_T,
        "injection": {"T0": T0, "alpha": alpha, "auroc_maha": auroc_maha, "auroc_knn": auroc_knn,
                      "maha_sign": maha_sign, "knn_sign": knn_sign},
        "tta_raw": M.compute_all_metrics(probs_raw, val_labels),
        "tta_global_T": M.compute_all_metrics(probs_global_T, val_labels),
        "tta_injected": M.compute_all_metrics(probs_injected, val_labels),
        "tta_beta_naive_fit_eq_eval": M.compute_all_metrics(probs_beta, val_labels),
        "tta_beta_oof": beta_kfold["oof_metrics"],
        "toplabel_params": {"a": toplabel_params[0], "b": toplabel_params[1]},
        "tta_toplabel_naive_fit_eq_eval": M.compute_all_metrics(probs_toplabel, val_labels),
        "tta_toplabel_oof": toplabel_kfold["oof_metrics"],
        "toplabel_T_params": {"a": toplabel_T_params[0], "b": toplabel_T_params[1]},
        "tta_toplabel_T_naive_fit_eq_eval": M.compute_all_metrics(probs_toplabel_T, val_labels),
        "tta_toplabel_T_oof": toplabel_T_kfold["oof_metrics"],
    }
    for domain in ("id", "ood"):
        mask = val_domains == domain
        if mask.any():
            results[f"tta_raw/{domain}"] = M.compute_all_metrics(probs_raw[mask], val_labels[mask])
            results[f"tta_global_T/{domain}"] = M.compute_all_metrics(probs_global_T[mask], val_labels[mask])
            results[f"tta_injected/{domain}"] = M.compute_all_metrics(probs_injected[mask], val_labels[mask])
            results[f"tta_beta_naive_fit_eq_eval/{domain}"] = M.compute_all_metrics(probs_beta[mask], val_labels[mask])
            results[f"tta_beta_oof/{domain}"] = M.compute_all_metrics(
                beta_kfold["oof_probs"][mask], val_labels[mask])
            results[f"tta_toplabel_naive_fit_eq_eval/{domain}"] = M.compute_all_metrics(
                probs_toplabel[mask], val_labels[mask])
            results[f"tta_toplabel_oof/{domain}"] = M.compute_all_metrics(
                toplabel_kfold["oof_probs"][mask], val_labels[mask])
            results[f"tta_toplabel_T_naive_fit_eq_eval/{domain}"] = M.compute_all_metrics(
                probs_toplabel_T[mask], val_labels[mask])
            results[f"tta_toplabel_T_oof/{domain}"] = M.compute_all_metrics(
                toplabel_T_kfold["oof_probs"][mask], val_labels[mask])

    print(f"final_method={final_method!r} (choose from {{injected, beta, toplabel, global_T}} via "
          f"--final-method after comparing the tta_* candidates above)")
    final_probs = {"injected": probs_injected, "beta": probs_beta,
                   "toplabel": probs_toplabel, "toplabel_T": probs_toplabel_T,
                   "global_T": probs_global_T}[final_method]

    # --- probability floor, grid-searched on top of the chosen final_method's probs ---
    floor_table = []
    for eps in FLOOR_GRID:
        K = final_probs.shape[1]
        floored = (1 - eps) * final_probs + eps / K
        floor_table.append({"eps": eps, **M.compute_all_metrics(floored, val_labels)})
    best = min(floor_table, key=lambda r: r["nll"])
    best_eps = best["eps"]
    results["floor_grid"] = floor_table
    results["best_eps"] = best_eps
    print(f"probability floor grid (best eps={best_eps} by val NLL):")
    print(json.dumps(floor_table, indent=2))

    print(json.dumps(results, indent=2))
    (out_dir / "val_metrics.json").write_text(json.dumps(results, indent=2))

    if skip_test:
        return results

    # --- apply the full recipe to test_public/test_private ---
    print("scoring test_public / test_private...")
    all_uids: list[str] = []
    all_probs: list[np.ndarray] = []
    for split in ("test_public", "test_private"):
        avg_probs_raw, _, _, uids = load_tta(tta_dir, split)
        logits_raw = pseudo_logits(avg_probs_raw)
        pred_raw = avg_probs_raw.argmax(axis=1)
        if tau != 0.0:
            logits = apply_logit_adjustment(logits_raw, log_prior, tau)
            avg_probs = apply_T(logits, 1.0)
        else:
            logits, avg_probs = logits_raw, avg_probs_raw
        emb, _, _, feat_uids = load_split(features_dir, split, "cls")
        assert list(feat_uids) == list(uids), f"{split}: tta and cached-feature uid order mismatch"
        emb_np = emb.numpy()
        if final_method == "injected":
            u = compute_signal(features_dir, split, emb_np, pred_raw, train_emb_np, train_labels_np,
                               train_maha, train_knn, W, class_means, maha_sign, knn_sign, knn_weight, device)
            probs = probs_with_injection(logits_raw, u, T0, alpha)
        elif final_method == "beta":
            probs = apply_beta_calibration(avg_probs, beta_calibrators)
        elif final_method == "toplabel":
            probs = apply_toplabel_calibration(avg_probs, toplabel_params)
        elif final_method == "toplabel_T":
            probs = apply_toplabel_calibration(apply_T(logits, base_T), toplabel_T_params)
        else:
            probs = apply_T(logits, base_T)
        K = probs.shape[1]
        probs = (1 - best_eps) * probs + best_eps / K
        all_uids.extend(list(uids))
        all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    out_path = out_dir / "submission.csv"
    write_submission(all_uids, probs, out_path)
    print(f"wrote {out_path} ({len(all_uids)} rows)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TTA + calibration checklist + per-sample injection -> submission.")
    parser.add_argument("--tta-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--knn-weight", type=float, default=0.5)
    parser.add_argument("--final-method", type=str, default="injected",
                        choices=["injected", "beta", "toplabel", "toplabel_T", "global_T"],
                        help="Which calibration to apply for the final submission -- compare "
                             "the tta_* candidates in val_metrics.json first.")
    parser.add_argument("--tau", type=float, default=0.0,
                        help="Logit adjustment strength (Menon et al. 2021): subtract "
                             "tau*log(train class prior) from pseudo-logits before the "
                             "raw/global_T/beta candidates, to stop frequent classes from "
                             "dominating rare ones. 0.0 = off. Does not affect 'injected'.")
    args = parser.parse_args()
    run(args.tta_dir, args.features_dir, args.checkpoint, args.out_dir, args.knn_weight,
        args.final_method, args.tau)


if __name__ == "__main__":
    main()
