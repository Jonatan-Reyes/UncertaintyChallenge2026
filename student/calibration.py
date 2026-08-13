"""Beta calibration (Kull, Filho & Flach 2017) — not in sklearn's
`CalibratedClassifierCV`, which only offers `sigmoid` (Platt) and `isotonic`.

Per-class logistic regression on `(log p, -log(1-p))` instead of Platt's
single feature `p`. The extra term lets the calibration curve be asymmetric
(handles over- and under-confidence differently), which plain Platt scaling
can't express, while staying a 3-parameter-per-class parametric fit rather
than isotonic's per-bin step function — so it doesn't produce the hard
0-probability bins that wreck NLL on a small calibration set (`train_tree.py`
saw this: isotonic hit ECE 0.043 but NLL 2.41).

`p = 1` reduces to Platt scaling (only `log p` survives); `p = q = 1`
further reduces to no rescaling — Platt and identity are corner cases of
this family, not a different model class.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def _logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_beta_calibration(probs: np.ndarray, labels: np.ndarray,
                          eps: float = 1e-6) -> list[LogisticRegression | None]:
    """One binary beta-calibrator per class, fit directly on a probability
    matrix + labels. For use when a proper held-out set already exists (e.g.
    `val` in the TTA/injection pipeline) -- no internal CV needed, unlike
    `BetaCalibratedClassifier` below which has to manufacture out-of-fold
    probabilities from `train` itself.

    Classes with zero positive examples in `labels` (easy to hit with 918 val
    samples spread over 57 classes) get `None` -- logistic regression needs
    both classes present, and there's nothing to calibrate against anyway.
    `apply_beta_calibration` passes those columns through unscaled."""
    p = np.clip(probs, eps, 1 - eps)
    log_p, log_1mp = np.log(p), -np.log(1 - p)
    calibrators: list[LogisticRegression | None] = []
    for i in range(probs.shape[1]):
        y_bin = (labels == i).astype(int)
        if y_bin.sum() == 0:
            calibrators.append(None)
            continue
        feats = np.column_stack([log_p[:, i], log_1mp[:, i]])
        lr = LogisticRegression()
        lr.fit(feats, y_bin)
        calibrators.append(lr)
    return calibrators


def apply_beta_calibration(probs: np.ndarray, calibrators: list[LogisticRegression | None],
                            eps: float = 1e-6) -> np.ndarray:
    p = np.clip(probs, eps, 1 - eps)
    calibrated = p.copy()
    for i, lr in enumerate(calibrators):
        if lr is None:
            continue
        feats = np.column_stack([np.log(p[:, i]), -np.log(1 - p[:, i])])
        pos_idx = list(lr.classes_).index(1)
        calibrated[:, i] = lr.predict_proba(feats)[:, pos_idx]
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def fit_toplabel_calibration(probs: np.ndarray, labels: np.ndarray,
                              eps: float = 1e-7) -> tuple[float, float]:
    """Top-label calibration (Gupta & Ramdas 2021): a Platt fit on the *scalar*
    max-softmax confidence against correct/incorrect, `c' = sigmoid(a*logit(c) + b)`.

    Two parameters on a 1-D score, against beta calibration's 3*K. That matters
    here because val's reliability curve is S-shaped -- overconfident below
    c~0.5, underconfident from 0.5 to 0.93 -- which no single temperature can
    express (it needs opposite corrections in the two regions) but a monotone
    reshaping of confidence can.

    `a` is parameterised as `exp(.)` so the map stays monotone increasing; that
    is what keeps the confidence ranking, and hence misclassification AUROC,
    essentially untouched.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    z = _logit(conf)

    def objective(theta: np.ndarray) -> float:
        q = np.clip(_sigmoid(np.exp(theta[0]) * z + theta[1]), eps, 1 - eps)
        return float(-(correct * np.log(q) + (1 - correct) * np.log(1 - q)).mean())

    res = minimize(objective, np.zeros(2), method="Nelder-Mead")
    return float(np.exp(res.x[0])), float(res.x[1])


def apply_toplabel_calibration(probs: np.ndarray, params: tuple[float, float],
                                eps: float = 1e-6) -> np.ndarray:
    """Rescale the top class to its calibrated confidence and spread the
    remaining mass over the other classes in proportion to their current
    values, so only the confidence changes and the relative ordering of the
    non-top classes is preserved.

    The clamp is load-bearing: with no lower bound, pushing `q` below the
    renormalised runner-up silently flips the argmax, which changes accuracy
    and costs misclassification AUROC (measured: ECE 0.041->0.029 but AUROC
    0.929->0.920). Bounding `q` above `second/(second + 1 - c)` -- the value at
    which the runner-up would catch up after renormalisation -- keeps the
    prediction, and therefore accuracy, exactly invariant.
    """
    a, b = params
    conf = probs.max(axis=1)
    rows = np.arange(len(probs))
    top = probs.argmax(axis=1)

    without_top = probs.copy()
    without_top[rows, top] = -np.inf
    second = without_top.max(axis=1)

    q = _sigmoid(a * _logit(conf) + b)
    tie = second / np.clip(second + 1 - conf, eps, None)
    q = np.clip(q, np.minimum(tie + eps, 1 - eps), 1 - eps)

    scale = np.where(1 - conf > eps, (1 - q) / np.clip(1 - conf, eps, None), 0.0)
    out = probs * scale[:, None]
    out[rows, top] = q
    return out / out.sum(axis=1, keepdims=True)


class BetaCalibratedClassifier:
    """sklearn-style calibration wrapper: `cv`-fold out-of-fold predictions
    fit one binary beta-calibrator per class, then the base estimator is
    refit on all data for final raw probabilities at predict time."""

    def __init__(self, base_estimator, cv: int = 3, eps: float = 1e-6, seed: int = 0):
        self.base_estimator = base_estimator
        self.cv = cv
        self.eps = eps
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.classes_ = np.unique(y)
        oof_probs = np.zeros((len(y), len(self.classes_)))
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        for train_idx, calib_idx in skf.split(X, y):
            est = clone(self.base_estimator)
            sw = sample_weight[train_idx] if sample_weight is not None else None
            est.fit(X[train_idx], y[train_idx], sample_weight=sw)
            oof_probs[calib_idx] = est.predict_proba(X[calib_idx])

        self.calibrators_ = fit_beta_calibration(oof_probs, y, eps=self.eps)

        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_estimator_.predict_proba(X)
        return apply_beta_calibration(raw, self.calibrators_, eps=self.eps)
