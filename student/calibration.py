"""Post-hoc probability calibration helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def identity_beta_calibration(num_classes: int, eps: float = 1e-6) -> dict[str, Any]:
    return {
        "method": "beta",
        "a": [1.0] * int(num_classes),
        "b": [1.0] * int(num_classes),
        "c": [0.0] * int(num_classes),
        "eps": float(eps),
        "objective": "one_vs_rest_bce",
        "valid_classes": [],
    }


def fit_beta_calibration(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 200,
    l2: float = 1e-4,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Fit class-wise beta calibration on multiclass probabilities."""
    if probs.ndim != 2:
        raise ValueError(f"expected probs with shape (N, K), got {tuple(probs.shape)}")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("labels must have shape (N,) matching probs")

    probs = probs.detach().float().clamp(eps, 1.0 - eps)
    labels = labels.detach().long().to(probs.device)
    n, num_classes = probs.shape
    if n == 0:
        raise ValueError("cannot fit beta calibration on an empty validation set")

    targets = F.one_hot(labels, num_classes=num_classes).float()
    positives = targets.sum(dim=0)
    valid = (positives > 0) & (positives < n)
    if not bool(valid.any()):
        return identity_beta_calibration(num_classes, eps)

    log_p = probs.log()
    neg_log1m_p = -torch.log1p(-probs)
    init_ab = _inverse_softplus(torch.ones(num_classes, 2, device=probs.device))
    raw_ab = nn.Parameter(init_ab.clone())
    c = nn.Parameter(torch.zeros(num_classes, device=probs.device))
    optimizer = torch.optim.LBFGS([raw_ab, c], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        ab = F.softplus(raw_ab)
        beta_logits = ab[:, 0] * log_p + ab[:, 1] * neg_log1m_p + c
        loss = F.binary_cross_entropy_with_logits(beta_logits[:, valid], targets[:, valid])
        if l2 > 0:
            identity_penalty = (
                (ab[valid, 0] - 1.0).pow(2)
                + (ab[valid, 1] - 1.0).pow(2)
                + c[valid].pow(2)
            ).mean()
            loss = loss + float(l2) * identity_penalty
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return identity_beta_calibration(num_classes, eps)

    with torch.no_grad():
        ab = F.softplus(raw_ab)
        a = ab[:, 0]
        b = ab[:, 1]
        invalid = ~valid
        a[invalid] = 1.0
        b[invalid] = 1.0
        c[invalid] = 0.0
        if not (torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(c).all()):
            return identity_beta_calibration(num_classes, eps)

    return {
        "method": "beta",
        "a": a.detach().cpu().tolist(),
        "b": b.detach().cpu().tolist(),
        "c": c.detach().cpu().tolist(),
        "eps": float(eps),
        "objective": "one_vs_rest_bce",
        "l2": float(l2),
        "max_iter": int(max_iter),
        "valid_classes": torch.nonzero(valid, as_tuple=False).flatten().cpu().tolist(),
    }


def apply_beta_calibration(probs: torch.Tensor, beta: dict[str, Any]) -> torch.Tensor:
    eps = float(beta.get("eps", 1e-6))
    probs = probs.float().clamp(eps, 1.0 - eps)
    dtype = probs.dtype
    device = probs.device
    a = torch.as_tensor(beta["a"], device=device, dtype=dtype)
    b = torch.as_tensor(beta["b"], device=device, dtype=dtype)
    c = torch.as_tensor(beta["c"], device=device, dtype=dtype)
    if a.numel() != probs.shape[1] or b.numel() != probs.shape[1] or c.numel() != probs.shape[1]:
        raise ValueError("beta calibration parameter count does not match number of classes")
    beta_logits = a * probs.log() - b * torch.log1p(-probs) + c
    calibrated = torch.sigmoid(beta_logits).clamp_min(eps)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(eps)


def checkpoint_calibration(ckpt: dict[str, Any]) -> dict[str, Any]:
    calibration = ckpt.get("calibration")
    if calibration is not None:
        return calibration
    return {"method": "temperature", "temperature": float(ckpt.get("temperature", 1.0))}


def apply_calibration_to_logits(logits: torch.Tensor, calibration: dict[str, Any] | float | None) -> torch.Tensor:
    if calibration is None:
        return torch.softmax(logits.float(), dim=1)
    if isinstance(calibration, (float, int)):
        return torch.softmax(logits.float() / float(calibration), dim=1)

    method = calibration.get("method", "temperature")
    if method == "none":
        return torch.softmax(logits.float(), dim=1)
    if method == "temperature":
        return torch.softmax(logits.float() / float(calibration.get("temperature", 1.0)), dim=1)
    if method == "beta":
        return apply_beta_calibration(torch.softmax(logits.float(), dim=1), calibration)
    if method == "temperature_beta":
        temperature = float(calibration.get("temperature", 1.0))
        probs = torch.softmax(logits.float() / temperature, dim=1)
        return apply_beta_calibration(probs, calibration["beta"])
    raise ValueError(f"unknown calibration method {method!r}")


def describe_calibration(calibration: dict[str, Any] | float | None) -> str:
    if calibration is None:
        return "none"
    if isinstance(calibration, (float, int)):
        return f"temperature={float(calibration):.4f}"
    method = calibration.get("method", "temperature")
    if method == "temperature":
        return f"temperature={float(calibration.get('temperature', 1.0)):.4f}"
    if method == "temperature_beta":
        return f"temperature={float(calibration.get('temperature', 1.0)):.4f}+beta"
    return str(method)
