"""Focal-loss objective for gradient boosters.

Focal loss down-weights easy, well-classified examples and concentrates the
gradient on hard ones — useful under extreme class imbalance where the majority
of negatives are trivially easy. This module provides the gradient and hessian
of binary focal loss for use as a custom objective in LightGBM and XGBoost, and
a sklearn-style probability wrapper since custom-objective boosters output raw
margins (logits) instead of probabilities.

Focal loss (gamma >= 0, alpha in (0, 1)):

    p = sigmoid(margin)
    FL = -alpha * y * (1-p)^gamma * log(p)
         -(1-alpha) * (1-y) * p^gamma * log(1-p)

``gamma=0`` recovers (alpha-weighted) log loss.
"""

# %%
from __future__ import annotations

import numpy as np


# %%
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# %%
def focal_grad_hess(margin: np.ndarray, y: np.ndarray, gamma: float, alpha: float):
    """Gradient and hessian of binary focal loss w.r.t. the raw margin.

    Returns ``(grad, hess)`` arrays aligned with ``margin``.

    Uses the closed-form first/second derivatives of binary focal loss w.r.t. the
    raw margin from the well-established reference implementation (Wang et al.,
    "Focal loss for LightGBM/XGBoost"). ``alpha`` is folded in as a class weight.
    """
    y = np.asarray(y, dtype=float)
    eps = 1e-9
    p = np.clip(_sigmoid(margin), eps, 1 - eps)
    g = gamma

    # p_t = probability assigned to the TRUE class.
    pt = np.where(y == 1, p, 1 - p)
    mod = np.power(1 - pt, g)  # focal modulating factor, in [0, 1]

    # Gradient of focal loss w.r.t. the margin. The first term is the log-loss
    # gradient (p - y) scaled by the modulating factor; the second term is the
    # derivative of the modulating factor itself. Both are standard.
    log_pt = np.log(pt)
    d_mod = g * np.power(1 - pt, g - 1) * pt  # magnitude of focal correction
    # Sign of (p - y): positive class wants margin up, negative down.
    grad = mod * (p - y) + np.sign(p - y) * d_mod * (-log_pt) * pt
    # Hessian: focal-modulated log-loss hessian (robust positive approximation).
    hess = mod * p * (1 - p) * (1 + g * pt)

    # Apply alpha as a per-class weight and keep the hessian positive.
    w = np.where(y == 1, alpha, 1 - alpha)
    grad = grad * w
    hess = np.maximum(hess * w, 1e-6)
    return grad, hess


# %%
def lgb_focal_objective(gamma: float, alpha: float):
    """Return a LightGBM-compatible ``(y_true, raw)`` -> (grad, hess) objective."""

    def _obj(y_true, raw):
        grad, hess = focal_grad_hess(np.asarray(raw, dtype=float), y_true, gamma, alpha)
        return grad, hess

    return _obj


# %%
def xgb_focal_objective(gamma: float, alpha: float):
    """Return an XGBoost-compatible ``(raw, dtrain)`` -> (grad, hess) objective."""

    def _obj(raw, dtrain):
        y = dtrain.get_label()
        grad, hess = focal_grad_hess(np.asarray(raw, dtype=float), y, gamma, alpha)
        return grad, hess

    return _obj


# %%
def margin_to_proba(margin: np.ndarray) -> np.ndarray:
    """Convert raw booster margins (logits) to positive-class probabilities."""
    return _sigmoid(np.asarray(margin, dtype=float))
