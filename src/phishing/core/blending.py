"""Probability blending of calibrated base models.

A blend averages (optionally weighting) the calibrated positive-class
probabilities of several base models. Weights are optimised on the validation
set against PR-AUC; equal weights are the simple default. The blend is treated
as just another candidate model and flows through the same metrics/threshold/
persistence path.
"""

# %%
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score


# %%
@dataclass
class BlendResult:
    """Blend weights (aligned with ``model_names``) and validation PR-AUC."""

    model_names: list[str]
    weights: np.ndarray
    val_pr_auc: float


# %%
def blend_proba(proba_matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted average of per-model probabilities.

    Parameters
    ----------
    proba_matrix : array, shape (n_samples, n_models)
        Calibrated positive-class probabilities, one column per base model.
    weights : array, shape (n_models,), optional
        Non-negative weights; normalised internally. Equal weights if omitted.
    """
    proba_matrix = np.asarray(proba_matrix, dtype=float)
    n_models = proba_matrix.shape[1]
    if weights is None:
        weights = np.ones(n_models)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return proba_matrix @ weights


# %%
def optimize_weights(
    y_true: np.ndarray,
    proba_matrix: np.ndarray,
    model_names: list[str],
    n_grid: int = 11,
) -> BlendResult:
    """Search simplex weights maximising validation PR-AUC.

    For 2-3 models a coarse simplex grid is exhaustive and robust; for more
    models it stays tractable and degrades gracefully. Equal-weight average is
    always among the evaluated candidates.
    """
    y_true = np.asarray(y_true).astype(int)
    proba_matrix = np.asarray(proba_matrix, dtype=float)
    n_models = proba_matrix.shape[1]

    best_weights = np.ones(n_models) / n_models
    best_score = float(average_precision_score(y_true, blend_proba(proba_matrix, best_weights)))

    for w in _simplex_grid(n_models, n_grid):
        score = float(average_precision_score(y_true, blend_proba(proba_matrix, w)))
        if score > best_score:
            best_score = score
            best_weights = w

    return BlendResult(model_names=list(model_names), weights=best_weights, val_pr_auc=best_score)


# %%
def _simplex_grid(n_models: int, n_grid: int):
    """Yield weight vectors on a simplex grid summing to 1.

    Uses integer compositions of ``n_grid - 1`` into ``n_models`` parts.
    """
    levels = n_grid - 1

    def _compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in _compositions(total - first, parts - 1):
                yield (first, *rest)

    for comp in _compositions(levels, n_models):
        w = np.array(comp, dtype=float)
        if w.sum() == 0:
            continue
        yield w / w.sum()
