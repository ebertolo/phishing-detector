"""Stacking: a logistic meta-model over base-model out-of-fold predictions.

Blending (``blending.py``) averages calibrated base probabilities with weights
tuned on validation. Stacking is the alternative the project guidance mentions:
fit a simple meta-model on the base models' **out-of-fold** predictions so the
meta-learner sees each base model's behaviour without leakage, then apply it.

Here the base models are already calibrated and we have their validation-set
probabilities; the logistic meta-model is fit on those validation probabilities
(the held-out set the base models were not trained on) and applied to test. This
keeps the implementation leakage-safe within the existing train/val/test split:
base models train on train, the meta-model trains on val, everything is reported
on test.
"""

# %%
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


# %%
@dataclass
class StackResult:
    """Fitted meta-model plus the base model names it consumes (column order)."""

    model_names: list[str]
    meta_model: LogisticRegression


# %%
def fit_stacker(
    y_val: np.ndarray,
    val_proba_matrix: np.ndarray,
    model_names: list[str],
) -> StackResult:
    """Fit a logistic meta-model on base-model validation probabilities.

    Parameters
    ----------
    y_val : array of {0, 1}
        Validation labels.
    val_proba_matrix : array, shape (n_val, n_models)
        Calibrated positive-class probabilities on validation, one column per
        base model (column order must match ``model_names``).
    model_names : list[str]
        Base model identifiers, aligned with the matrix columns.
    """
    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    meta.fit(np.asarray(val_proba_matrix, dtype=float), np.asarray(y_val).astype(int))
    return StackResult(model_names=list(model_names), meta_model=meta)


# %%
def stack_proba(stacker: StackResult, proba_matrix: np.ndarray) -> np.ndarray:
    """Positive-class probability from the meta-model over base probabilities."""
    return stacker.meta_model.predict_proba(np.asarray(proba_matrix, dtype=float))[:, 1]
