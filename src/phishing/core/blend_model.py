"""Serializable blend model.

A blend (``blending.py``) is otherwise a set of base ``ModelWrapper`` estimators
plus weights — not a single object that can be persisted with the existing
versioned ``save``/``load`` path. ``BlendModel`` packages the fitted base
estimators and their weights into one estimator exposing ``predict_proba`` /
``predict``, so the winning blend can be saved as a single joblib artifact and
loaded for inference exactly like any other model.

It contains already-fitted estimators, so it needs no ``fit``; it is a thin,
picklable container that reproduces the blended probability at inference time.
"""

# %%
from __future__ import annotations

import numpy as np

from .blending import blend_proba


# %%
class BlendModel:
    """Weighted average of several fitted base estimators' positive probabilities.

    Parameters
    ----------
    estimators : list
        Fitted estimators (each exposing ``predict_proba`` returning a 2-column
        array, e.g. the calibrated pipelines held by each ``ModelWrapper``).
    weights : array-like
        Non-negative blend weights aligned with ``estimators`` (normalised).
    names : list[str]
        Base model names, for metadata/inspection.
    """

    def __init__(self, estimators: list, weights, names: list[str]) -> None:
        self.estimators = list(estimators)
        self.weights = np.asarray(weights, dtype=float)
        self.names = list(names)

    # %%
    def _proba_matrix(self, X) -> np.ndarray:
        """Stack each base estimator's positive-class probability as a column."""
        cols = [est.predict_proba(X)[:, 1] for est in self.estimators]
        return np.column_stack(cols)

    # %%
    def predict_proba(self, X) -> np.ndarray:
        """sklearn-style 2-column probability array for the blended score."""
        p = blend_proba(self._proba_matrix(X), self.weights)
        return np.column_stack([1 - p, p])

    # %%
    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Hard 0/1 predictions at ``threshold``."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
