"""Standalone logistic regression.

Distinct from ``logreg_woe`` (which always forces a WOE front-end): this one
respects whatever ``feature_mode`` it is given (default ``engineered``), so it
serves as a plain, interpretable linear baseline over the engineered features.
Imbalance handled with balanced class weights.
"""

# %%
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from ._common import make_pipeline

NAME = "logreg"


# %%
def build(feature_mode: str = "engineered", y=None, embedding_kwargs: dict | None = None):
    """Unfitted logistic-regression pipeline for the given feature mode."""
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    return make_pipeline(estimator, feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    }
