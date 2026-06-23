"""RandomForest model — interpretable baseline and importance reference.

Imbalance handled via ``class_weight="balanced_subsample"``. Useful both as a
candidate model and as the source of feature importances in the UI.
"""

# %%
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from ._common import make_pipeline

NAME = "randomforest"


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted RandomForest pipeline for the given feature mode."""
    estimator = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return make_pipeline(estimator, feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_leaf": [1, 5],
    }
