"""CatBoost with focal loss.

CatBoost supports focal loss natively via ``loss_function="Focal:focal_alpha=..;
focal_gamma=.."``, so unlike the LightGBM/XGBoost variants no custom-objective
wrapper is needed — CatBoost still outputs calibrated-ish probabilities directly.
"""

# %%
from __future__ import annotations

from catboost import CatBoostClassifier

from ._common import make_pipeline

NAME = "catboost_focal"


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted focal-loss CatBoost pipeline for the given feature mode."""
    estimator = CatBoostClassifier(
        loss_function="Focal:focal_alpha=0.25;focal_gamma=2.0",
        eval_metric="PRAUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    return make_pipeline(estimator, feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    # Focal params live in the loss_function string, so the grid tunes the tree
    # structure here; alpha/gamma swept via distinct loss_function values.
    return {
        "model__iterations": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__depth": [4, 6],
    }
