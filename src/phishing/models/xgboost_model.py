"""XGBoost gradient-boosting model.

Classic booster; imbalance handled via ``scale_pos_weight``. Useful blend member
alongside LightGBM/CatBoost.
"""

# %%
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier

from ._common import gpu_available, make_pipeline, scale_pos_weight

NAME = "xgboost"


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted XGBoost pipeline for the given feature mode.

    Trains on GPU automatically when one is detected (e.g. a Colab GPU
    runtime); falls back to CPU otherwise. Set ``PHISHING_FORCE_CPU=1`` to
    force CPU on a GPU machine. ``tree_method="hist"`` is used either way
    (the modern XGBoost API selects GPU execution via ``device="cuda"``).
    """
    spw = scale_pos_weight(y) if y is not None else 1.0
    params = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    if gpu_available():
        params["device"] = "cuda"
    estimator = XGBClassifier(**params)
    return make_pipeline(estimator, feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [3, 6],
        "model__subsample": [0.8, 1.0],
    }


# %%
def param_distributions() -> dict:
    """Wider search space for RandomizedSearchCV."""
    return {
        "model__n_estimators": randint(200, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(3, 10),
        "model__subsample": uniform(0.6, 0.4),          # 0.6–1.0
        "model__colsample_bytree": uniform(0.6, 0.4),   # 0.6–1.0
        "model__min_child_weight": randint(1, 10),
        "model__reg_lambda": loguniform(1e-3, 10.0),
    }
