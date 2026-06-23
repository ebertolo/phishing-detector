"""CatBoost gradient-boosting model.

Robust booster; imbalance handled via ``auto_class_weights="Balanced"``. Runs
silently so it composes cleanly inside GridSearchCV.
"""

# %%
from __future__ import annotations

from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint

from ._common import gpu_available, make_pipeline

NAME = "catboost"


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted CatBoost pipeline for the given feature mode.

    Trains on GPU automatically when one is detected (e.g. a Colab GPU
    runtime); falls back to CPU otherwise. Set ``PHISHING_FORCE_CPU=1`` to
    force CPU on a GPU machine.
    """
    params = dict(
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    if gpu_available():
        params["task_type"] = "GPU"
    estimator = CatBoostClassifier(**params)
    return make_pipeline(estimator, feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__iterations": [200, 400],
        "model__learning_rate": [0.03, 0.1],
        "model__depth": [4, 6],
    }


# %%
def param_distributions() -> dict:
    """Wider search space for RandomizedSearchCV."""
    return {
        "model__iterations": randint(200, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__depth": randint(4, 9),
        "model__l2_leaf_reg": loguniform(1.0, 30.0),
    }
