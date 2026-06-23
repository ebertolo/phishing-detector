"""LightGBM gradient-boosting model.

Fast tabular booster; handles the ~1% imbalance via ``scale_pos_weight``
(negatives/positives). Strong candidate for the best individual performer and a
good blend member.
"""

# %%
from __future__ import annotations

from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform

from ._common import lightgbm_cuda_available, make_pipeline, scale_pos_weight

NAME = "lightgbm"


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted LightGBM pipeline for the given feature mode.

    Trains on GPU only when this LightGBM build actually has CUDA support
    (the default PyPI wheel does not — see ``lightgbm_cuda_available()``).
    Using the generic OpenCL ``device="gpu"`` is deliberately avoided: on a
    machine with both an NVIDIA GPU and an integrated GPU it can silently pick
    the integrated one, which is slower than CPU, not faster. Falls back to
    CPU otherwise. Set ``PHISHING_FORCE_CPU=1`` to force CPU on a GPU machine.
    """
    spw = scale_pos_weight(y) if y is not None else 1.0
    params = dict(
        objective="binary",
        scale_pos_weight=spw,
        subsample_freq=1,  # enable bagging so subsample in the search takes effect
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    if lightgbm_cuda_available():
        params["device"] = "cuda"
    estimator = LGBMClassifier(**params)
    return make_pipeline(estimator, feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    """GridSearchCV grid (pipeline-prefixed by the ``model`` step)."""
    return {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.03, 0.1],
        "model__num_leaves": [15, 31],
        "model__min_child_samples": [20, 50],
    }


# %%
def param_distributions() -> dict:
    """Wider search space for RandomizedSearchCV (covers more of the space)."""
    return {
        "model__n_estimators": randint(200, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": randint(15, 96),
        "model__min_child_samples": randint(10, 100),
        "model__subsample": uniform(0.6, 0.4),          # 0.6–1.0
        "model__colsample_bytree": uniform(0.6, 0.4),   # 0.6–1.0
        "model__reg_lambda": loguniform(1e-3, 10.0),
    }
