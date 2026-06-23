"""LightGBM focal loss via the **native** ``lgb.train`` API.

The earlier `lightgbm_focal` model drives a focal objective through the sklearn
``LGBMClassifier`` wrapper and proved unstable at this ~1.3% imbalance. This
variant uses the native ``lgb.train`` / ``lgb.Dataset`` API directly, which gives
full control over the base margin (``init_score``) and a cleaner training loop.
It is kept as a separate, comparable model — the original is preserved.

A thin sklearn-compatible wrapper exposes ``fit``/``predict_proba`` so it runs in
the existing pipeline/runner; the native Booster is serialised via its own
``model_to_string`` so the wrapper still round-trips through joblib.
"""

# %%
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ._common import make_pipeline
from ._focal import focal_grad_hess, margin_to_proba

NAME = "lightgbm_focal_native"


# %%
class _LGBMFocalNative(BaseEstimator, ClassifierMixin):
    """Native-API LightGBM trained with a focal-loss objective."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 n_estimators: int = 400, learning_rate: float = 0.05,
                 num_leaves: int = 31):
        self.gamma = gamma
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def fit(self, X, y):
        import lightgbm as lgb

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        # Seed the base margin at the prior log-odds so training starts calibrated
        # (custom objectives otherwise begin at margin 0 and collapse under imbalance).
        prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self._init_score = float(np.log(prior / (1 - prior)))
        gamma, alpha = self.gamma, self.alpha

        def fobj(preds, dataset):
            # Native fobj signature: (raw_preds, Dataset) -> (grad, hess).
            return focal_grad_hess(np.asarray(preds, dtype=float),
                                   dataset.get_label(), gamma, alpha)

        dtrain = lgb.Dataset(
            X, label=y.astype(float),
            init_score=np.full(len(y), self._init_score),
        )
        params = {
            # LightGBM 4.x: the custom objective is passed via params, not fobj.
            "objective": fobj,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "verbosity": -1,
            "seed": 42,
        }
        self.booster_ = lgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        margin = self.booster_.predict(X) + self._init_score
        p = margin_to_proba(margin)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # %%
    def __getstate__(self):
        state = self.__dict__.copy()
        booster = state.pop("booster_", None)
        if booster is not None:
            state["_booster_str"] = booster.model_to_string()
        return state

    def __setstate__(self, state):
        import lightgbm as lgb

        booster_str = state.pop("_booster_str", None)
        self.__dict__.update(state)
        if booster_str is not None:
            self.booster_ = lgb.Booster(model_str=booster_str)


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted native-focal LightGBM pipeline for the given feature mode."""
    return make_pipeline(_LGBMFocalNative(), feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__gamma": [1.0, 2.0],
        "model__alpha": [0.25, 0.5],
        "model__n_estimators": [400],
        "model__learning_rate": [0.05],
    }
