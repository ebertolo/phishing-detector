"""XGBoost focal loss via the **native** ``xgb.train`` API.

Counterpart to ``lightgbm_focal_native``: uses ``xgboost.train`` / ``DMatrix``
directly (with ``base_margin`` for the prior log-odds and a custom focal
objective) instead of the sklearn wrapper, which was the unstable part of the
earlier ``xgboost_focal``. Kept as a separate, comparable model.
"""

# %%
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ._common import make_pipeline
from ._focal import focal_grad_hess, margin_to_proba

NAME = "xgboost_focal_native"


# %%
class _XGBFocalNative(BaseEstimator, ClassifierMixin):
    """Native-API XGBoost trained with a focal-loss objective."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 n_estimators: int = 400, learning_rate: float = 0.05,
                 max_depth: int = 6):
        self.gamma = gamma
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        import xgboost as xgb

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self._init_score = float(np.log(prior / (1 - prior)))
        gamma, alpha = self.gamma, self.alpha

        def obj(preds, dtrain):
            # Native obj signature: (raw_preds, DMatrix) -> (grad, hess).
            return focal_grad_hess(np.asarray(preds, dtype=float),
                                   dtrain.get_label(), gamma, alpha)

        dtrain = xgb.DMatrix(
            X, label=y.astype(float),
            base_margin=np.full(len(y), self._init_score),
        )
        params = {
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "tree_method": "hist",
            "seed": 42,
        }
        self.booster_ = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators, obj=obj,
        )
        return self

    def predict_proba(self, X):
        import xgboost as xgb

        X = np.asarray(X, dtype=float)
        d = xgb.DMatrix(X, base_margin=np.full(len(X), self._init_score))
        margin = self.booster_.predict(d, output_margin=True)
        p = margin_to_proba(margin)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # %%
    def __getstate__(self):
        state = self.__dict__.copy()
        booster = state.pop("booster_", None)
        if booster is not None:
            state["_booster_raw"] = booster.save_raw()
        return state

    def __setstate__(self, state):
        import xgboost as xgb

        raw = state.pop("_booster_raw", None)
        self.__dict__.update(state)
        if raw is not None:
            self.booster_ = xgb.Booster()
            self.booster_.load_model(bytearray(raw))


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted native-focal XGBoost pipeline for the given feature mode."""
    return make_pipeline(_XGBFocalNative(), feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__gamma": [1.0, 2.0],
        "model__alpha": [0.25, 0.5],
        "model__n_estimators": [400],
        "model__max_depth": [6],
    }
