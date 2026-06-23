"""LightGBM with a focal-loss objective.

Same fast histogram booster as ``lightgbm_model`` but trained with focal loss
instead of weighted log loss, to concentrate learning on hard examples under
extreme imbalance. A thin sklearn wrapper converts the raw margins produced by a
custom objective into probabilities so it plugs into the generic pipeline.
"""

# %%
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from ._common import make_pipeline
from ._focal import lgb_focal_objective, margin_to_proba

NAME = "lightgbm_focal"


# %%
class _LGBMFocal(BaseEstimator, ClassifierMixin):
    """LGBMClassifier with a focal-loss objective and probability output."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, n_estimators: int = 400,
                 learning_rate: float = 0.05, num_leaves: int = 31):
        self.gamma = gamma
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y = np.asarray(y).astype(int)
        # Custom objectives start from margin 0; under extreme imbalance that
        # leaves every prediction near 0.5 and the focal gradient cannot recover.
        # Seed the base margin at the prior log-odds so training starts calibrated.
        prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self._init_score = float(np.log(prior / (1 - prior)))
        self.model_ = LGBMClassifier(
            objective=lgb_focal_objective(self.gamma, self.alpha),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self.model_.fit(X, y, init_score=np.full(len(y), self._init_score))
        return self

    def predict_proba(self, X):
        # raw_score=True returns margins; add the init score and map via sigmoid.
        margin = self.model_.predict(X, raw_score=True) + self._init_score
        p = margin_to_proba(margin)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted focal-loss LightGBM pipeline for the given feature mode."""
    return make_pipeline(_LGBMFocal(), feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__gamma": [1.0, 2.0],
        "model__alpha": [0.25, 0.5],
        "model__n_estimators": [400],
        "model__learning_rate": [0.05],
    }
