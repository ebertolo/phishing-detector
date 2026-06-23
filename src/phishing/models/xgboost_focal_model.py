"""XGBoost with a focal-loss objective.

Uses XGBoost's sklearn API with a custom focal-loss objective. Because a custom
objective makes the booster output raw margins, a thin wrapper applies a sigmoid
to yield probabilities for the generic pipeline.
"""

# %%
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from ._common import make_pipeline
from ._focal import focal_grad_hess, margin_to_proba

NAME = "xgboost_focal"


# %%
class _XGBFocal(BaseEstimator, ClassifierMixin):
    """XGBClassifier with a focal-loss objective and probability output."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, n_estimators: int = 400,
                 learning_rate: float = 0.05, max_depth: int = 6):
        self.gamma = gamma
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _objective(self, y_true, raw):
        # XGBClassifier custom objective signature: (y_true, y_pred).
        return focal_grad_hess(np.asarray(raw, dtype=float), y_true, self.gamma, self.alpha)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y = np.asarray(y).astype(int)
        # Seed the base margin at the prior log-odds (see lightgbm_focal_model):
        # custom objectives start at margin 0, which collapses under imbalance.
        prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self._init_score = float(np.log(prior / (1 - prior)))
        self.model_ = XGBClassifier(
            objective=self._objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            base_score=0.5,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        self.model_.fit(X, y, base_margin=np.full(len(y), self._init_score))
        return self

    def predict_proba(self, X):
        import numpy as _np
        margin = self.model_.predict(
            X, output_margin=True,
            base_margin=_np.full(len(X), self._init_score),
        )
        p = margin_to_proba(margin)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# %%
def build(feature_mode: str = "raw", y=None, embedding_kwargs: dict | None = None):
    """Unfitted focal-loss XGBoost pipeline for the given feature mode."""
    return make_pipeline(_XGBFocal(), feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    return {
        "model__gamma": [1.0, 2.0],
        "model__alpha": [0.25, 0.5],
        "model__n_estimators": [400],
        "model__max_depth": [6],
    }
