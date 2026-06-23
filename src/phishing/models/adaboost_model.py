"""AdaBoost (decision-tree ensemble) model.

Added as an independent algorithm (the user's "AdamBoost" interpreted as
AdaBoost, an ensemble method, distinct from the NN optimizers). AdaBoost has no
``class_weight``; imbalance is handled with a balanced base estimator so the rare
positive class is not ignored.
"""

# %%
from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ._common import make_pipeline

NAME = "adaboost"


# %%
def build(feature_mode: str = "engineered", y=None, embedding_kwargs: dict | None = None):
    """Unfitted AdaBoost-over-trees pipeline for the given feature mode."""
    base = DecisionTreeClassifier(max_depth=3, class_weight="balanced")
    estimator = AdaBoostClassifier(estimator=base, random_state=42)
    return make_pipeline(estimator, feature_mode, embedding_kwargs)


# %%
def param_grid() -> dict:
    # Depth-1 stumps become "worse than random" under this extreme imbalance and
    # make AdaBoost abort, so the base trees keep enough depth (>=2) to be valid.
    return {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.5, 1.0],
        "model__estimator__max_depth": [2, 3],
    }
