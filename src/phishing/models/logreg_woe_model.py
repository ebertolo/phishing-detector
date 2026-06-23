"""Penalised logistic regression on WOE bins — the explainable baseline.

This model deliberately forces the ``binned_woe`` front-end regardless of the
requested feature mode: logistic regression over Weight-of-Evidence bins is the
most interpretable member of the roster, with monotone, auditable coefficients.
Imbalance handled via ``class_weight="balanced"``.
"""

# %%
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from ._common import make_pipeline

NAME = "logreg_woe"


# %%
def build(feature_mode: str = "binned_woe", y=None, embedding_kwargs: dict | None = None):
    """Unfitted logistic-regression-on-WOE pipeline.

    The WOE front-end is always applied for this model, so ``feature_mode`` is
    ignored and forced to ``"binned_woe"``. ``embedding_kwargs`` is accepted for
    interface consistency with the other models but never used (WOE never
    composes with the NN embedding post-step).
    """
    # L2 penalty is the default; setting it explicitly is deprecated in
    # scikit-learn >= 1.8, so we rely on the default and tune C in the grid.
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    return make_pipeline(estimator, feature_mode="binned_woe")


# %%
def param_grid() -> dict:
    return {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    }
