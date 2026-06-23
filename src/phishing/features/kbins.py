"""Quantile discretisation transformer.

A simpler, unsupervised alternative to optimal binning: ``KBinsDiscretizer`` with
quantile strategy produces stable bins per feature without using the target.
Ordinal-encoded so it stays compact and tree-friendly. Leakage-safe by
construction (quantiles are learned at fit time only).
"""

# %%
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


# %%
class QuantileBinner(BaseEstimator, TransformerMixin):
    """Quantile ``KBinsDiscretizer`` returning ordinal bin indices as a DataFrame."""

    def __init__(self, n_bins: int = 8) -> None:
        self.n_bins = n_bins

    # %%
    def fit(self, X: pd.DataFrame, y=None) -> "QuantileBinner":
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy="quantile",
            subsample=None,
        )
        self.discretizer_.fit(X)
        return self

    # %%
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        binned = self.discretizer_.transform(X)
        cols = [f"{c}_bin" for c in self.feature_names_in_]
        return pd.DataFrame(binned, columns=cols, index=X.index)

    # %%
    def get_feature_names_out(self, input_features=None):
        return [f"{c}_bin" for c in self.feature_names_in_]
