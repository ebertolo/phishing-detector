"""Explicit pairwise interaction features.

Gradient boosters capture interactions implicitly through tree splits, but an
explicit product of two features can help linear models (logistic regression) and
sometimes sharpen boosters. This transformer appends the pairwise products of a
chosen set of columns (default: all input columns) to the frame.

Deterministic and row-wise (learns only the column list at fit time), so it is
leakage-safe and composes at the front of any pipeline, typically after
``FeatureEngineer``.
"""

# %%
from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# %%
class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Append pairwise products of selected columns.

    Parameters
    ----------
    columns : list[str] | None
        Columns to cross. If None, all input columns are used. Crossing every
        pair of many columns is quadratic, so pass a short list for wide frames.
    keep_raw : bool
        Keep the original columns alongside the new ``a__x__b`` products.
    max_features : int | None
        Optional cap on the number of input columns actually crossed (the first
        ``max_features`` of ``columns``), to bound the quadratic blow-up.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        keep_raw: bool = True,
        max_features: int | None = 8,
    ) -> None:
        self.columns = columns
        self.keep_raw = keep_raw
        self.max_features = max_features

    # %%
    def fit(self, X: pd.DataFrame, y=None) -> "InteractionFeatures":
        X = pd.DataFrame(X)
        cols = self.columns if self.columns is not None else list(X.columns)
        cols = [c for c in cols if c in X.columns]
        if self.max_features is not None:
            cols = cols[: self.max_features]
        self.cross_columns_ = cols
        self.pairs_ = list(combinations(cols, 2))
        return self

    # %%
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for a, b in self.pairs_:
            out[f"{a}__x__{b}"] = X[a].astype(float) * X[b].astype(float)
        if self.keep_raw:
            return pd.concat([X.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        return out

    # %%
    def get_feature_names_out(self, input_features=None):
        return [f"{a}__x__{b}" for a, b in self.pairs_]
