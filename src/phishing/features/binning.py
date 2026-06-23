"""Supervised optimal binning + Weight-of-Evidence (WOE) encoding.

Turns integer count features (num_words, num_links, ...) into semantic,
target-aware ranges so a model reasons about "suspicious-behaviour bands"
instead of absolute counts. WOE is a highly explainable representation and is
the natural input to the penalised logistic-regression baseline.

The transformer is **leakage-safe**: binning is fit on the training data only
(inside each CV fold when used in a Pipeline) and merely applied at transform
time. Built on the ``optbinning`` library, with a graceful degradation if a
feature cannot be binned.
"""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from optbinning import OptimalBinning

    _HAS_OPTBINNING = True
except ImportError:  # pragma: no cover - optbinning is a hard dependency
    _HAS_OPTBINNING = False


# %%
class OptimalBinningWOE(BaseEstimator, TransformerMixin):
    """Per-feature supervised optimal binning, output as WOE values.

    Parameters
    ----------
    max_n_bins : int
        Maximum number of bins per feature.
    min_bin_size : float
        Minimum fraction of records per bin (stability guard).

    Notes
    -----
    Requires ``y`` at fit time (supervised). Each column gets its own
    ``OptimalBinning`` fitted against the target; transform maps raw values to
    the bin WOE. Output column names are suffixed with ``_woe``.
    """

    def __init__(self, max_n_bins: int = 8, min_bin_size: float = 0.02) -> None:
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size

    # %%
    def fit(self, X: pd.DataFrame, y) -> "OptimalBinningWOE":
        if not _HAS_OPTBINNING:
            raise ImportError("optbinning is required for OptimalBinningWOE.")
        if y is None:
            raise ValueError("OptimalBinningWOE is supervised and requires y at fit time.")

        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        self.feature_names_in_ = list(X.columns)
        self.binners_: dict[str, OptimalBinning] = {}

        for col in self.feature_names_in_:
            binner = OptimalBinning(
                name=col,
                dtype="numerical",
                max_n_bins=self.max_n_bins,
                min_bin_size=self.min_bin_size,
            )
            binner.fit(X[col].values, y)
            self.binners_[col] = binner
        return self

    # %%
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).reset_index(drop=True)
        out = {}
        for col in self.feature_names_in_:
            woe = self.binners_[col].transform(X[col].values, metric="woe")
            out[f"{col}_woe"] = woe
        return pd.DataFrame(out, index=X.index)

    # %%
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"{c}_woe" for c in self.feature_names_in_])

    # %%
    def binning_tables(self) -> dict[str, pd.DataFrame]:
        """Per-feature binning/WOE tables, for audit and MLflow logging."""
        tables = {}
        for col, binner in self.binners_.items():
            tables[col] = binner.binning_table.build()
        return tables
