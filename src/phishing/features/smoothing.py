"""Feature-smoothing transformer for noisy, heavy-tailed counts.

Many raw count values are effectively noise (extreme outliers, long tails). This
transformer smooths the input so models see robust, denoised values:

- **winsorization** — clip each numeric feature to learned [p_low, p_high]
  percentiles, removing the influence of extreme outliers;
- **quantile** (optional) — additionally map the winsorized values through a
  rank/quantile transform to a smooth, uniform distribution.

Both steps learn their parameters on the training data only (percentiles and the
quantile mapping), so the transform is leakage-safe. Binary ``has_*`` flag
columns are passed through unchanged. This follows common robust-preprocessing
practice for fraud / rare-event tabular data, where winsorizing and rank
transforms tame noise without discarding signal.
"""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


# %%
class FeatureSmoothing(BaseEstimator, TransformerMixin):
    """Winsorize (and optionally quantile-transform) numeric features.

    Parameters
    ----------
    method : {"winsor", "winsor_quantile"}
        ``winsor`` clips outliers; ``winsor_quantile`` also applies a rank
        transform afterwards.
    lower, upper : float
        Winsorization percentiles (defaults clip the most extreme 1% each side).
    """

    def __init__(
        self,
        method: str = "winsor_quantile",
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> None:
        self.method = method
        self.lower = lower
        self.upper = upper

    # %%
    def _smoothable(self, X: pd.DataFrame) -> list[str]:
        """Columns to smooth: numeric, non-binary (skip has_* flags / 0-1 cols)."""
        cols = []
        for c in X.columns:
            s = X[c]
            if not np.issubdtype(s.dtype, np.number):
                continue
            if c.startswith("has_"):
                continue
            uniques = s.dropna().unique()
            if set(np.unique(uniques)).issubset({0, 1}):
                continue  # binary flag-like column
            cols.append(c)
        return cols

    # %%
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSmoothing":
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.smooth_cols_ = self._smoothable(X)
        self.bounds_ = {
            c: (float(X[c].quantile(self.lower)), float(X[c].quantile(self.upper)))
            for c in self.smooth_cols_
        }
        self.qt_ = None
        if self.method == "winsor_quantile" and self.smooth_cols_:
            clipped = self._winsorize(X)[self.smooth_cols_]
            n_q = min(1000, max(10, len(X)))
            self.qt_ = QuantileTransformer(
                n_quantiles=n_q, output_distribution="uniform", random_state=42
            )
            self.qt_.fit(clipped)
        return self

    # %%
    def _winsorize(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(X).copy()
        for c in self.smooth_cols_:
            lo, hi = self.bounds_[c]
            out[c] = out[c].clip(lower=lo, upper=hi)
        return out

    # %%
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._winsorize(pd.DataFrame(X))
        if self.qt_ is not None and self.smooth_cols_:
            out[self.smooth_cols_] = self.qt_.transform(out[self.smooth_cols_])
        return out

    # %%
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)
