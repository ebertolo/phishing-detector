"""Cross-fitted target (mean) encoding — leakage-safe.

Replaces each integer value with an estimate of P(phishing | value). Naive target
encoding leaks because a row contributes to its own encoding; this transformer
avoids that with **out-of-fold** encoding at fit time: the training data is split
into folds and each fold is encoded from statistics of the *other* folds, with
smoothing toward the global positive rate so rare values do not overfit.

At transform time (validation/test/inference) the full-train mapping is applied.
sklearn-compatible so it composes inside the model pipeline.
"""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


# %%
class CrossFitTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold mean-target encoder with smoothing.

    Parameters
    ----------
    n_splits : int
        Folds used for the out-of-fold encoding of the training data.
    smoothing : float
        Strength of the pull toward the global prior; higher = more shrinkage
        for low-count categories.
    random_state : int
        Reproducibility seed for the internal fold split.
    """

    def __init__(self, n_splits: int = 5, smoothing: float = 20.0, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    # %%
    def _smoothed_map(self, values: pd.Series, target: np.ndarray, prior: float) -> dict:
        """Smoothed mean-target per category: pulls small groups toward prior."""
        df = pd.DataFrame({"v": values.values, "y": target})
        stats = df.groupby("v")["y"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + prior * self.smoothing) / (
            stats["count"] + self.smoothing
        )
        return smooth.to_dict()

    # %%
    def fit(self, X: pd.DataFrame, y) -> "CrossFitTargetEncoder":
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        self.feature_names_in_ = list(X.columns)
        self.prior_ = float(y.mean())
        # Full-train mapping used at transform time.
        self.maps_ = {
            col: self._smoothed_map(X[col], y, self.prior_) for col in self.feature_names_in_
        }
        # Cache the out-of-fold training encoding so fit_transform is leakage-safe.
        self._oof_ = self._fit_transform_oof(X, y)
        return self

    # %%
    def _fit_transform_oof(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Out-of-fold encoding of the training rows."""
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        out = pd.DataFrame(index=X.index, columns=[f"{c}_te" for c in self.feature_names_in_], dtype=float)
        for train_idx, val_idx in skf.split(X, y):
            for col in self.feature_names_in_:
                fold_map = self._smoothed_map(
                    X[col].iloc[train_idx], y[train_idx], self.prior_
                )
                enc = X[col].iloc[val_idx].map(fold_map).fillna(self.prior_)
                out.loc[X.index[val_idx], f"{col}_te"] = enc.values
        return out.astype(float)

    # %%
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the full-train smoothed mapping (used on val/test/inference).

        Note: ``fit_transform`` returns the *out-of-fold* encoding for the
        training rows, so a sklearn Pipeline gets leakage-safe values at fit time
        and the stable full-train mapping at predict time.
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        out = {}
        for col in self.feature_names_in_:
            out[f"{col}_te"] = X[col].map(self.maps_[col]).fillna(self.prior_).values
        return pd.DataFrame(out, index=X.index)

    # %%
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self._oof_.copy()

    # %%
    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_te" for c in self.feature_names_in_])
