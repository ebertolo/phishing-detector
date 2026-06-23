"""Stratified data splitting and cross-validation helpers.

The ~1% phishing rate must be preserved in every split. Train feeds CV/GridSearch;
validation is held out for threshold tuning and blend-weight selection;
test is an untouched final holdout reported once.
"""

# %%
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# %%
@dataclass
class DataSplit:
    """Train / validation / test partition of features and labels."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def positive_rates(self) -> dict[str, float]:
        """Phishing rate per split — used to confirm stratification held."""
        return {
            "train": float(self.y_train.mean()),
            "val": float(self.y_val.mean()),
            "test": float(self.y_test.mean()),
        }


# %%
def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """Stratified train/val/test split preserving the positive rate.

    Two successive stratified splits: first carve out the test set, then split
    the remainder into train and validation.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # val_size is expressed relative to the full dataset; rescale to the remainder.
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=rel_val, stratify=y_rest, random_state=random_state
    )
    return DataSplit(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


# %%
def make_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Stratified K-fold splitter for GridSearchCV on the training set."""
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
