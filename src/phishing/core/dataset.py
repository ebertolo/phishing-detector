"""Stratified dataset preparation for headless (CLI) runs.

Implements the project's split convention: hold out the **last 5%** as the
validation set and split the remaining **95%** into train and test — every
partition stratified so the rare phishing rate is identical across train, test
and validation. Returned as the same ``DataSplit`` the experiment runner
consumes.
"""

# %%
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

from .data import TARGET, split_X_y
from .splits import DataSplit


# %%
def stratified_sample(
    df: pd.DataFrame,
    n: int,
    target: str = TARGET,
    random_state: int = 42,
) -> pd.DataFrame:
    """Down-sample to ~``n`` rows while preserving the target distribution.

    Useful for fast experiment iteration on a large dataset: the returned sample
    keeps the same phishing rate as the full data (stratified by ``target``).
    Returns the full frame unchanged if ``n`` >= len(df).
    """
    if n >= len(df) or target not in df.columns:
        return df.reset_index(drop=True)
    frac = n / len(df)
    sample = (
        df.groupby(target, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=random_state))
        .reset_index(drop=True)
    )
    return sample


# %%
def stratified_95_5_split(
    df: pd.DataFrame,
    val_fraction: float = 0.05,
    test_fraction_within_95: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Stratified split: 5% validation, the rest split into train/test.

    Parameters
    ----------
    df : DataFrame
        Labelled dataset (must contain the target column).
    val_fraction : float
        Fraction of the whole dataset held out for validation (default 0.05).
    test_fraction_within_95 : float
        Fraction *of the remaining 95%* used as the test set. The default 0.20
        gives roughly a 76/19/5 train/test/val split of the whole dataset.
    random_state : int
        Reproducibility seed.

    Notes
    -----
    All three splits preserve the target distribution (``stratify=y``), so the
    ~1.3% phishing rate is identical in train, test and validation.
    """
    X, y = split_X_y(df)
    if y is None:
        raise ValueError("stratified_95_5_split requires a labelled dataset.")

    # Step 1 — carve out the 5% validation set, stratified.
    X_95, X_val, y_95, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=random_state
    )

    # Step 2 — split the remaining 95% into train and test, stratified.
    X_train, X_test, y_train, y_test = train_test_split(
        X_95,
        y_95,
        test_size=test_fraction_within_95,
        stratify=y_95,
        random_state=random_state,
    )

    return DataSplit(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
