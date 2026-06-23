"""Dataset loading and schema helpers.

Shared by the Streamlit UI and any future service. Knows the expected integer
count features and the target column ``label_1`` (1 = phishing), and decides
whether an input file is labelled (evaluation mode) or unlabelled (inference
mode).
"""

# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical schema for this project (integer count features + target).
TARGET = "label_1"
FEATURES = [
    "num_words",
    "num_unique_words",
    "num_stopwords",
    "num_links",
    "num_unique_domains",
    "num_email_addresses",
    "num_spelling_errors",
    "num_urgent_keywords",
]


# %%
def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or an uploaded file-like object."""
    return pd.read_csv(source)


# %%
def has_labels(df: pd.DataFrame, target: str = TARGET) -> bool:
    """True when the target column is present (evaluation mode)."""
    return target in df.columns


# %%
def split_X_y(df: pd.DataFrame, target: str = TARGET):
    """Split a labelled frame into ``(X, y)`` using the known feature set.

    Falls back to "all columns except target" if the canonical features are not
    all present, so the framework tolerates schema variations.
    """
    feature_cols = [c for c in FEATURES if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].astype(int) if target in df.columns else None
    return X, y


# %%
def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the feature columns (for inference on unlabelled input)."""
    feature_cols = [c for c in FEATURES if c in df.columns]
    if not feature_cols:
        feature_cols = list(df.columns)
    return df[feature_cols].copy()


# %%
def positive_rate(y) -> float:
    """Phishing rate of a label vector (for the imbalance summary)."""
    return float(pd.Series(y).astype(int).mean())
