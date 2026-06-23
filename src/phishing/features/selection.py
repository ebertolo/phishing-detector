"""Feature-importance / selection helpers.

Mutual information gives a quick, model-free ranking of how informative each
integer feature is about the phishing label. RandomForest importances give a
model-based view. Both are used by the Streamlit "feature importance" step and
are independently runnable as notebook cells.
"""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


# %%
def mutual_information(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> pd.Series:
    """Mutual information of each feature with the target, sorted descending."""
    mi = mutual_info_classif(X, np.asarray(y), random_state=random_state)
    return pd.Series(mi, index=X.columns, name="mutual_information").sort_values(
        ascending=False
    )


# %%
def random_forest_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 300,
    random_state: int = 42,
) -> pd.Series:
    """RandomForest impurity-based importances, sorted descending.

    ``class_weight='balanced'`` keeps the rare positive class from being ignored.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, np.asarray(y))
    return pd.Series(
        rf.feature_importances_, index=X.columns, name="rf_importance"
    ).sort_values(ascending=False)
