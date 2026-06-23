"""Diagnostic plots logged to MLflow as artifacts.

Each function returns a matplotlib Figure so callers can log it or render it in
the UI. Headless-safe (Agg backend) so it works inside a container.
"""

# %%
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
)


# %%
def pr_curve_figure(y_true, y_score):
    """Precision-Recall curve with average precision in the legend."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ap = average_precision_score(y_true, y_score)
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name=f"AP={ap:.3f}")
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()
    return fig


# %%
def roc_curve_figure(y_true, y_score):
    """ROC curve (secondary diagnostic for imbalanced data)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    return fig


# %%
def confusion_figure(cm: np.ndarray):
    """Render a 2x2 confusion matrix ``[[tn, fp], [fn, tp]]``."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


# %%
def importance_figure(importances) -> "plt.Figure":
    """Horizontal bar chart of a feature-importance Series (descending)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    importances = importances.sort_values()
    ax.barh(importances.index, importances.values)
    ax.set_title("Feature importance")
    fig.tight_layout()
    return fig
