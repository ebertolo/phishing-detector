"""Imbalance-aware classification metrics.

For a ~1% positive (phishing) problem, plain accuracy is meaningless and is
never reported as a headline metric. The metric set here focuses on the
positive/phishing class: PR-AUC (primary), ROC-AUC, precision, recall, F1, MCC,
and the confusion matrix at a chosen operating point.

Every function is small and independently runnable as a notebook cell.
"""

# %%
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# %%
@dataclass
class ClassificationMetrics:
    """Container for the imbalance-aware metric set at one operating threshold."""

    threshold: float
    pr_auc: float          # average precision — primary ranking metric
    roc_auc: float
    precision: float
    recall: float
    f1: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int
    n_samples: int
    n_positives: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def confusion(self) -> np.ndarray:
        """Confusion matrix as ``[[tn, fp], [fn, tp]]``."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])

    def as_dict(self) -> dict[str, Any]:
        """Flat dict, JSON-serialisable, suitable for metadata and MLflow."""
        d = asdict(self)
        d.pop("extra", None)
        d.update(self.extra)
        return d


# %%
def _to_label(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a decision threshold to probability/score values."""
    return (np.asarray(y_score) >= threshold).astype(int)


# %%
def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute the full imbalance-aware metric set at ``threshold``.

    Parameters
    ----------
    y_true : array of {0, 1}
        Ground-truth labels (1 = phishing).
    y_score : array of float in [0, 1]
        Predicted positive-class probabilities (or calibrated scores).
    threshold : float
        Operating point used for the label-based metrics and confusion matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = _to_label(y_score, threshold)

    # Ranking metrics are threshold-independent; guard the degenerate single-
    # class case so a fold with no positives does not raise.
    if len(np.unique(y_true)) < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, y_score))
        roc_auc = float(roc_auc_score(y_true, y_score))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        threshold=float(threshold),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if tp + fp + fn > 0 else 0.0,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        n_samples=int(y_true.size),
        n_positives=int(y_true.sum()),
    )


# %%
# Primary scorer name used by GridSearchCV. PR-AUC (average precision) is the
# refit metric for this imbalanced problem.
PRIMARY_SCORING = "average_precision"
SECONDARY_SCORING = "roc_auc"
