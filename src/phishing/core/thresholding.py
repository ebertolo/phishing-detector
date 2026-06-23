"""Operating-point (decision threshold) selection.

The threshold is a deliberate business choice, not a fixed 0.5. Three modes are
supported, all evaluated on validation scores:

- ``recall_target``  : smallest threshold reaching recall >= target while keeping
                       precision >= floor (cost of a false negative is high).
- ``max_f1``         : threshold maximising F1.
- ``manual``         : a user-provided threshold (the Streamlit slider).
"""

# %%
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


# %%
@dataclass
class ThresholdResult:
    """Chosen threshold plus the precision/recall achieved there on validation."""

    threshold: float
    mode: str
    precision: float
    recall: float
    note: str = ""


# %%
def _pr_grid(y_true: np.ndarray, y_score: np.ndarray):
    """Precision/recall/threshold arrays aligned for selection.

    ``precision_recall_curve`` returns one more precision/recall point than
    thresholds; drop the trailing point so all three arrays align.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return precision[:-1], recall[:-1], thresholds


# %%
def threshold_for_recall_target(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float = 0.90,
    precision_floor: float = 0.0,
) -> ThresholdResult:
    """Lowest-score threshold that reaches ``recall_target`` with precision >= floor.

    Among candidate thresholds meeting both constraints, pick the one with the
    highest precision (tightest operating point that still satisfies recall).
    Falls back to the max-recall point if the constraints cannot be met.
    """
    precision, recall, thresholds = _pr_grid(y_true, y_score)
    feasible = (recall >= recall_target) & (precision >= precision_floor)
    if feasible.any():
        idx_candidates = np.where(feasible)[0]
        best = idx_candidates[np.argmax(precision[idx_candidates])]
        return ThresholdResult(
            threshold=float(thresholds[best]),
            mode="recall_target",
            precision=float(precision[best]),
            recall=float(recall[best]),
        )
    # Constraints infeasible: choose the highest-recall point available.
    best = int(np.argmax(recall))
    return ThresholdResult(
        threshold=float(thresholds[best]),
        mode="recall_target",
        precision=float(precision[best]),
        recall=float(recall[best]),
        note="recall_target/precision_floor infeasible; using max-recall point",
    )


# %%
def threshold_for_max_f1(
    y_true: np.ndarray, y_score: np.ndarray
) -> ThresholdResult:
    """Threshold maximising F1 on the validation scores."""
    precision, recall, thresholds = _pr_grid(y_true, y_score)
    denom = precision + recall
    # Compute F1 only where the denominator is non-zero to avoid 0/0 warnings.
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    best = int(np.argmax(f1))
    return ThresholdResult(
        threshold=float(thresholds[best]),
        mode="max_f1",
        precision=float(precision[best]),
        recall=float(recall[best]),
    )


# %%
def manual_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> ThresholdResult:
    """Wrap a user-chosen threshold, reporting precision/recall achieved there."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return ThresholdResult(
        threshold=float(threshold),
        mode="manual",
        precision=precision,
        recall=recall,
    )


# %%
def threshold_for_min_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> ThresholdResult:
    """Cost-sensitive threshold minimising ``fn_cost*FN + fp_cost*FP``.

    For phishing, a missed phishing email (false negative) is typically far more
    costly than a false alarm (false positive), so ``fn_cost`` defaults to 10x
    ``fp_cost``. Sweeps the candidate thresholds from the PR curve and picks the
    one with the lowest total expected cost.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    precision, recall, thresholds = _pr_grid(y_true, y_score)

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    best_idx, best_cost = 0, float("inf")
    for i, thr in enumerate(thresholds):
        tp = recall[i] * n_pos
        fn = n_pos - tp
        # precision = tp / (tp + fp)  =>  fp = tp*(1-precision)/precision
        fp = (tp * (1 - precision[i]) / precision[i]) if precision[i] > 0 else n_neg
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost, best_idx = cost, i

    return ThresholdResult(
        threshold=float(thresholds[best_idx]),
        mode="cost",
        precision=float(precision[best_idx]),
        recall=float(recall[best_idx]),
        note=f"min expected cost (FN={fn_cost}, FP={fp_cost})",
    )


# %%
def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mode: str = "max_f1",
    recall_target: float = 0.90,
    precision_floor: float = 0.0,
    manual_value: float = 0.5,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> ThresholdResult:
    """Dispatch to the requested threshold-selection mode."""
    if mode == "recall_target":
        return threshold_for_recall_target(
            y_true, y_score, recall_target, precision_floor
        )
    if mode == "max_f1":
        return threshold_for_max_f1(y_true, y_score)
    if mode == "manual":
        return manual_threshold(y_true, y_score, manual_value)
    if mode == "cost":
        return threshold_for_min_cost(y_true, y_score, fn_cost, fp_cost)
    raise ValueError(f"Unknown threshold mode: {mode!r}")
