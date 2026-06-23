"""Generic model wrapper.

A single class wraps any estimator or pipeline so the rest of the application is
agnostic to the underlying algorithm. It exposes uniform ``fit`` / ``predict`` /
``predict_proba`` / ``validate`` / ``metrics`` / ``save`` / ``load`` methods and
is importable by both the Streamlit UI now and a future FastAPI service — there
are no UI/framework imports here.
"""

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import ClassificationMetrics, compute_metrics
from .persistence import ModelMetadata, load_model, save_model
from .thresholding import select_threshold


# %%
class ModelWrapper:
    """Algorithm-agnostic wrapper around a single estimator or a pipeline.

    Parameters
    ----------
    estimator : object
        Any fitted-or-unfitted sklearn-compatible estimator or ``Pipeline``
        (including imbalanced-learn pipelines). May also be a blend object that
        exposes ``predict_proba``.
    name : str
        Human-readable model/pipeline name, used in persistence and the UI.
    feature_mode : str
        ``"raw"`` or ``"binned_woe"`` — records how features were prepared so a
        saved version can be reproduced.
    threshold : float
        Decision threshold; defaults to 0.5 until tuned on validation.
    """

    def __init__(
        self,
        estimator: Any,
        name: str = "model",
        feature_mode: str = "raw",
        threshold: float = 0.5,
    ) -> None:
        self.estimator = estimator
        self.name = name
        self.feature_mode = feature_mode
        self.threshold = float(threshold)
        self.threshold_mode = "default"
        self.feature_names_: list[str] | None = None

    # %%
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelWrapper":
        """Fit the wrapped estimator/pipeline."""
        self.feature_names_ = list(X.columns)
        self.estimator.fit(X, y)
        return self

    # %%
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability for each sample.

        Uses ``predict_proba`` when available, otherwise falls back to a
        min-max normalised ``decision_function``.
        """
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)[:, 1]
        if hasattr(self.estimator, "decision_function"):
            scores = np.asarray(self.estimator.decision_function(X), dtype=float)
            lo, hi = scores.min(), scores.max()
            if hi > lo:
                return (scores - lo) / (hi - lo)
            return np.full_like(scores, 0.5)
        raise AttributeError(
            f"{type(self.estimator).__name__} exposes neither predict_proba "
            "nor decision_function."
        )

    # %%
    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Hard 0/1 predictions at ``threshold`` (defaults to the tuned one)."""
        thr = self.threshold if threshold is None else float(threshold)
        return (self.predict_proba(X) >= thr).astype(int)

    # %%
    def set_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        mode: str = "max_f1",
        recall_target: float = 0.90,
        precision_floor: float = 0.0,
        manual_value: float = 0.5,
        fn_cost: float = 10.0,
        fp_cost: float = 1.0,
    ) -> float:
        """Tune and store the decision threshold from validation scores."""
        result = select_threshold(
            y_true,
            y_score,
            mode=mode,
            recall_target=recall_target,
            precision_floor=precision_floor,
            manual_value=manual_value,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        self.threshold = result.threshold
        self.threshold_mode = result.mode
        return self.threshold

    # %%
    def metrics(self, X: pd.DataFrame, y: pd.Series) -> ClassificationMetrics:
        """Imbalance-aware metrics for ``X``/``y`` at the current threshold."""
        return compute_metrics(np.asarray(y), self.predict_proba(X), self.threshold)

    # %%
    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Compute metrics and return them as a flat, serialisable dict."""
        return self.metrics(X, y).as_dict()

    # %%
    def save(self, models_dir: Path | str = "models", **metadata_extra: Any) -> Path:
        """Persist the wrapped model and a self-describing metadata record."""
        meta = ModelMetadata(
            name=self.name,
            algorithm=type(self.estimator).__name__,
            created_at="",
            feature_names=self.feature_names_ or [],
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            feature_mode=self.feature_mode,
            extra=metadata_extra,
        )
        return save_model(self.estimator, meta, models_dir=models_dir)

    # %%
    @classmethod
    def load(cls, version_dir: Path | str) -> "ModelWrapper":
        """Reconstruct a wrapper from a saved version folder."""
        estimator, metadata = load_model(version_dir)
        wrapper = cls(
            estimator,
            name=metadata.get("name", "model"),
            feature_mode=metadata.get("feature_mode", "raw"),
            threshold=metadata.get("threshold", 0.5),
        )
        wrapper.threshold_mode = metadata.get("threshold_mode", "default")
        wrapper.feature_names_ = metadata.get("feature_names")
        return wrapper
