"""MLflow tracking wrapper for fit and inference observability.

Backend is a local file store by default (``file:./mlruns``); override with the
``MLFLOW_TRACKING_URI`` environment variable. Launch the UI with
``uv run mlflow ui``.

Two entry points:

- ``log_fit``       : per-model training run — params, CV metrics, validation/
                      test metrics, model artifact, diagnostic curves, and the
                      per-feature binning/WOE tables.
- ``log_inference`` : per-batch summary — sample count, model version, threshold,
                      predicted positive rate, and (evaluation mode) the metrics
                      against provided labels. No per-row sensitive data.
"""

# %%
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

DEFAULT_TRACKING_URI = "file:./mlruns"
FIT_EXPERIMENT = "phishing-fit"
INFERENCE_EXPERIMENT = "phishing-inference"


# %%
def configure(tracking_uri: str | None = None) -> str:
    """Point MLflow at the tracking store and return the resolved URI.

    The project uses the local file store (``file:./mlruns``). MLflow >= 3.x put
    the file store in "maintenance mode" and raises unless explicitly opted in,
    so we set ``MLFLOW_ALLOW_FILE_STORE`` for file-based URIs.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if uri.startswith("file:") or uri.startswith("./") or uri.startswith("mlruns"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    return uri


# %%
def _log_figures(figures: dict[str, Any]) -> None:
    """Log a name -> matplotlib Figure mapping as PNG artifacts."""
    for fname, fig in figures.items():
        mlflow.log_figure(fig, f"plots/{fname}.png")


# %%
def _log_tables(tables: dict[str, pd.DataFrame], subdir: str) -> None:
    """Log a name -> DataFrame mapping as CSV artifacts under ``subdir``."""
    if not tables:
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name, df in tables.items():
            path = Path(tmp) / f"{name}.csv"
            df.to_csv(path, index=True)
            mlflow.log_artifact(str(path), artifact_path=subdir)


# %%
def log_fit(
    model_name: str,
    params: dict[str, Any],
    cv_metrics: dict[str, float],
    val_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
    model: Any = None,
    figures: dict[str, Any] | None = None,
    woe_tables: dict[str, pd.DataFrame] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log one training run; returns the MLflow run id.

    Logs hyperparameters and the best combo, CV/validation/test metrics, the
    serialised model, diagnostic curves, and per-feature binning/WOE tables.
    """
    configure(tracking_uri)
    mlflow.set_experiment(FIT_EXPERIMENT)
    with mlflow.start_run(run_name=model_name) as run:
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params(_flatten(params))
        for key, value in cv_metrics.items():
            mlflow.log_metric(f"cv_{key}", float(value))
        if val_metrics:
            for key, value in _numeric_only(val_metrics).items():
                mlflow.log_metric(f"val_{key}", float(value))
        if test_metrics:
            for key, value in _numeric_only(test_metrics).items():
                mlflow.log_metric(f"test_{key}", float(value))
        if model is not None:
            try:
                mlflow.sklearn.log_model(model, name="model")
            except Exception:  # pragma: no cover - fallback for exotic estimators
                pass
        if figures:
            _log_figures(figures)
        if woe_tables:
            _log_tables(woe_tables, subdir="woe_tables")
        return run.info.run_id


# %%
def log_inference(
    model_version: str,
    n_samples: int,
    threshold: float,
    predicted_positive_rate: float,
    eval_metrics: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log a per-batch inference summary; returns the MLflow run id.

    No per-row data is logged — only aggregate counts, the model version, the
    threshold, the predicted positive rate, and (evaluation mode) metrics.
    """
    configure(tracking_uri)
    mlflow.set_experiment(INFERENCE_EXPERIMENT)
    with mlflow.start_run(run_name=f"infer:{model_version}") as run:
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("n_samples", int(n_samples))
        mlflow.log_metric("predicted_positive_rate", float(predicted_positive_rate))
        if eval_metrics:
            for key, value in _numeric_only(eval_metrics).items():
                mlflow.log_metric(key, float(value))
        return run.info.run_id


# %%
def _flatten(params: dict[str, Any]) -> dict[str, Any]:
    """Stringify nested values so MLflow accepts them as params."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


# %%
def _numeric_only(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep only scalar numeric metric values (drop confusion arrays etc.)."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out
