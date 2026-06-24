"""Tests for MODEL_VERSION_DIR environment-variable version selection.

Verifies that setting MODEL_VERSION_DIR before startup pins the API to exactly
that version rather than falling back to the most recently saved one.  This
covers the documented usage pattern::

    export MODEL_VERSION_DIR=models/lightgbm__20260619T230000Z
    uv run uvicorn fastapi_app.main:app --port 8000
"""

from __future__ import annotations

import importlib

import pytest

from phishing.core import data as data_mod
from phishing.core.splits import stratified_split
from phishing.core.wrapper import ModelWrapper
from phishing.experiments.runner import ThresholdConfig, run_experiments

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from make_sample_data import make_dataset  # noqa: E402

_AUTH = {"Authorization": "Bearer test-token"}


def _train_and_save(tmp_path, model_name: str, models_dir: Path, monkeypatch) -> Path:
    """Train a minimal model and save it; returns the version directory path."""
    monkeypatch.setattr(
        "phishing.core.param_cache.CACHE_DIR", tmp_path / "best_params"
    )
    monkeypatch.setattr(
        "phishing.core.embedding_cache.CACHE_DIR", tmp_path / "embeddings"
    )
    df = make_dataset(n=600, pos_rate=0.1, seed=42)
    X, y = data_mod.split_X_y(df)
    split = stratified_split(X, y, val_size=0.2, test_size=0.2, random_state=42)
    results, _ = run_experiments(
        split,
        model_names=[model_name],
        feature_mode="raw",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=False,
        n_splits=3,
        log_mlflow=False,
    )
    wrapper = results[0].wrapper
    return wrapper.save(models_dir=models_dir, test_metrics=results[0].test_metrics)


def test_model_version_dir_loads_pinned_version(tmp_path, monkeypatch):
    """When MODEL_VERSION_DIR is set, the app must serve exactly that version."""
    from fastapi.testclient import TestClient

    models_dir = tmp_path / "models"

    # Save two different model versions into the same models dir.
    v1 = _train_and_save(tmp_path, "lightgbm", models_dir, monkeypatch)
    v2 = _train_and_save(tmp_path, "catboost", models_dir, monkeypatch)

    # v2 is newer; without the pin the API would load it.
    assert v1.name != v2.name

    # Pin to v1 via the environment variable.
    monkeypatch.setenv("MODEL_VERSION_DIR", str(v1))
    monkeypatch.setenv("API_TOKEN", "test-token")

    import fastapi_app.main as main_mod
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        # The health endpoint exposes the loaded version name.
        assert resp.json()["model_version"] == v1.name

        # Predictions must also succeed with the pinned version.
        payload = {
            "samples": [{
                "num_words": 120, "num_unique_words": 80, "num_stopwords": 40,
                "num_links": 3, "num_unique_domains": 2, "num_email_addresses": 1,
                "num_spelling_errors": 2, "num_urgent_keywords": 1,
            }]
        }
        pred_resp = client.post("/predict", json=payload,
                                headers={"Authorization": "Bearer test-token"})
        assert pred_resp.status_code == 200
        assert pred_resp.json()["model_version"] == v1.name


def test_model_version_dir_missing_raises_on_startup(tmp_path, monkeypatch):
    """If MODEL_VERSION_DIR points to a non-existent path, startup must fail."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_VERSION_DIR", str(tmp_path / "nonexistent_version"))
    monkeypatch.setenv("API_TOKEN", "test-token")

    import fastapi_app.main as main_mod
    importlib.reload(main_mod)

    with pytest.raises(Exception):
        with TestClient(main_mod.app):
            pass
