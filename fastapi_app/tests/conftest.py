"""Shared fixtures for the FastAPI test suite.

Trains a tiny model on synthetic data, saves it to an isolated temporary
``models/`` directory, and points ``MODEL_VERSION_DIR`` at it before the app
is imported/instantiated — so tests never touch the project's real
``models/``/``best_params/``/``embeddings/`` directories or require a
pre-trained model to exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from make_sample_data import make_dataset  # noqa: E402

from phishing.core import data as data_mod
from phishing.core.splits import stratified_split
from phishing.experiments.runner import ThresholdConfig, run_experiments


@pytest.fixture()
def trained_model_version_dir(tmp_path, monkeypatch):
    """Train one fast model on synthetic data and save it to a temp dir."""
    monkeypatch.setattr(
        "phishing.core.param_cache.CACHE_DIR", tmp_path / "best_params"
    )
    monkeypatch.setattr(
        "phishing.core.embedding_cache.CACHE_DIR", tmp_path / "embeddings"
    )

    df = make_dataset(n=600, pos_rate=0.1, seed=1)
    X, y = data_mod.split_X_y(df)
    split = stratified_split(X, y, val_size=0.2, test_size=0.2, random_state=1)

    results, _ = run_experiments(
        split,
        model_names=["lightgbm"],
        feature_mode="raw",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=False,
        n_splits=3,
        log_mlflow=False,
    )
    wrapper = results[0].wrapper
    models_dir = tmp_path / "models"
    version_dir = wrapper.save(models_dir=models_dir, test_metrics=results[0].test_metrics)
    return version_dir


@pytest.fixture()
def api_client(trained_model_version_dir, monkeypatch):
    """A TestClient wired to the trained-model fixture via MODEL_VERSION_DIR."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_VERSION_DIR", str(trained_model_version_dir))
    monkeypatch.setenv("API_TOKEN", "test-token")

    # Import after env vars are set so the lifespan loads the right version.
    from fastapi_app.main import app

    with TestClient(app) as client:
        yield client
