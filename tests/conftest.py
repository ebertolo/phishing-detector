"""Shared pytest fixtures."""

import os

import pytest

# Force the experiment runner's hyperparameter search to run sequentially during
# tests. On Windows, joblib/loky's parallel-worker teardown (the default
# n_jobs=-1) can crash the interpreter with a native StackOverflowException after
# the search finishes — the tests pass, but the process dies on exit (exit code
# 9/253). Setting this before any test imports/runs the runner avoids it and
# keeps `uv run pytest tests/ -v` green on Windows. Has no effect on production
# runs, which do not set this var.
os.environ.setdefault("PHISHING_SEARCH_N_JOBS", "1")


@pytest.fixture(autouse=True)
def isolate_param_cache(tmp_path, monkeypatch):
    """Redirect the best-params cache to a per-test tmp dir.

    Without this, any test that trains a model via ``run_model``/``run_experiments``
    (the default ``use_param_cache=True``) would write real entries into the
    project's ``best_params/`` directory — polluting it with cache keys from
    synthetic test data that would never match a real dataset anyway, but are
    still noise to clean up by hand.
    """
    import phishing.core.param_cache as param_cache_mod

    monkeypatch.setattr(param_cache_mod, "CACHE_DIR", tmp_path / "best_params")


@pytest.fixture(autouse=True)
def isolate_embedding_cache(tmp_path, monkeypatch):
    """Redirect the trained-embedding cache to a per-test tmp dir.

    Same rationale as ``isolate_param_cache``: anything that trains and caches an
    ``NNEmbedding`` during a test should not write into the project's real
    ``embeddings/`` directory.
    """
    import phishing.core.embedding_cache as embedding_cache_mod

    monkeypatch.setattr(embedding_cache_mod, "CACHE_DIR", tmp_path / "embeddings")
