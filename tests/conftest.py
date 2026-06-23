"""Shared pytest fixtures."""

import pytest


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
