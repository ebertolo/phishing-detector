"""Tests for POST /predict — the prediction payload and response shape."""

from __future__ import annotations

_AUTH = {"Authorization": "Bearer test-token"}


def _sample(**overrides):
    base = {
        "num_words": 100,
        "num_unique_words": 60,
        "num_stopwords": 30,
        "num_links": 2,
        "num_unique_domains": 1,
        "num_email_addresses": 0,
        "num_spelling_errors": 1,
        "num_urgent_keywords": 0,
    }
    base.update(overrides)
    return base


def test_predict_batch_returns_likelihood_per_sample(api_client):
    payload = {"samples": [_sample(), _sample(num_links=10, num_urgent_keywords=5)]}
    resp = api_client.post("/predict", json=payload, headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["predictions"]) == 2
    assert body["model_name"] == "lightgbm"
    assert body["model_version"]

    for pred in body["predictions"]:
        assert 0.0 <= pred["phishing_likelihood"] <= 1.0
        assert isinstance(pred["is_phishing"], bool)
        assert pred["is_phishing"] == (pred["phishing_likelihood"] >= pred["threshold_used"])


def test_predict_single_sample(api_client):
    resp = api_client.post("/predict", json={"samples": [_sample()]}, headers=_AUTH)
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 1


def test_predict_rejects_empty_batch(api_client):
    resp = api_client.post("/predict", json={"samples": []}, headers=_AUTH)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(api_client):
    incomplete = _sample()
    del incomplete["num_words"]
    resp = api_client.post("/predict", json={"samples": [incomplete]}, headers=_AUTH)
    assert resp.status_code == 422


def test_predict_rejects_negative_counts(api_client):
    resp = api_client.post(
        "/predict", json={"samples": [_sample(num_links=-1)]}, headers=_AUTH
    )
    assert resp.status_code == 422
