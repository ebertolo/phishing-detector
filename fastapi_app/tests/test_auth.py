"""Tests for the Bearer-token authentication on protected endpoints."""

from __future__ import annotations

_PAYLOAD = {
    "samples": [
        {
            "num_words": 100,
            "num_unique_words": 60,
            "num_stopwords": 30,
            "num_links": 2,
            "num_unique_domains": 1,
            "num_email_addresses": 0,
            "num_spelling_errors": 1,
            "num_urgent_keywords": 0,
        }
    ]
}


def test_predict_without_authorization_header_is_rejected(api_client):
    resp = api_client.post("/predict", json=_PAYLOAD)
    assert resp.status_code in (401, 403)


def test_predict_with_wrong_token_is_rejected(api_client):
    resp = api_client.post(
        "/predict", json=_PAYLOAD, headers={"Authorization": "Bearer wrong-token"}
    )
    assert resp.status_code == 401


def test_predict_with_correct_token_is_accepted(api_client):
    resp = api_client.post(
        "/predict", json=_PAYLOAD, headers={"Authorization": "Bearer test-token"}
    )
    assert resp.status_code == 200
