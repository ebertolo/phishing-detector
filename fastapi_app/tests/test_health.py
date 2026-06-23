"""Tests for GET /health."""

from __future__ import annotations


def test_health_requires_no_auth_and_reports_loaded_model(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "lightgbm"
    assert body["model_version"]


def test_health_does_not_need_authorization_header(api_client):
    # Explicitly confirm no header is required (distinct from /predict).
    resp = api_client.get("/health", headers={})
    assert resp.status_code == 200
