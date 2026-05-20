"""Web dashboard API (password gate + data payload)."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-secret")
    from infra.status_api.app import app

    return TestClient(app)


def test_dashboard_disabled_without_password(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    from infra.status_api.app import app

    c = TestClient(app)
    r = c.get("/dashboard")
    assert r.status_code == 503


def test_dashboard_requires_auth(client):
    r = client.get("/dashboard")
    assert r.status_code == 401


def test_dashboard_html_with_auth(client):
    token = base64.b64encode(b"dashboard:test-secret").decode()
    r = client.get("/dashboard", headers={"Authorization": f"Basic {token}"})
    assert r.status_code == 200
    assert "ABC Profit Dashboard" in r.text
    assert "Chart" in r.text


def test_dashboard_data_json(client):
    token = base64.b64encode(b"admin:test-secret").decode()
    r = client.get("/dashboard/data", headers={"Authorization": f"Basic {token}"})
    assert r.status_code == 200
    body = r.json()
    assert "active_profile" in body
    assert "pnl_series" in body
    assert "sim_vs_live" in body
    assert "quality_matrix" in body


def test_build_dashboard_payload():
    from core.observability.dashboard_data import build_dashboard_payload

    payload = build_dashboard_payload(window_hours=24, chart_days=3)
    assert "research_heartbeat" in payload
    assert "pnl_series" in payload
    assert "labels" in payload["pnl_series"]
