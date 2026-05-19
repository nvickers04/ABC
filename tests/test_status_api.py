"""Status API endpoint tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_status_endpoint_returns_profile():
    sample = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "overall_status": "healthy",
        "active_profile": "balanced",
        "alerts": [],
        "alert_counts": {"critical": 0, "warn": 0, "info": 0},
        "profit_config": {"key_levers": {}},
        "daily_summary": {"cycles": 0},
        "window_summary": {},
        "research_heartbeat": {},
        "safety": {},
        "simulation": {},
        "operating_context": {},
        "role": "trader",
    }
    with patch("core.observability.health_report.build_health_report", return_value=sample):
        from infra.status_api.app import app

        client = TestClient(app)
        resp = client.get("/status?role=trader")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_profile"] == "balanced"
    assert data["overall_status"] == "healthy"


def test_health_ready_unhealthy_returns_503():
    sample = {
        "overall_status": "unhealthy",
        "active_profile": "balanced",
        "alert_counts": {"critical": 1, "warn": 0, "info": 0},
    }
    with patch("core.observability.health_report.build_health_report", return_value=sample):
        from infra.status_api.app import app

        client = TestClient(app)
        resp = client.get("/health/ready")
    assert resp.status_code == 503
