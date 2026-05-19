"""Profit summary API."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_profit_summary_endpoint(tmp_path, monkeypatch):
    from core import profit_cycle_logger as pcl

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)

    now = datetime.now(timezone.utc)
    entry = {
        "ts": now.isoformat(),
        "session_date": now.strftime("%Y-%m-%d"),
        "cycle_id": 1,
        "profit_profile": "balanced",
        "trading_mode": "paper",
        "outcome": "done",
        "pnl": {
            "cycle_realized_pnl_usd": 10.0,
            "cumulative_realized_pnl_usd": 110.0,
            "llm_cost_usd": 0.05,
            "intraday_drawdown_pct": 0.5,
        },
        "quality": {"overall_quality": "full", "risk_multiplier": 0.8},
        "trade_outcome": {"action": "done"},
    }
    path = tmp_path / f"profit_cycles_{now.strftime('%Y-%m-%d')}.json"
    path.write_text(json.dumps({"date": now.strftime("%Y-%m-%d"), "entries": [entry]}), encoding="utf-8")

    from api.app import app

    client = TestClient(app)
    resp = client.get("/profit_summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["window_hours"] == 24
    assert data["active_config"]["profit_profile"]
    assert data["stats_24h"]["cycles"] >= 1
    assert data["stats_24h"]["total_cycle_pnl_usd"] == 10.0


def test_status_endpoint_delegates_to_health_report():
    from unittest.mock import patch

    with patch(
        "core.observability.health_report.build_health_report",
        return_value={"overall_status": "healthy", "active_profile": "balanced", "alerts": []},
    ):
        from api.app import app

        client = TestClient(app)
        resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["active_profile"] == "balanced"


def test_load_entries_since_filters_old(tmp_path, monkeypatch):
    from core import profit_cycle_logger as pcl
    from core.profit_cycle_logger import load_entries_since

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)

    old = datetime.now(timezone.utc) - timedelta(hours=48)
    new = datetime.now(timezone.utc) - timedelta(hours=1)
    for ts, cid in ((old, 1), (new, 2)):
        row = {
            "ts": ts.isoformat(),
            "cycle_id": cid,
            "profit_profile": "balanced",
            "pnl": {"cycle_realized_pnl_usd": 0},
        }
        d = ts.strftime("%Y-%m-%d")
        p = tmp_path / f"profit_cycles_{d}.json"
        p.write_text(json.dumps({"entries": [row]}), encoding="utf-8")

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    rows = load_entries_since(since)
    assert all(r.get("cycle_id") != 1 for r in rows)
    assert any(r.get("cycle_id") == 2 for r in rows)
