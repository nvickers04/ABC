"""Unit tests for core.observability.health_report (no Postgres)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_build_health_report_includes_profile_and_alerts(monkeypatch, tmp_path):
    from core import profit_cycle_logger as pcl
    from core.observability.health_report import build_health_report, overall_status_from_alerts, Alert

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)

    mock_risk = MagicMock()
    mock_risk.max_daily_loss_pct = 3.0
    mock_risk.intraday_drawdown_pct = 5.0
    mock_risk.max_daily_llm_cost = 10.0
    mock_cfg = MagicMock()
    mock_cfg.risk = mock_risk
    mock_cfg.trading_mode = "paper"

    with patch("core.central_profit_config.get_profit_config") as gp:
        gp.return_value.reload.return_value = mock_cfg
        with patch("core.profit_cycle_logger.snapshot_profit_config") as snap:
            snap.return_value = {
                "profit_profile": "balanced",
                "trading_mode": "paper",
                "risk": {"max_daily_loss_pct": 3.0},
                "loop": {"cooldown_seconds": 30},
                "memory": {},
                "prompt": {},
                "tools": {},
            }
            with patch(
                "core.observability.health_report._collect_research_heartbeat",
                return_value={"last_ts": None, "operational": False, "alive": False},
            ):
                report = build_health_report(role="trader")

    assert report["active_profile"] == "balanced"
    assert report["profit_config"]["key_levers"]["trading_mode"] == "paper"
    assert any(a["code"] == "research_heartbeat_missing" for a in report["alerts"])
    assert report["overall_status"] in ("degraded", "unhealthy")

    assert overall_status_from_alerts([Alert("x", "warn", "m")]) == "degraded"
    assert overall_status_from_alerts([Alert("x", "critical", "m")]) == "unhealthy"


def test_build_daily_summary(tmp_path, monkeypatch):
    import json
    from core import profit_cycle_logger as pcl
    from core.profit_cycle_logger import build_daily_summary

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    now = datetime.now(timezone.utc)
    d = now.strftime("%Y-%m-%d")
    entry = {
        "ts": now.isoformat(),
        "cycle_id": 1,
        "profit_profile": "balanced",
        "pnl": {"cycle_realized_pnl_usd": 5.0, "cumulative_realized_pnl_usd": 5.0, "llm_cost_usd": 0.1},
        "trade_outcome": {"action": "done"},
    }
    path = tmp_path / f"profit_cycles_{d}.json"
    path.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    summary = build_daily_summary(d)
    assert summary["cycles"] == 1
    assert summary["total_cycle_pnl_usd"] == 5.0


def test_safety_observe_snapshot():
    from core.runtime.safety import SafetyController, SafetyObserveSnapshot

    gw = MagicMock()
    gw.net_liquidation = 100_000
    gw.cash_value = 50_000
    ct = MagicMock()
    bs = MagicMock()
    bs.today_llm_cost = 8.5
    ct.get_budget_summary.return_value = bs

    ctrl = SafetyController(gw, ct, max_daily_loss_pct=3.0, intraday_drawdown_pct=5.0, max_daily_llm_cost=10.0)
    ctrl.capture_start_of_day_cash()
    snap = ctrl.observe(warn_ratio=0.85)
    assert isinstance(snap, SafetyObserveSnapshot)
    assert snap.near_limit.get("llm_cost") is True
    assert snap.to_dict()["thresholds"]["max_daily_llm_cost_usd"] == 10.0
