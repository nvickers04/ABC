"""Daily summary report (dashboard + recommendation merge)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override root conftest — cycle log tests use tmp_path only."""
    yield


@pytest.fixture(autouse=True)
def _isolated_cycle_logs(tmp_path, monkeypatch):
    from core import profit_cycle_logger as pcl

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)
    yield


def _entry(profile: str, pnl: float, *, hours_ago: int = 1) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "ts": ts,
        "session_date": ts[:10],
        "cycle_id": hours_ago,
        "profit_profile": profile,
        "outcome": "done",
        "pnl": {
            "cycle_realized_pnl_usd": pnl,
            "cumulative_realized_pnl_usd": 1000.0 + pnl,
            "llm_cost_usd": 0.05,
        },
        "trade_outcome": {"action": "done"},
    }


def test_merge_agreement():
    from core.daily_summary import merge_tomorrow_recommendation

    live = {"suggested_profile": "aggressive", "confidence": "high", "rationale": "live"}
    sim = {"mode": "grid", "best": {"base_profile": "aggressive", "candidate_id": "aggressive", "metrics": {}}}
    out = merge_tomorrow_recommendation(live, sim)
    assert out["recommended_profile"] == "aggressive"
    assert out["confidence"] == "high"
    assert out["agreement"] is True


def test_merge_live_none_uses_sim():
    from core.daily_summary import merge_tomorrow_recommendation

    live = {"suggested_profile": "balanced", "confidence": "none", "rationale": "no logs"}
    sim = {
        "best": {
            "base_profile": "conservative",
            "candidate_id": "conservative",
            "metrics": {"composite_score": 0.8},
        }
    }
    out = merge_tomorrow_recommendation(live, sim)
    assert out["recommended_profile"] == "conservative"
    assert out["confidence"] == "medium"


def test_run_daily_summary_with_mocked_sim(tmp_path):
    from core.daily_summary import run_daily_summary

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entries = [_entry("aggressive", 15.0, hours_ago=i + 1) for i in range(5)]
    path = tmp_path / f"profit_cycles_{day}.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    sim_payload = {
        "mode": "grid",
        "start_date": "2026-01-01",
        "end_date": "2026-01-07",
        "best": {
            "candidate_id": "aggressive",
            "base_profile": "aggressive",
            "metrics": {"composite_score": 0.9, "total_profit_usd": 500, "sharpe_ratio": 1.2},
        },
    }

    mock_cfg = MagicMock()
    with patch("core.daily_summary.run_sim_optimizer", return_value=sim_payload):
        with patch("core.central_profit_config.get_profit_config") as gp:
            gp.return_value.reload.return_value = mock_cfg
            with patch(
                "core.profit_cycle_logger.snapshot_profit_config",
                return_value={"profit_profile": "balanced", "trading_mode": "paper", "risk": {}},
            ):
                report = run_daily_summary(run_sim=True, dashboard_days=1)

    assert report["tomorrow"]["recommended_profile"] == "aggressive"
    assert report["dashboard"]["entries"] == 5
    assert "console_text" in report["dashboard"]


def test_format_daily_summary_report_includes_recommendation():
    from core.daily_summary import format_daily_summary_report

    text = format_daily_summary_report(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "tomorrow": {
                "session_date": "2026-01-02",
                "recommended_profile": "balanced",
                "confidence": "high",
                "source": "live+simulation",
                "rationale": "test",
            },
            "action": {"set_env": "PROFIT_PROFILE=balanced", "cli_trader": "python __main__.py --profit-profile balanced"},
            "dashboard": {"console_text": "dashboard block"},
            "live_optimization": {
                "lookback_days": 7,
                "entries_analyzed": 0,
                "suggest_for_session": "2026-01-02",
                "current_profit_profile": "balanced",
                "suggested_profile": "balanced",
                "confidence": "none",
                "rationale": "default",
                "min_cycles_per_profile": 3,
                "action": {"set_env": "PROFIT_PROFILE=balanced", "cli": "x"},
                "rankings": [],
            },
        }
    )
    assert "balanced" in text
    assert "DAILY SUMMARY" in text
