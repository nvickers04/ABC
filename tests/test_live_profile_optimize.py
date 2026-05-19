"""Live profile optimization from cycle logs."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from core.live_profile_optimize import (
    format_live_optimize_report,
    run_live_optimize,
    score_profile_entries,
)


@pytest.fixture(autouse=True)
def _isolated_cycle_logs(tmp_path, monkeypatch):
    """Keep live-optimize tests off real logs/ and Postgres."""
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


def test_score_profile_entries_composite():
    entries = [_entry("balanced", 10.0, hours_ago=i) for i in range(5)]
    scored = score_profile_entries(entries)
    assert scored["cycles"] == 5
    assert scored["composite_score"] >= 0


def test_run_live_optimize_suggests_best_builtin(tmp_path):
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    entries = []
    for i in range(5):
        entries.append(_entry("aggressive", 20.0, hours_ago=i + 1))
    for i in range(5):
        entries.append(_entry("conservative", -5.0, hours_ago=i + 10))
    path = tmp_path / f"profit_cycles_{day}.json"
    path.write_text(json.dumps({"date": day, "entries": entries}), encoding="utf-8")

    result = run_live_optimize(days=7, min_cycles=3)
    assert result["suggested_profile"] == "aggressive"
    assert "aggressive" in format_live_optimize_report(result)


def test_run_live_optimize_empty_logs():
    result = run_live_optimize(days=7)
    assert result["suggested_profile"] == "balanced"
    assert result["confidence"] == "none"
