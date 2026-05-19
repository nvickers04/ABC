"""HTML dashboard export."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.dashboard_html import (
    build_daily_profile_rows,
    build_profile_timeseries,
    render_html_report,
    write_html_report,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def _sample_entries() -> list[dict]:
    now = datetime.now(timezone.utc)
    entries = []
    for i, (prof, pnl) in enumerate(
        [
            ("conservative", 5.0),
            ("balanced", -2.0),
            ("balanced", 8.0),
            ("aggressive", 3.0),
        ],
        start=1,
    ):
        ts = (now - timedelta(hours=4 - i)).isoformat()
        entries.append(
            {
                "ts": ts,
                "session_date": now.strftime("%Y-%m-%d"),
                "cycle_id": i,
                "profit_profile": prof,
                "outcome": "done",
                "pnl": {
                    "cycle_realized_pnl_usd": pnl,
                    "cumulative_realized_pnl_usd": 100.0 + pnl * i,
                    "llm_cost_usd": 0.1 * i,
                },
                "quality": {"overall_quality": "full", "risk_multiplier": 0.8},
                "trade_outcome": {"action": "done"},
            }
        )
    return entries


def test_build_profile_timeseries_running_sum():
    entries = _sample_entries()
    ts = build_profile_timeseries(entries)
    assert set(ts["profiles"]) == {"aggressive", "balanced", "conservative"}
    assert ts["by_profile"]["balanced"]["cumulative_profile_pnl"][-1] == 6.0


def test_build_daily_profile_rows():
    rows = build_daily_profile_rows(_sample_entries())
    assert len(rows) >= 3
    balanced = next(r for r in rows if r["profit_profile"] == "balanced")
    assert balanced["cycles"] == 2
    assert balanced["cycle_pnl_usd"] == 6.0


def test_render_html_contains_plotly_and_profiles():
    html = render_html_report(_sample_entries(), window_label="last 7 days")
    assert "plotly" in html.lower()
    assert "chart_cumulative" in html
    assert "conservative" in html
    assert "Cumulative cycle P&L by profile" in html


def test_write_html_report(tmp_path):
    out = tmp_path / "report.html"
    write_html_report(out, _sample_entries(), window_label="test")
    text = out.read_text(encoding="utf-8")
    assert out.is_file()
    data_marker = "const DATA = "
    assert data_marker in text
    payload = json.loads(text.split(data_marker, 1)[1].split(";", 1)[0])
    assert "balanced" in payload["profiles"]
