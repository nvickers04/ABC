"""Build JSON profit summaries for API and dashboards."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from core.central_profit_config import get_profit_config
from core.profit_cycle_logger import load_entries_since, snapshot_profit_config


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def aggregate_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize cycle log entries (same shape as scripts/dashboard.py)."""
    if not entries:
        return {
            "cycles": 0,
            "total_cycle_pnl_usd": 0.0,
            "cumulative_realized_pnl_usd": 0.0,
            "llm_cost_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "trade_like_cycles": 0,
            "outcomes": {},
            "by_profile": {},
            "last_quality": {},
        }

    sorted_entries = sorted(
        entries,
        key=lambda e: e.get("ts") or "",
    )
    total_cycle_pnl = sum(
        float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0) for e in sorted_entries
    )
    last = sorted_entries[-1]
    cum_pnl = float(last.get("pnl", {}).get("cumulative_realized_pnl_usd", 0) or 0)
    llm = float(last.get("pnl", {}).get("llm_cost_usd", 0) or 0)
    wins = 0
    trades = 0
    for e in sorted_entries:
        pnl_delta = float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
        action = (e.get("trade_outcome") or {}).get("action", "")
        if action and action not in ("quality_status", "briefing", "market_hours", ""):
            if pnl_delta != 0 or (e.get("trade_outcome") or {}).get("order_id"):
                trades += 1
                if pnl_delta > 0:
                    wins += 1
    win_rate = (wins / trades * 100.0) if trades else 0.0
    by_profile: dict[str, dict[str, Any]] = {}
    for e in sorted_entries:
        prof = str(e.get("profit_profile") or "balanced")
        if prof not in by_profile:
            by_profile[prof] = {
                "cycles": 0,
                "total_cycle_pnl_usd": 0.0,
                "last_cumulative_realized_pnl_usd": 0.0,
                "llm_cost_usd": 0.0,
            }
        by_profile[prof]["cycles"] += 1
        by_profile[prof]["total_cycle_pnl_usd"] += float(
            e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0
        )
        by_profile[prof]["last_cumulative_realized_pnl_usd"] = float(
            e.get("pnl", {}).get("cumulative_realized_pnl_usd", 0) or 0
        )
        by_profile[prof]["llm_cost_usd"] = float(e.get("pnl", {}).get("llm_cost_usd", 0) or 0)
    outcomes: dict[str, int] = defaultdict(int)
    for e in sorted_entries:
        outcomes[str(e.get("outcome", "unknown"))] += 1

    first_ts = _parse_ts(str(sorted_entries[0].get("ts", "")))
    last_ts = _parse_ts(str(last.get("ts", "")))
    return {
        "cycles": len(sorted_entries),
        "total_cycle_pnl_usd": round(total_cycle_pnl, 4),
        "cumulative_realized_pnl_usd": round(cum_pnl, 4),
        "llm_cost_usd": round(llm, 4),
        "max_drawdown_pct": round(
            max(float(e.get("pnl", {}).get("intraday_drawdown_pct", 0) or 0) for e in sorted_entries),
            4,
        ),
        "win_rate_pct": round(win_rate, 2),
        "trade_like_cycles": trades,
        "outcomes": dict(outcomes),
        "by_profile": by_profile,
        "last_quality": dict(last.get("quality") or {}),
        "first_entry_ts": first_ts.isoformat() if first_ts else None,
        "last_entry_ts": last_ts.isoformat() if last_ts else None,
        "last_cycle_id": last.get("cycle_id"),
        "last_outcome": last.get("outcome"),
    }


def build_profit_summary(*, window_hours: int = 24) -> dict[str, Any]:
    """Last N hours of cycle logs + current active ProfitConfig snapshot."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=window_hours)
    entries = load_entries_since(since)
    try:
        cfg = get_profit_config()
        active = snapshot_profit_config(cfg)
    except Exception:
        active = {"profit_profile": "unknown", "error": "get_profit_config failed"}

    stats = aggregate_entries(entries)
    return {
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
        "window_start": since.isoformat(),
        "active_config": active,
        "stats_24h": stats,
        "entries_in_window": len(entries),
    }
