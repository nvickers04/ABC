"""Aggregate data for the web operations dashboard (cycle logs, health, sim vs live)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.profit_cycle_logger import load_entries_since
from core.profit_summary import aggregate_entries

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPT_PATHS = (
    _REPO_ROOT / "data" / "profile_optimization.json",
    _REPO_ROOT / "data" / "genetic_optimization.json",
)


def _load_optimizer_results() -> dict[str, Any] | None:
    for path in _OPT_PATHS:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["_source_path"] = str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _pnl_time_series(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Cumulative and per-cycle P&L for charting."""
    sorted_entries = sorted(entries, key=lambda e: str(e.get("ts") or ""))
    labels: list[str] = []
    cycle_pnl: list[float] = []
    cumulative: list[float] = []
    cum = 0.0
    for e in sorted_entries:
        ts = str(e.get("ts") or "")[:19]
        delta = float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
        cum = float(e.get("pnl", {}).get("cumulative_realized_pnl_usd", cum + delta) or cum + delta)
        labels.append(ts or "?")
        cycle_pnl.append(round(delta, 2))
        cumulative.append(round(cum, 2))
    return {
        "labels": labels,
        "cycle_pnl_usd": cycle_pnl,
        "cumulative_pnl_usd": cumulative,
    }


def _sim_vs_live_comparison(
    live_stats: dict[str, Any],
    optimizer: dict[str, Any] | None,
    active_profile: str,
) -> dict[str, Any]:
    """Bar chart data: live logged P&L by profile vs last simulation optimizer run."""
    live_by_profile = live_stats.get("by_profile") or {}
    live_labels = sorted(live_by_profile.keys())
    live_pnl = [round(float(live_by_profile[p].get("total_cycle_pnl_usd", 0) or 0), 2) for p in live_labels]

    sim_labels: list[str] = []
    sim_composite: list[float] = []
    sim_pnl: list[float] = []
    sim_meta: dict[str, Any] = {}
    if optimizer:
        sim_meta = {
            "source": optimizer.get("_source_path"),
            "start_date": optimizer.get("start_date"),
            "end_date": optimizer.get("end_date"),
            "lookback_days": optimizer.get("lookback_days"),
            "mode": optimizer.get("mode"),
        }
        for row in optimizer.get("rankings") or optimizer.get("results") or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("candidate_id") or row.get("profile") or "")
            metrics = row.get("metrics") or row
            if not cid:
                continue
            sim_labels.append(cid)
            sim_composite.append(round(float(metrics.get("composite_score", 0) or 0), 4))
            sim_pnl.append(round(float(metrics.get("total_profit_usd", 0) or 0), 2))
        best = optimizer.get("best") or {}
        if best and not sim_labels:
            cid = str(best.get("candidate_id") or best.get("base_profile") or "best")
            m = best.get("metrics") or {}
            sim_labels = [cid]
            sim_composite = [round(float(m.get("composite_score", 0) or 0), 4)]
            sim_pnl = [round(float(m.get("total_profit_usd", 0) or 0), 2)]

    active_live_pnl = float((live_by_profile.get(active_profile) or {}).get("total_cycle_pnl_usd", 0) or 0)
    active_sim = None
    if optimizer and sim_labels:
        for i, lab in enumerate(sim_labels):
            if lab == active_profile or lab.startswith(active_profile):
                active_sim = {
                    "composite_score": sim_composite[i] if i < len(sim_composite) else 0,
                    "total_profit_usd": sim_pnl[i] if i < len(sim_pnl) else 0,
                }
                break
        if active_sim is None and optimizer.get("best"):
            m = (optimizer["best"].get("metrics") or {})
            active_sim = {
                "composite_score": m.get("composite_score"),
                "total_profit_usd": m.get("total_profit_usd"),
                "note": "optimizer best candidate (may differ from active profile)",
            }

    return {
        "active_profile": active_profile,
        "live": {"labels": live_labels, "total_cycle_pnl_usd": live_pnl},
        "simulation": {
            "labels": sim_labels,
            "composite_score": sim_composite,
            "total_profit_usd": sim_pnl,
            "meta": sim_meta,
        },
        "active_comparison": {
            "live_cycle_pnl_usd": round(active_live_pnl, 2),
            "sim": active_sim,
        },
    }


def _quality_matrix_stats() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from core.quality.quality_matrix import get_quality_matrix_service

        m = get_quality_matrix_service().get_matrix()
        out = {
            "overall_quality": str(m.overall_quality),
            "risk_multiplier": round(float(m.risk_multiplier), 4),
            "execution_quality": round(float(getattr(m, "execution_quality", 0) or 0), 4),
            "global_execution_quality": round(float(m.global_execution_quality), 4),
            "symbol_count": len(m.symbol_quality),
            "blocked_tool_categories": list(m.blocked_tool_categories),
            "force_conservative_reasoning": bool(m.force_conservative_reasoning),
            "last_populated": m.last_populated.isoformat() if m.last_populated else None,
        }
    except Exception as e:
        out["error"] = str(e)

    learned_path = _REPO_ROOT / "data" / "quality_matrix_learned.json"
    if learned_path.is_file():
        try:
            learned = json.loads(learned_path.read_text(encoding="utf-8"))
            out["learning"] = {
                "profile_label": learned.get("profile_label"),
                "last_reward": learned.get("last_reward"),
                "trades_since_refit": learned.get("trades_since_refit"),
                "total_trades": learned.get("total_trades"),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return out


def build_dashboard_payload(
    *,
    role: str = "trader",
    window_hours: int | None = None,
    chart_days: int = 7,
) -> dict[str, Any]:
    """JSON payload for ``GET /dashboard/data`` and HTML render."""
    from core.observability.health_report import build_health_report

    hours = window_hours if window_hours is not None else int(
        os.getenv("DASHBOARD_WINDOW_HOURS", os.getenv("PROFIT_SUMMARY_WINDOW_HOURS", "24"))
    )
    days = max(1, int(os.getenv("DASHBOARD_CHART_DAYS", str(chart_days))))

    report = build_health_report(role=role, window_hours=hours)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    chart_entries = load_entries_since(since)
    chart_stats = aggregate_entries(chart_entries)

    optimizer = _load_optimizer_results()
    active = str(report.get("active_profile") or "balanced")
    window_stats = (report.get("window_summary") or {}).get("stats") or {}

    return {
        "generated_at": report.get("generated_at"),
        "overall_status": report.get("overall_status"),
        "active_profile": active,
        "profit_config": report.get("profit_config"),
        "key_levers": (report.get("profit_config") or {}).get("key_levers"),
        "research_heartbeat": report.get("research_heartbeat"),
        "safety": report.get("safety"),
        "daily_summary": report.get("daily_summary"),
        "window_hours": hours,
        "window_stats": window_stats,
        "chart_days": days,
        "chart_entries": len(chart_entries),
        "pnl_series": _pnl_time_series(chart_entries),
        "chart_stats": chart_stats,
        "quality_matrix": _quality_matrix_stats(),
        "operating_context": report.get("operating_context"),
        "alerts": report.get("alerts"),
        "sim_vs_live": _sim_vs_live_comparison(chart_stats, optimizer, active),
        "optimizer": {
            "best": (optimizer or {}).get("best"),
            "recommended": (optimizer or {}).get("recommended_config_changes"),
            "source": (optimizer or {}).get("_source_path"),
        }
        if optimizer
        else None,
    }


__all__ = ["build_dashboard_payload"]
