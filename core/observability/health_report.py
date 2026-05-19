"""Build a JSON health report from ProfitConfig, cycle logs, safety, and research heartbeat.

All thresholds and caps come from :func:`core.central_profit_config.get_profit_config`
so trader, research host, CLI, and APIs stay aligned.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from core.profit_config_state import get_active_profile_label, summarize_profit_config
from core.profit_cycle_logger import build_daily_summary, snapshot_profit_config
from core.profit_summary import build_profit_summary

OverallStatus = Literal["healthy", "degraded", "unhealthy"]
AlertSeverity = Literal["info", "warn", "critical"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_EXPORT_DIR = _REPO_ROOT / "data" / "sim_exports"
_LIVE_SUGGESTION = _REPO_ROOT / "data" / "live_profile_suggestion.json"

_WARN_RATIO_DEFAULT = 0.85


@dataclass
class Alert:
    code: str
    severity: AlertSeverity
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def overall_status_from_alerts(alerts: list[Alert]) -> OverallStatus:
    if any(a.severity == "critical" for a in alerts):
        return "unhealthy"
    if any(a.severity == "warn" for a in alerts):
        return "degraded"
    return "healthy"


def _warn_ratio() -> float:
    raw = os.getenv("ABC_ALERT_WARN_RATIO", "").strip()
    if raw:
        try:
            return max(0.5, min(0.99, float(raw)))
        except ValueError:
            pass
    return _WARN_RATIO_DEFAULT


def _load_profit_config() -> tuple[Any, dict[str, Any]]:
    from core.central_profit_config import get_profit_config

    cfg = get_profit_config().reload(dotenv=False)
    snap = snapshot_profit_config(cfg, profile_label=get_active_profile_label())
    return cfg, snap


def _collect_research_heartbeat() -> dict[str, Any]:
    out: dict[str, Any] = {
        "last_ts": None,
        "age_s": None,
        "alive": False,
        "operational": False,
        "status_code": 0.0,
        "round": 0,
        "usage_pct": 0.0,
        "host_profile": None,
    }
    try:
        from core.research_settings import RESEARCH_HOST_PROFILE_KEY
        from core.runtime.heartbeat import (
            ResearchHostStatus,
            heartbeat_age_s,
            is_research_host_alive,
            is_research_host_operational,
            read_heartbeat,
            read_research_host_status,
        )
        from memory import get_research_config

        last = read_heartbeat()
        age = heartbeat_age_s()
        out["last_ts"] = last if last > 0 else None
        out["age_s"] = None if age == float("inf") else round(age, 2)
        out["alive"] = bool(is_research_host_alive())
        out["operational"] = bool(is_research_host_operational())
        st = read_research_host_status()
        out["status_code"] = st
        out["status_label"] = {
            ResearchHostStatus.RUNNING: "running",
            ResearchHostStatus.SCORING: "scoring",
            ResearchHostStatus.STARTING: "starting",
            ResearchHostStatus.SHUTTING_DOWN: "shutting_down",
            ResearchHostStatus.CAP_STOPPED: "cap_stopped",
            ResearchHostStatus.STOPPED: "stopped",
        }.get(st, "unknown")
        out["round"] = int(float(get_research_config("research_host_round", 0.0)))
        out["usage_pct"] = float(get_research_config("research_host_usage_pct", 0.0))
        out["host_profile"] = str(get_research_config(RESEARCH_HOST_PROFILE_KEY, "") or "")
    except Exception as e:
        out["error"] = str(e)
    return out


def _collect_safety_metrics(cfg: Any) -> dict[str, Any]:
    """Metrics vs ProfitConfig risk caps (no live broker required)."""
    risk = cfg.risk
    metrics: dict[str, Any] = {
        "thresholds": {
            "max_daily_loss_pct": float(risk.max_daily_loss_pct),
            "intraday_drawdown_pct": float(risk.intraday_drawdown_pct),
            "max_daily_llm_cost_usd": float(risk.max_daily_llm_cost),
        },
        "today_llm_cost_usd": 0.0,
        "llm_cost_ratio": 0.0,
        "today_realized_pnl_usd": 0.0,
        "intraday_drawdown_pct": 0.0,
        "daily_loss_pct": None,
        "net_liquidation": None,
        "session_high_water": None,
        "breached": {
            "daily_loss": False,
            "drawdown": False,
            "llm_cost": False,
        },
        "near_limit": {
            "daily_loss": False,
            "drawdown": False,
            "llm_cost": False,
        },
    }
    warn_ratio = _warn_ratio()

    try:
        from data.cost_tracker import get_cost_tracker

        bs = get_cost_tracker().get_budget_summary()
        metrics["today_llm_cost_usd"] = round(float(bs.today_llm_cost), 6)
        metrics["today_realized_pnl_usd"] = round(float(bs.today_realized_pnl), 4)
        cap = float(risk.max_daily_llm_cost)
        if cap > 0:
            ratio = metrics["today_llm_cost_usd"] / cap
            metrics["llm_cost_ratio"] = round(ratio, 4)
            metrics["breached"]["llm_cost"] = ratio >= 1.0
            metrics["near_limit"]["llm_cost"] = ratio >= warn_ratio and ratio < 1.0
    except Exception as e:
        metrics["cost_tracker_error"] = str(e)

    try:
        from core.profit_cycle_logger import load_daily_entries

        entries = load_daily_entries()
        if entries:
            last = sorted(entries, key=lambda e: str(e.get("ts") or ""))[-1]
            pnl = last.get("pnl") or {}
            dd = float(pnl.get("intraday_drawdown_pct") or 0)
            metrics["intraday_drawdown_pct"] = round(dd, 4)
            dl = pnl.get("daily_loss_pct")
            if dl is not None:
                metrics["daily_loss_pct"] = round(float(dl), 4)
            if pnl.get("net_liquidation") is not None:
                metrics["net_liquidation"] = float(pnl["net_liquidation"])
            if pnl.get("session_high_water") is not None:
                metrics["session_high_water"] = float(pnl["session_high_water"])

            dd_lim = float(risk.intraday_drawdown_pct)
            if dd_lim > 0:
                metrics["breached"]["drawdown"] = dd >= dd_lim
                metrics["near_limit"]["drawdown"] = dd >= dd_lim * warn_ratio and dd < dd_lim

            loss_lim = float(risk.max_daily_loss_pct)
            if metrics["daily_loss_pct"] is not None and loss_lim > 0:
                loss = float(metrics["daily_loss_pct"])
                metrics["breached"]["daily_loss"] = loss >= loss_lim
                metrics["near_limit"]["daily_loss"] = (
                    loss >= loss_lim * warn_ratio and loss < loss_lim
                )
    except Exception as e:
        metrics["cycle_log_error"] = str(e)

    return metrics


def _collect_simulation_stats() -> dict[str, Any]:
    out: dict[str, Any] = {
        "simulation_mode": os.getenv("ABC_SIMULATION", "").strip() in ("1", "true", "yes"),
        "latest_export_csv": None,
        "live_profile_suggestion": None,
    }
    if _SIM_EXPORT_DIR.is_dir():
        csvs = sorted(_SIM_EXPORT_DIR.glob("backtest_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csvs:
            p = csvs[0]
            out["latest_export_csv"] = {
                "path": str(p.relative_to(_REPO_ROOT)).replace("\\", "/"),
                "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                "size_bytes": p.stat().st_size,
            }
    if _LIVE_SUGGESTION.is_file():
        try:
            data = json.loads(_LIVE_SUGGESTION.read_text(encoding="utf-8"))
            out["live_profile_suggestion"] = {
                "path": "data/live_profile_suggestion.json",
                "suggested_profile": data.get("suggested_profile"),
                "confidence": data.get("confidence"),
                "days_analyzed": data.get("days_analyzed"),
            }
        except (json.JSONDecodeError, OSError) as e:
            out["live_profile_suggestion"] = {"error": str(e)}
    return out


def _collect_operating_context() -> dict[str, Any]:
    try:
        from core.runtime.operating_context import get_operating_context

        ctx = get_operating_context()
        return {
            "researcher_available": bool(ctx.quality.researcher_available),
            "memory_source": str(ctx.quality.memory_source),
            "overall_quality": str(ctx.quality.overall_quality),
            "risk_multiplier": round(float(ctx.risk_multiplier), 4),
            "independent_mode": bool(ctx.is_independent_mode),
        }
    except Exception as e:
        return {"error": str(e)}


def _build_alerts(
    *,
    role: str,
    heartbeat: dict[str, Any],
    safety: dict[str, Any],
    daily: dict[str, Any],
    window: dict[str, Any],
) -> list[Alert]:
    alerts: list[Alert] = []
    warn_ratio = _warn_ratio()

    if not heartbeat.get("last_ts"):
        alerts.append(
            Alert(
                "research_heartbeat_missing",
                "critical" if role in ("trader", "all") else "warn",
                "Research host heartbeat missing",
                "Start research host: python -m research",
            )
        )
    elif not heartbeat.get("operational"):
        sev: AlertSeverity = "critical" if role in ("trader", "all") else "warn"
        age = heartbeat.get("age_s")
        alerts.append(
            Alert(
                "research_heartbeat_stale",
                sev,
                "Research host not operational",
                f"age_s={age} status={heartbeat.get('status_label')}",
            )
        )
    elif heartbeat.get("age_s") is not None and float(heartbeat["age_s"]) > 600:
        alerts.append(
            Alert(
                "research_heartbeat_stale",
                "warn",
                "Research heartbeat aging",
                f"age_s={heartbeat['age_s']}",
            )
        )

    if safety.get("breached", {}).get("llm_cost"):
        alerts.append(
            Alert(
                "llm_cost_breached",
                "critical",
                "Daily LLM cost at or above ProfitConfig ceiling",
                f"spent=${safety.get('today_llm_cost_usd')} cap=${safety['thresholds']['max_daily_llm_cost_usd']}",
            )
        )
    elif safety.get("near_limit", {}).get("llm_cost"):
        alerts.append(
            Alert(
                "llm_cost_near_ceiling",
                "warn",
                "Daily LLM cost approaching ceiling",
                f"ratio={safety.get('llm_cost_ratio')} warn_at={warn_ratio}",
            )
        )

    if safety.get("breached", {}).get("drawdown"):
        alerts.append(
            Alert(
                "drawdown_breached",
                "critical",
                "Intraday drawdown at or above limit",
                f"drawdown_pct={safety.get('intraday_drawdown_pct')}",
            )
        )
    elif safety.get("near_limit", {}).get("drawdown"):
        alerts.append(
            Alert(
                "drawdown_approaching_limit",
                "warn",
                "Intraday drawdown approaching limit",
                f"drawdown_pct={safety.get('intraday_drawdown_pct')}",
            )
        )

    if safety.get("breached", {}).get("daily_loss"):
        alerts.append(
            Alert(
                "daily_loss_breached",
                "critical",
                "Daily loss at or above limit",
                f"daily_loss_pct={safety.get('daily_loss_pct')}",
            )
        )
    elif safety.get("near_limit", {}).get("daily_loss"):
        alerts.append(
            Alert(
                "daily_loss_approaching_limit",
                "warn",
                "Daily loss approaching limit",
                f"daily_loss_pct={safety.get('daily_loss_pct')}",
            )
        )

    try:
        from core.config import RESEARCHER_DAILY_TOKEN_CAP
        from memory import get_research_config

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        usage = float(get_research_config(f"researcher_daily_usage_{today}", 0.0))
        cap = int(RESEARCHER_DAILY_TOKEN_CAP)
        if cap > 0:
            pct = usage / cap
            if usage >= cap:
                alerts.append(
                    Alert(
                        "researcher_token_cap_exceeded",
                        "critical",
                        "Researcher daily token cap exceeded",
                        f"{usage:,.0f}/{cap:,}",
                    )
                )
            elif pct >= warn_ratio:
                alerts.append(
                    Alert(
                        "researcher_token_cap_near",
                        "warn",
                        "Researcher token usage approaching cap",
                        f"{pct * 100:.1f}% of {cap:,}",
                    )
                )
    except Exception:
        pass

    if int(daily.get("cycles") or 0) == 0 and role in ("trader", "all"):
        alerts.append(
            Alert(
                "profit_cycles_missing_today",
                "info",
                "No profit cycle logs for today (UTC)",
                "Trader may be idle or logging disabled",
            )
        )

    stats = window.get("stats_24h") or {}
    if int(stats.get("cycles") or 0) == 0 and role in ("trader", "all"):
        alerts.append(
            Alert(
                "profit_cycles_missing_window",
                "info",
                "No profit cycle logs in summary window",
                f"window_hours={window.get('window_hours')}",
            )
        )

    if os.getenv("ABC_SIMULATION", "").strip() in ("1", "true", "yes"):
        alerts.append(
            Alert(
                "simulation_mode_active",
                "info",
                "ABC_SIMULATION is enabled",
                "Live trading and broker paths may be stubbed",
            )
        )

    return alerts


def build_health_report(
    *,
    role: str = "trader",
    window_hours: int | None = None,
    session_date: str | None = None,
) -> dict[str, Any]:
    """Full JSON health report for CLI, Docker, and ``GET /status``."""
    now = datetime.now(timezone.utc)
    hours = window_hours if window_hours is not None else int(
        os.getenv("PROFIT_SUMMARY_WINDOW_HOURS", "24")
    )
    day = session_date or now.strftime("%Y-%m-%d")

    cfg, config_snap = _load_profit_config()
    active_profile = str(config_snap.get("profit_profile") or get_active_profile_label())

    heartbeat = _collect_research_heartbeat()
    safety = _collect_safety_metrics(cfg)
    daily_summary = build_daily_summary(day)
    window_summary = build_profit_summary(window_hours=hours)
    simulation = _collect_simulation_stats()
    operating = _collect_operating_context()

    alerts = _build_alerts(
        role=role,
        heartbeat=heartbeat,
        safety=safety,
        daily=daily_summary,
        window=window_summary,
    )
    overall = overall_status_from_alerts(alerts)

    key_levers = {
        "trading_mode": config_snap.get("trading_mode"),
        "risk_per_trade_pct": (config_snap.get("risk") or {}).get("risk_per_trade_pct"),
        "max_daily_loss_pct": (config_snap.get("risk") or {}).get("max_daily_loss_pct"),
        "intraday_drawdown_pct": (config_snap.get("risk") or {}).get("intraday_drawdown_pct"),
        "max_daily_llm_cost": (config_snap.get("risk") or {}).get("max_daily_llm_cost"),
        "cooldown_seconds": (config_snap.get("loop") or {}).get("cooldown_seconds"),
        "quality_matrix_enabled": (config_snap.get("memory") or {}).get("quality_matrix_enabled"),
    }

    return {
        "generated_at": now.isoformat(),
        "role": role,
        "overall_status": overall,
        "active_profile": active_profile,
        "profit_config": {
            "snapshot": config_snap,
            "summary": summarize_profit_config(cfg),
            "key_levers": key_levers,
        },
        "research_heartbeat": heartbeat,
        "safety": safety,
        "daily_summary": daily_summary,
        "window_summary": {
            "window_hours": window_summary.get("window_hours"),
            "window_start": window_summary.get("window_start"),
            "entries_in_window": window_summary.get("entries_in_window"),
            "stats": window_summary.get("stats_24h"),
        },
        "simulation": simulation,
        "operating_context": operating,
        "alerts": [a.to_dict() for a in alerts],
        "alert_counts": {
            "critical": sum(1 for a in alerts if a.severity == "critical"),
            "warn": sum(1 for a in alerts if a.severity == "warn"),
            "info": sum(1 for a in alerts if a.severity == "info"),
        },
    }


