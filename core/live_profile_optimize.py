"""Nightly profile suggestion from real profitability cycle logs (no backtest)."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from core.profile_optimization import composite_score, profit_factor_from_trades
from core.profit_cycle_logger import load_entries_since
from core.profit_profiles import VALID_PROFILES, is_evolved_profile, profile_note
from core.simulation.sim_broker import SimFill

BUILTIN_PROFILES = tuple(sorted(VALID_PROFILES))
DEFAULT_LOOKBACK_DAYS = 7
MIN_CYCLES_PER_PROFILE = 3


def _parse_ts(entry: dict[str, Any]) -> datetime | None:
    raw = entry.get("ts")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def load_live_entries(*, days: int = DEFAULT_LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """Last ``days`` calendar days of cycle log entries."""
    since = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    return load_entries_since(since)


def _cycle_pnls(entries: list[dict[str, Any]]) -> list[float]:
    return [float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0) for e in entries]


def _daily_session_pnl(entries: list[dict[str, Any]]) -> list[float]:
    by_day: dict[str, float] = defaultdict(float)
    for e in entries:
        day = str(e.get("session_date") or "")[:10] or "unknown"
        by_day[day] += float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
    return [by_day[d] for d in sorted(by_day) if d != "unknown"]


def _sharpe_from_daily_pnl(
    daily_pnl: list[float],
    *,
    notional: float = 100_000.0,
) -> float:
    if len(daily_pnl) < 2 or notional <= 0:
        return 0.0
    returns = [p / notional for p in daily_pnl]
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    if std_r <= 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)


def _profit_factor_from_cycles(entries: list[dict[str, Any]]) -> float:
    pnls = _cycle_pnls(entries)
    fills = [SimFill("", "SELL", 1, 1.0, pnl=p) for p in pnls if p != 0]
    if not fills:
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        if gross_loss <= 0:
            return min(3.0, gross_profit / 100.0) if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    return profit_factor_from_trades(fills)


def _win_rate_pct(entries: list[dict[str, Any]]) -> float:
    wins = 0
    trades = 0
    for e in entries:
        pnl_delta = float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
        action = (e.get("trade_outcome") or {}).get("action", "")
        if action and action not in ("quality_status", "briefing", "market_hours", ""):
            if pnl_delta != 0 or (e.get("trade_outcome") or {}).get("order_id"):
                trades += 1
                if pnl_delta > 0:
                    wins += 1
    return (wins / trades * 100.0) if trades else 0.0


def score_profile_entries(
    entries: list[dict[str, Any]],
    *,
    notional: float = 100_000.0,
) -> dict[str, Any]:
    """Composite score for one profile's logged cycles."""
    daily = _daily_session_pnl(entries)
    sharpe = _sharpe_from_daily_pnl(daily, notional=notional)
    pf = _profit_factor_from_cycles(entries)
    wr = _win_rate_pct(entries)
    comp = composite_score(sharpe, pf, wr)
    return {
        "composite_score": round(comp, 4),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "win_rate_pct": round(wr, 2),
        "cycles": len(entries),
        "sessions": len(daily),
        "total_cycle_pnl_usd": round(sum(_cycle_pnls(entries)), 2),
        "llm_cost_usd": round(
            float(entries[-1].get("pnl", {}).get("llm_cost_usd", 0) or 0) if entries else 0.0,
            4,
        ),
    }


def _group_by_profile(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        prof = str(e.get("profit_profile") or "balanced").strip().lower()
        groups[prof].append(e)
    return dict(groups)


def run_live_optimize(
    *,
    days: int = DEFAULT_LOOKBACK_DAYS,
    notional: float = 100_000.0,
    min_cycles: int = MIN_CYCLES_PER_PROFILE,
) -> dict[str, Any]:
    """
  Analyze real cycle logs and rank profiles for tomorrow's ``PROFIT_PROFILE``.

  Lightweight: no simulation — uses logged P&L, outcomes, and profile tags only.
  """
    entries = load_live_entries(days=days)
    since = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    grouped = _group_by_profile(entries)

    profile_scores: list[dict[str, Any]] = []
    for prof, prof_entries in sorted(grouped.items()):
        metrics = score_profile_entries(prof_entries, notional=notional)
        eligible = metrics["cycles"] >= min_cycles
        profile_scores.append(
            {
                "profile": prof,
                "builtin": prof in VALID_PROFILES,
                "evolved": is_evolved_profile(prof),
                "eligible": eligible,
                "metrics": metrics,
            }
        )

    eligible_builtin = [
        r for r in profile_scores if r["eligible"] and r["profile"] in VALID_PROFILES
    ]
    eligible_builtin.sort(key=lambda r: r["metrics"]["composite_score"], reverse=True)

    if eligible_builtin:
        suggested = eligible_builtin[0]["profile"]
        confidence = "high"
        reason = (
            f"Highest composite score over {days}d of live cycles "
            f"({eligible_builtin[0]['metrics']['cycles']} cycles, "
            f"composite={eligible_builtin[0]['metrics']['composite_score']:.3f})."
        )
    elif profile_scores:
        # Fall back to best profile with any data, prefer built-in
        ranked_any = sorted(
            profile_scores,
            key=lambda r: (
                r["profile"] in VALID_PROFILES,
                r["metrics"]["composite_score"],
                r["metrics"]["cycles"],
            ),
            reverse=True,
        )
        suggested = ranked_any[0]["profile"]
        confidence = "low"
        reason = (
            f"Limited data (<{min_cycles} cycles per built-in profile); "
            f"best available tag {suggested!r} ({ranked_any[0]['metrics']['cycles']} cycles)."
        )
        if suggested not in VALID_PROFILES:
            suggested = "balanced"
            reason += " Defaulting suggestion to balanced (insufficient built-in samples)."
    else:
        suggested = "balanced"
        confidence = "none"
        reason = f"No profitability cycle logs in the last {days} days; default balanced."

    import os

    from core.profit_profiles import PROFIT_PROFILE_ENV

    current = os.getenv(PROFIT_PROFILE_ENV, "balanced").strip().lower() or "balanced"

    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suggest_for_session": tomorrow,
        "lookback_days": days,
        "window_start": since.isoformat(),
        "entries_analyzed": len(entries),
        "min_cycles_per_profile": min_cycles,
        "composite_formula": "0.4*sharpe_norm + 0.3*profit_factor_norm + 0.3*win_rate_norm",
        "data_source": "profit_cycle_logs (logs/profit_cycles_*.json + Postgres)",
        "current_profit_profile": current,
        "suggested_profile": suggested,
        "suggested_profile_note": profile_note(suggested) if suggested in VALID_PROFILES else "",
        "confidence": confidence,
        "rationale": reason,
        "action": {
            "set_env": f"{PROFIT_PROFILE_ENV}={suggested}",
            "cli": f"python __main__.py --profit-profile {suggested}",
        },
        "rankings": profile_scores,
        "builtin_rankings": eligible_builtin,
    }


def format_live_optimize_report(result: dict[str, Any]) -> str:
    """Human-readable nightly suggestion for the terminal."""
    lines = [
        "=" * 64,
        "  Live profile optimization (real cycle logs)",
        "=" * 64,
        f"  Window:           last {result['lookback_days']} days ({result['entries_analyzed']} cycles)",
        f"  Suggest for:      {result['suggest_for_session']}",
        f"  Current profile:  {result['current_profit_profile']}",
        f"  Suggested:        {result['suggested_profile']}  [{result['confidence']} confidence]",
        f"  Why:              {result['rationale']}",
        "",
        "  Tomorrow",
        f"    export {result['action']['set_env']}",
        f"    # or: {result['action']['cli']}",
        "",
        "  Rankings (logged profit_profile tags)",
    ]
    for row in sorted(
        result.get("rankings") or [],
        key=lambda r: r.get("metrics", {}).get("composite_score", 0),
        reverse=True,
    ):
        m = row.get("metrics") or {}
        flag = " *" if row.get("profile") == result["suggested_profile"] else ""
        elig = "ok" if row.get("eligible") else f"<{result['min_cycles_per_profile']} cyc"
        lines.append(
            f"    {row.get('profile', '?'):14}  composite={m.get('composite_score', 0):.3f}  "
            f"pnl=${m.get('total_cycle_pnl_usd', 0):+,.2f}  cycles={m.get('cycles', 0)}  "
            f"({elig}){flag}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "format_live_optimize_report",
    "load_live_entries",
    "run_live_optimize",
    "score_profile_entries",
]
