#!/usr/bin/env python3
"""Terminal summary and HTML export of profitability cycle logs.

Reads ``logs/profit_cycles_YYYY-MM-DD.json`` (and Postgres when configured) written by
:class:`~core.central_profit_config.ProfitConfig` via :func:`core.profit_cycle_logger.log_cycle`.
Each entry snapshots the active **ProfitConfig profile** and lever subset for attribution.

Examples::

    python scripts/dashboard.py
    python scripts/dashboard.py --days 7
    python scripts/dashboard.py --days 7 --html logs/profit_report.html --open

See ``docs/simulation-and-optimization.md`` (live dashboard section).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_entries(session_date: str | None) -> list[dict]:
    from core.profit_cycle_logger import find_latest_log_file, load_daily_entries

    d = session_date or date.today().isoformat()
    entries = load_daily_entries(d)
    if entries:
        return entries
    latest = find_latest_log_file()
    if latest and latest.is_file():
        import json

        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            return list(data.get("entries") or [])
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _load_entries_window(*, session_date: str | None, days: int) -> list[dict]:
    """Load entries for one day or a rolling multi-day window."""
    if days <= 1 and session_date:
        return _load_entries(session_date)
    if days <= 1 and not session_date:
        return _load_entries(None)
    from core.profit_cycle_logger import load_entries_since

    since = datetime.now(timezone.utc) - timedelta(days=days)
    return load_entries_since(since)


def _aggregate(entries: list[dict]) -> dict:
    from core.profit_summary import aggregate_entries

    raw = aggregate_entries(entries)
    if not raw.get("cycles"):
        return {}
    by_profile = raw.get("by_profile") or {}
    top_profiles = sorted(
        by_profile.items(),
        key=lambda kv: kv[1].get("last_cumulative_realized_pnl_usd", 0),
        reverse=True,
    )[:3]
    top_mapped = [
        (
            prof,
            {
                "cycles": stats["cycles"],
                "cycle_pnl": stats["total_cycle_pnl_usd"],
                "last_cum_pnl": stats["last_cumulative_realized_pnl_usd"],
                "llm": stats["llm_cost_usd"],
            },
        )
        for prof, stats in top_profiles
    ]
    session_date = (
        str(entries[0].get("session_date", date.today().isoformat())) if entries else date.today().isoformat()
    )
    return {
        "session_date": session_date,
        "cycles": raw["cycles"],
        "total_cycle_pnl": raw["total_cycle_pnl_usd"],
        "cumulative_pnl": raw["cumulative_realized_pnl_usd"],
        "llm_cost": raw["llm_cost_usd"],
        "max_drawdown_pct": raw["max_drawdown_pct"],
        "win_rate": raw["win_rate_pct"],
        "trade_like_cycles": raw["trade_like_cycles"],
        "top_profiles": top_mapped,
        "outcomes": raw.get("outcomes") or {},
        "quality": raw.get("last_quality") or {},
    }


def format_dashboard(agg: dict, *, window_label: str | None = None) -> str:
    if not agg:
        return (
            "No profitability cycle logs found.\n"
            "  Run the trader or check logs/profit_cycles_YYYY-MM-DD.json\n"
        )
    title = window_label or f"session {agg['session_date']}"
    lines = [
        "=" * 60,
        "  Profitability dashboard",
        "=" * 60,
        f"  Window:           {title}",
        f"  Cycles logged:    {agg['cycles']}",
        "",
        "  P&L (window)",
        f"    Cycle sum:      ${agg['total_cycle_pnl']:+,.2f}",
        f"    Cumulative:     ${agg['cumulative_pnl']:+,.2f}",
        f"    LLM cost:       ${agg['llm_cost']:.4f}",
        f"    Max drawdown:   {agg['max_drawdown_pct']:.2f}%",
        f"    Win rate:       {agg['win_rate']:.1f}%  ({agg['trade_like_cycles']} action cycles)",
        "",
        f"  Quality (last cycle): {agg['quality'].get('overall_quality', '?')}"
        f"  rm={float(agg['quality'].get('risk_multiplier', 0)):.2f}"
        f"  exec={float(agg['quality'].get('execution_quality', 0)):.2f}",
        "",
        "  Top ProfitConfig profiles (by cumulative realized P&L)",
    ]
    if agg["top_profiles"]:
        for i, (prof, stats) in enumerate(agg["top_profiles"], 1):
            lines.append(
                f"    {i}. {prof:12}  cum=${stats['last_cum_pnl']:+,.2f}  "
                f"cycle_sum=${stats['cycle_pnl']:+,.2f}  cycles={stats['cycles']}  "
                f"llm=${stats['llm']:.4f}"
            )
    else:
        lines.append("    (none)")
    lines.extend(["", "  Cycle outcomes"])
    for outcome, count in sorted(agg["outcomes"].items(), key=lambda x: -x[1]):
        lines.append(f"    {outcome:28} {count}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _window_label(session_date: str | None, days: int) -> str:
    if days <= 1 and session_date:
        return f"session {session_date}"
    if days <= 1:
        return f"session {date.today().isoformat()}"
    return f"last {days} days"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ProfitConfig cycle logs (terminal or HTML)",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Session anchor when --days 1 (default: today)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        metavar="N",
        help="Rolling window in days (default: 1 = single day)",
    )
    parser.add_argument(
        "--html",
        metavar="PATH",
        default=None,
        help="Write browser report to PATH (e.g. logs/profit_report.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open --html file in the default browser after writing",
    )
    args = parser.parse_args(argv)

    days = max(1, int(args.days))
    entries = _load_entries_window(session_date=args.date, days=days)
    label = _window_label(args.date, days)

    if args.html:
        from scripts.dashboard_html import write_html_report

        out = Path(args.html)
        if not out.is_absolute():
            out = _REPO / out
        write_html_report(out, entries, window_label=label)
        print(f"Wrote HTML report: {out}")
        if args.open:
            import webbrowser

            webbrowser.open(out.resolve().as_uri())
        return 0

    print(format_dashboard(_aggregate(entries), window_label=label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
