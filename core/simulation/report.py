"""Markdown report for backtest results."""

from __future__ import annotations

from collections.abc import Sequence

from core.simulation.types import BacktestResult

_COMPARISON_ROWS: tuple[tuple[str, str], ...] = (
    ("Total profit", "total_profit"),
    ("Final equity", "final_equity"),
    ("Win rate", "win_rate"),
    ("Max drawdown", "max_drawdown_pct"),
    ("Sharpe (daily, ann.)", "sharpe_ratio"),
    ("Avg R:R (approx.)", "avg_rr"),
    ("Trade count", "trade_count"),
    ("LLM cost (est.)", "llm_cost_usd"),
    ("Agent cycles", "cycles_run"),
)


def _fmt_metric(attr: str, value: float | int) -> str:
    if attr == "total_profit":
        return f"${value:,.2f}"
    if attr == "final_equity":
        return f"${value:,.2f}"
    if attr == "win_rate":
        return f"{value:.1f}%"
    if attr == "max_drawdown_pct":
        return f"{value:.2f}%"
    if attr in ("sharpe_ratio", "avg_rr"):
        return f"{value:.2f}"
    if attr == "llm_cost_usd":
        return f"${value:.4f}"
    return str(value)


def format_backtest_comparison(results: Sequence[BacktestResult]) -> str:
    """Side-by-side markdown table for multiple profiles on the same date range."""
    if not results:
        return "# Backtest profile comparison\n\nNo results.\n"
    if len(results) == 1:
        return format_backtest_report(results[0])

    first = results[0]
    profiles = [r.profile for r in results]
    header = "| Metric | " + " | ".join(f"`{p}`" for p in profiles) + " |"
    sep = "|--------|" + "|".join("----------:" for _ in profiles) + "|"

    lines = [
        "# Backtest profile comparison",
        "",
        f"**Period:** {first.start_date} → {first.end_date}  ",
        f"**Trading days:** {first.trading_days}  ",
        f"**Initial equity:** ${first.initial_equity:,.2f} (each run)",
        "",
        header,
        sep,
    ]

    for label, attr in _COMPARISON_ROWS:
        cells = [_fmt_metric(attr, getattr(r, attr)) for r in results]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")

    best = max(results, key=lambda r: r.total_profit)
    lines.extend(
        [
            "",
            f"**Highest total profit:** `{best.profile}` (${best.total_profit:,.2f})",
            "",
            "## Per-profile notes",
            "",
        ]
    )
    for r in results:
        lines.append(f"### `{r.profile}`")
        if r.notes:
            for note in r.notes[:3]:
                lines.append(f"- {note}")
        else:
            lines.append("- (no notes)")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_backtest_report(result: BacktestResult) -> str:
    """Return a clean markdown summary."""
    lines = [
        "# Backtest simulation report",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Profile | `{result.profile}` |",
        f"| Date range | {result.start_date} → {result.end_date} |",
        f"| Trading days | {result.trading_days} |",
        f"| Agent cycles | {result.cycles_run} |",
        f"| Initial equity | ${result.initial_equity:,.2f} |",
        f"| Final equity | ${result.final_equity:,.2f} |",
        f"| **Total profit** | **${result.total_profit:,.2f}** |",
        f"| Win rate | {result.win_rate:.1f}% |",
        f"| Max drawdown | {result.max_drawdown_pct:.2f}% |",
        f"| Sharpe (daily, ann.) | {result.sharpe_ratio:.2f} |",
        f"| Avg R:R (approx.) | {result.avg_rr:.2f} |",
        f"| Trade count | {result.trade_count} |",
        f"| LLM cost (est.) | ${result.llm_cost_usd:.4f} |",
        "",
        "## Notes",
        "",
    ]
    if result.notes:
        for n in result.notes:
            lines.append(f"- {n}")
    else:
        lines.append("- QualityMatrix and SafetyController remained active (simulated broker + replay data).")
    lines.append("")
    if result.equity_curve:
        lines.extend(["## Equity curve (EOD)", ""])
        for day, eq in result.equity_curve[-20:]:
            lines.append(f"- {day}: ${eq:,.2f}")
        if len(result.equity_curve) > 20:
            lines.insert(len(lines) - len(result.equity_curve[-20:]), f"- … ({len(result.equity_curve) - 20} earlier days omitted)")
    return "\n".join(lines)
