"""CSV export for simulation trade logs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.simulation.trade_log import TRADE_CSV_COLUMNS, SimTradeLog
from core.simulation.types import BacktestResult

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = _REPO_ROOT / "data" / "sim_exports"


def default_trade_csv_path(
    start_date: str,
    end_date: str,
    profiles: Sequence[str],
) -> Path:
    """Default path: ``data/sim_exports/backtest_<profiles>_<start>_<end>.csv``."""
    prof = "-".join(profiles) if len(profiles) <= 3 else f"{len(profiles)}profiles"
    name = f"backtest_{prof}_{start_date}_{end_date}.csv"
    return DEFAULT_EXPORT_DIR / name


def collect_trade_logs(results: Sequence[BacktestResult]) -> list[SimTradeLog]:
    logs: list[SimTradeLog] = []
    for r in results:
        logs.extend(r.trade_log)
    logs.sort(key=lambda t: (t.entry_time_utc, t.symbol))
    return logs


def write_trade_log_csv(
    path: Path | str,
    results: Sequence[BacktestResult],
    *,
    include_header: bool = True,
) -> Path:
    """Write all trades from one or more backtest runs to a CSV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    trades = collect_trade_logs(results)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRADE_CSV_COLUMNS), extrasaction="ignore")
        if include_header:
            writer.writeheader()
        for trade in trades:
            writer.writerow(trade.to_csv_row())
    return out


def export_simulation_csv(
    results: Sequence[BacktestResult],
    *,
    path: Path | str | None = None,
) -> Path:
    """Write trade log CSV; pick default path from result metadata when ``path`` is omitted."""
    if not results:
        raise ValueError("no backtest results to export")
    first = results[0]
    profiles = [r.profile for r in results]
    target = Path(path) if path else default_trade_csv_path(first.start_date, first.end_date, profiles)
    return write_trade_log_csv(target, results)
