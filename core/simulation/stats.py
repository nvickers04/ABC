"""Aggregate backtest performance statistics."""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from core.simulation.sim_broker import SimFill
from core.simulation.trade_log import SimTradeLog
from core.simulation.llm_cost_estimate import BacktestRunStats
from core.simulation.types import BacktestResult


def _max_drawdown_pct(equity: Sequence[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def _sharpe(daily_returns: Sequence[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean_r = statistics.mean(daily_returns)
    std_r = statistics.stdev(daily_returns)
    if std_r <= 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)


def _profit_factor(trades: Sequence[SimFill]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss <= 0:
        return min(3.0, gross_profit / 100.0) if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _win_rate(trades: Sequence[SimFill]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades) * 100.0


def _avg_rr_from_logs(trade_log: Sequence[SimTradeLog]) -> float:
    if not trade_log:
        return 0.0
    return statistics.mean(t.realized_rr for t in trade_log)


def _avg_rr(trades: Sequence[SimFill]) -> float:
    """Approximate R:R from realized P&L vs notional at fill."""
    ratios: list[float] = []
    for t in trades:
        notional = abs(t.price * t.qty)
        if notional <= 0:
            continue
        r = abs(t.pnl) / notional
        ratios.append(r)
    return statistics.mean(ratios) if ratios else 0.0


def build_backtest_result(
    *,
    profile: str,
    start_date: str,
    end_date: str,
    trading_days: int,
    cycles_run: int,
    initial_equity: float,
    cycles_per_day: int = 1,
    equity_curve: list[tuple[str, float]],
    closed_trades: Sequence[SimFill],
    llm_cost_usd: float,
    trade_log: Sequence[SimTradeLog] | None = None,
    notes: list[str] | None = None,
    run_stats: BacktestRunStats | None = None,
) -> BacktestResult:
    equities = [e for _, e in equity_curve] or [initial_equity]
    final = equities[-1]
    daily_returns: list[float] = []
    for i in range(1, len(equities)):
        prev, cur = equities[i - 1], equities[i]
        if prev > 0:
            daily_returns.append((cur - prev) / prev)

    logs = list(trade_log or [])
    avg_rr = _avg_rr_from_logs(logs) if logs else _avg_rr(closed_trades)

    return BacktestResult(
        profile=profile,
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        cycles_run=cycles_run,
        cycles_per_day=cycles_per_day,
        total_profit=final - initial_equity,
        win_rate=_win_rate(closed_trades),
        profit_factor=_profit_factor(closed_trades),
        max_drawdown_pct=_max_drawdown_pct(equities),
        sharpe_ratio=_sharpe(daily_returns),
        avg_rr=avg_rr,
        trade_count=len(logs) if logs else len(closed_trades),
        llm_cost_usd=llm_cost_usd,
        initial_equity=initial_equity,
        final_equity=final,
        equity_curve=equity_curve,
        trade_log=logs,
        notes=notes or [],
        run_stats=run_stats,
    )
