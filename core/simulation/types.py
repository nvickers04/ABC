"""Backtest result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.simulation.llm_cost_estimate import BacktestRunStats
    from core.simulation.trade_log import SimTradeLog


@dataclass
class BacktestResult:
    """Aggregated simulation output."""

    profile: str
    start_date: str
    end_date: str
    trading_days: int
    cycles_run: int
    cycles_per_day: int
    total_profit: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_rr: float
    trade_count: int
    llm_cost_usd: float
    initial_equity: float
    final_equity: float
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    trade_log: list["SimTradeLog"] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    run_stats: "BacktestRunStats | None" = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_days": self.trading_days,
            "cycles_run": self.cycles_run,
            "cycles_per_day": self.cycles_per_day,
            "total_profit": self.total_profit,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_rr": self.avg_rr,
            "trade_count": self.trade_count,
            "llm_cost_usd": self.llm_cost_usd,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "equity_curve": self.equity_curve,
            "trade_log": self.trade_log,
            "notes": self.notes,
            "run_stats": self.run_stats.as_dict() if self.run_stats else None,
        }

