"""Per-trade round-trip records for simulation CSV export."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SimTradeLog:
    """One closed round-trip (entry → exit) from the simulated broker."""

    profit_profile: str
    symbol: str
    qty: int
    entry_time_utc: str
    exit_time_utc: str
    entry_price: float
    exit_price: float
    realized_pnl: float
    realized_rr: float
    session_date: str
    exit_reason: str = "sell"

    def to_csv_row(self) -> dict[str, str | int | float]:
        row = asdict(self)
        row["ticker"] = self.symbol
        return row


TRADE_CSV_COLUMNS: tuple[str, ...] = (
    "profit_profile",
    "session_date",
    "ticker",
    "symbol",
    "qty",
    "entry_time_utc",
    "exit_time_utc",
    "entry_price",
    "exit_price",
    "realized_pnl",
    "realized_rr",
    "exit_reason",
)


def realized_r_multiple(
    *,
    entry_price: float,
    exit_price: float,
    qty: int,
    realized_pnl: float,
    assumed_risk_pct: float = 2.0,
) -> float:
    """R-multiple = P&L / (entry notional × risk %%). Uses 2%% when no stop is modeled."""
    if qty <= 0 or entry_price <= 0:
        return 0.0
    risk_dollars = entry_price * qty * (assumed_risk_pct / 100.0)
    if risk_dollars <= 0:
        return 0.0
    return realized_pnl / risk_dollars
