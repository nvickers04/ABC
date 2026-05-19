"""Simulated broker for backtests (fills at replay mid, tracks P&L)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from core.simulation.replay_data import ReplaySessionDataProvider
from core.simulation.trade_log import SimTradeLog, realized_r_multiple

logger = logging.getLogger(__name__)


@dataclass
class SimFill:
    """Legacy per-exit fill (kept for stats compatibility)."""

    symbol: str
    side: str
    qty: int
    price: float
    pnl: float = 0.0
    session_date: str = ""


@dataclass
class SimulatedBroker:
    """Minimal gateway surface for :class:`tools.tools_executor.ToolExecutor`."""

    initial_cash: float
    replay: ReplaySessionDataProvider
    profit_profile: str = "balanced"
    cash_value: float = 0.0
    net_liquidation: float = 0.0
    _positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed_trades: list[SimFill] = field(default_factory=list)
    trade_log: list[SimTradeLog] = field(default_factory=list)
    _session_date: str = ""
    _now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.cash_value = float(self.initial_cash)
        self.net_liquidation = float(self.initial_cash)

    def __bool__(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def account_id(self) -> str:
        return "SIM"

    @property
    def day_trades_remaining(self) -> int:
        return 999

    def set_session_date(self, session_date: str) -> None:
        self._session_date = session_date

    def set_sim_time(self, now_utc: datetime) -> None:
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        self._now_utc = now_utc

    def _mark_price(self, symbol: str) -> float:
        q = self.replay.get_quote(symbol)
        if isinstance(q, dict):
            return float(q.get("last") or q.get("close") or 0)
        return float(getattr(q, "last", 0) or 0)

    def _update_nlv(self) -> None:
        mv = 0.0
        for sym, pos in self._positions.items():
            px = self._mark_price(sym)
            qty = int(pos["qty"])
            mv += px * qty
        self.net_liquidation = self.cash_value + mv

    def _record_closed_trade(
        self,
        *,
        symbol: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        entry_ts: str,
        pnl: float,
        exit_reason: str,
    ) -> None:
        rr = realized_r_multiple(
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            realized_pnl=pnl,
        )
        self.trade_log.append(
            SimTradeLog(
                profit_profile=self.profit_profile,
                symbol=symbol,
                qty=qty,
                entry_time_utc=entry_ts,
                exit_time_utc=self._now_utc.isoformat(),
                entry_price=entry_price,
                exit_price=exit_price,
                realized_pnl=pnl,
                realized_rr=rr,
                session_date=self._session_date,
                exit_reason=exit_reason,
            )
        )
        self.closed_trades.append(
            SimFill(
                symbol=symbol,
                side="SELL",
                qty=qty,
                price=exit_price,
                pnl=pnl,
                session_date=self._session_date,
            )
        )
        # QualityMatrix learning for backtests is batched in simulate_backtest (ingest_backtest_trades_and_refit).

    def get_cached_portfolio(self) -> list[Any]:
        out = []
        for sym, pos in self._positions.items():
            px = self._mark_price(sym)
            qty = int(pos["qty"])
            out.append(
                SimpleNamespace(
                    symbol=sym,
                    position=qty,
                    marketPrice=px,
                    marketValue=px * qty,
                    averageCost=float(pos["avg_cost"]),
                    unrealizedPNL=(px - float(pos["avg_cost"])) * qty,
                )
            )
        return out

    def get_cached_account_values(self) -> list[dict[str, Any]]:
        self._update_nlv()
        return [
            {"tag": "TotalCashValue", "value": str(self.cash_value)},
            {"tag": "NetLiquidation", "value": str(self.net_liquidation)},
        ]

    def get_cached_trades(self) -> list[Any]:
        return []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def refresh_positions(self) -> None:
        self._update_nlv()

    async def get_account_summary(self) -> dict[str, float]:
        self._update_nlv()
        return {"totalcashvalue": self.cash_value, "netliquidation": self.net_liquidation}

    async def get_position(self, symbol: str) -> dict[str, Any] | None:
        pos = self._positions.get(symbol.upper())
        if not pos:
            return None
        px = self._mark_price(symbol)
        qty = int(pos["qty"])
        return {
            "symbol": symbol.upper(),
            "qty": qty,
            "avg_cost": float(pos["avg_cost"]),
            "market_price": px,
            "unrealized_pnl": (px - float(pos["avg_cost"])) * qty,
        }

    async def get_open_orders(self) -> list[Any]:
        return []

    async def cancel_stops(self, underlying: str) -> dict[str, Any]:
        return {"cancelled": 0}

    async def close_option_position(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "skipped", "reason": "options_not_simulated"}

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        *,
        exit_reason: str = "sell",
    ) -> dict[str, Any]:
        sym = symbol.upper()
        side_u = side.upper()
        px = self._mark_price(sym)
        if px <= 0:
            return {"error": f"no replay price for {sym}"}
        qty = int(qty)
        if qty <= 0:
            return {"error": "quantity must be positive"}

        pos = self._positions.get(sym, {"qty": 0, "avg_cost": 0.0})
        cur_qty = int(pos["qty"])
        pnl = 0.0

        if side_u == "BUY":
            cost = px * qty
            if cost > self.cash_value:
                return {"error": f"insufficient cash: need {cost:.2f}, have {self.cash_value:.2f}"}
            if cur_qty == 0:
                pos["entry_ts"] = self._now_utc.isoformat()
            new_qty = cur_qty + qty
            if new_qty > 0:
                pos["avg_cost"] = (
                    (float(pos["avg_cost"]) * cur_qty + px * qty) / new_qty
                    if cur_qty > 0
                    else px
                )
            pos["qty"] = new_qty
            self.cash_value -= cost
        else:
            if cur_qty < qty:
                return {"error": f"cannot sell {qty}; position {cur_qty}"}
            entry_px = float(pos["avg_cost"])
            entry_ts = str(pos.get("entry_ts") or self._now_utc.isoformat())
            pnl = (px - entry_px) * qty
            pos["qty"] = cur_qty - qty
            self.cash_value += px * qty
            self._record_closed_trade(
                symbol=sym,
                qty=qty,
                entry_price=entry_px,
                exit_price=px,
                entry_ts=entry_ts,
                pnl=pnl,
                exit_reason=exit_reason,
            )
            if int(pos["qty"]) == 0:
                pos.pop("entry_ts", None)

        if int(pos.get("qty", 0)) == 0:
            self._positions.pop(sym, None)
        else:
            self._positions[sym] = pos

        self._update_nlv()
        return {"status": "filled", "symbol": sym, "side": side_u, "qty": qty, "price": px, "pnl": pnl}

    async def flatten_all(self) -> dict[str, Any]:
        closed = 0
        errors: list[str] = []
        for sym in list(self._positions.keys()):
            qty = int(self._positions[sym]["qty"])
            if qty > 0:
                r = await self.place_market_order(sym, "SELL", qty, exit_reason="eod_flatten")
                if r.get("error"):
                    errors.append(str(r["error"]))
                else:
                    closed += 1
        return {
            "positions_closed": closed,
            "positions_total": closed + len(errors),
            "errors": errors,
        }
