"""
Tool Execution Layer

Direct routing to broker methods. No abstraction, no business logic.
The agent decides what to call and with what parameters.

AVAILABLE TOOLS (for agent prompt):

=== RESEARCH ===
quote: {symbol} -> price, bid, ask, volume, change_pct
market_scan: {symbols?} -> top movers from liquid watchlist
candles: {symbol, days?=30, resolution?='D'} -> OHLCV data (resolution: D=daily, H=hourly, 5=5min, 15=15min, 1=1min, W=weekly, M=monthly)
fundamentals: {symbol} -> sector, industry, market_cap, pe_ratio, earnings_date
earnings: {symbol} -> next_earnings_date, days_until_earnings
atr: {symbol, period?=14} -> ATR value and ATR as % of price (for stop calibration)
iv_info: {symbol, dte_min, dte_max, strike_pct?=auto} -> current IV from ATM options (dte_min/dte_max required, strike_pct as decimal e.g. 0.15=±15%)
news: {symbol} -> recent headlines + basic sentiment (positive/negative/neutral)
analysts: {symbol} -> consensus, price targets, upside_pct, recent upgrades/downgrades
extended_fundamentals: {symbol} -> short_interest, beta, debt_to_equity, ROE, margins, growth
institutional_data: {symbol} -> institutional ownership %, top holders
insider_data: {symbol} -> recent insider buys/sells, net sentiment, transactions
peer_comparison: {symbol} -> 20-day performance vs sector ETF

=== KNOWLEDGE ===
market_hours: {} -> current session (premarket/regular/postmarket/closed), next transition time
budget: {} -> LLM cost tracking, today's P&L, budget remaining
economic_calendar: {} -> today's macro events (FOMC, NFP, CPI, etc.) + 3-day look-ahead

=== ORDER PLANNING ===
plan_order: {symbol, side, quantity, urgency?='normal', intent?='entry', execute?=false,
             stop_distance_pct?, stop_type?='trailing'|'fixed'|'none', trail_pct?,
             order_type?, limit_price?} -> recommended order type, params, stop, execution
  Agent overrides: pass stop_distance_pct/stop_type/trail_pct to control your stop.
                   pass order_type/limit_price to choose your entry order type.
                   Omit for smart ATR-based defaults.

=== OPTION ENTRY GATEWAY ===
enter_option: {symbol, strategy, quantity?=1, dte_target?=30, delta_target?=auto, max_spread_pct?=15, execute?=false} -> contract selection + execution

=== ACCOUNT STATE ===
positions: {} -> all positions with qty, avg_cost, unrealized_pnl
account: {} -> net_liq, available_funds, cash, pnl (CASH-ONLY, no margin)
open_orders: {} -> all open orders with order_id, symbol, action, qty, type, price
get_position: {symbol} -> single position details

=== ORDER MANAGEMENT ===
cancel_order: {order_id} -> cancels specific order
cancel_stops: {symbol} -> cancels all stop orders for symbol (verifies cancellation)
cancel_all_orphans: {} -> cancels ALL orders for symbols with no position (dangerous orphans!)
flatten_limits: {} -> cancel open orders and flatten all positions using LIMIT orders at midpoint

=== STOCK ORDERS - BASIC ===
market_order: {symbol, side, quantity} -> side='BUY'|'SELL'
limit_order: {symbol, side, quantity, limit_price}
stop_order: {symbol, side, quantity, stop_price}
stop_limit: {symbol, side, quantity, stop_price, limit_price}
trailing_stop: {symbol, quantity, direction, trail_percent} -> direction='LONG'|'SHORT', trail_percent as decimal (0.05 = 5%)
bracket_order: {symbol, side, quantity, limit_price, stop_loss, take_profit}

=== STOCK ORDERS - ADVANCED ===
modify_stop: {order_id, new_stop_price} -> adjust stop on existing order
oca_order: {symbol, quantity, direction, stop_price, target_price} -> OCA stop+target pair
moc_order: {symbol, side, quantity} -> Market-on-Close (closing auction)
loc_order: {symbol, side, quantity, limit_price} -> Limit-on-Close
moo_order: {symbol, side, quantity} -> Market-on-Open (opening auction)
loo_order: {symbol, side, quantity, limit_price} -> Limit-on-Open
trailing_stop_limit: {symbol, quantity, direction, trail_amount?, trail_percent?, limit_offset?=0.10}
adaptive_order: {symbol, side, quantity, order_type?='MKT', limit_price?, priority?='Normal'} -> IBKR adaptive algo
midprice_order: {symbol, side, quantity, price_cap?} -> pegged to bid/ask midpoint
relative_order: {symbol, side, quantity, offset?=0.01, limit_price?} -> pegged/relative order
gtd_order: {symbol, side, quantity, limit_price, good_till_date} -> Good-Till-Date ('YYYYMMDD HH:MM:SS')
fok_order: {symbol, side, quantity, limit_price} -> Fill-or-Kill
ioc_order: {symbol, side, quantity, limit_price} -> Immediate-or-Cancel

=== ALGO ORDERS ===
vwap_order: {symbol, side, quantity, start_time?, end_time?, max_pct_volume?=25} -> VWAP execution
twap_order: {symbol, side, quantity, start_time?, end_time?, randomize_pct?=55} -> TWAP execution
iceberg_order: {symbol, side, total_quantity, display_size, limit_price} -> hidden size
snap_mid_order: {symbol, side, quantity} -> snap-to-midpoint pegged

=== OPTIONS - SINGLE LEG ===
buy_option: {symbol, expiration, strike, right, quantity?=1} -> right='C'|'P', expiration='YYYYMMDD'
covered_call: {symbol, expiration, strike, shares?=100}
cash_secured_put: {symbol, expiration, strike, contracts?=1}
protective_put: {symbol, expiration, strike, shares?=100}

=== OPTIONS - SPREADS ===
vertical_spread: {symbol, expiration, long_strike, short_strike, right, quantity?=1}
iron_condor: {symbol, expiration, put_long_strike, put_short_strike, call_short_strike, call_long_strike, quantity?=1}
iron_butterfly: {symbol, expiration, center_strike, wing_width, quantity?=1}
straddle: {symbol, expiration, strike, quantity?=1}
strangle: {symbol, expiration, put_strike, call_strike, quantity?=1}
collar: {symbol, expiration, put_strike, call_strike, shares?=100}
calendar_spread: {symbol, strike, near_expiration, far_expiration, right?='C', quantity?=1}
diagonal_spread: {symbol, near_strike, far_strike, near_expiration, far_expiration, right?='C', quantity?=1}
butterfly: {symbol, expiration, lower_strike, middle_strike, upper_strike, right?='C', quantity?=1}
ratio_spread: {symbol, expiration, long_strike, short_strike, right?='C', ratio?=[1,2], quantity?=1}
jade_lizard: {symbol, expiration, put_strike, call_short_strike, call_long_strike, quantity?=1}

=== OPTIONS - MANAGEMENT ===
close_option: {symbol, expiration, strike, right, limit_price?} -> auto-midpoint if no limit_price
roll_option: {symbol, old_expiration, old_strike, new_expiration, new_strike, right, quantity?=1}
option_chain: {symbol, expiration?='YYYY-MM-DD', side?='call'|'put', dte_min?, dte_max?, strike_min?, strike_max?, limit?=20} -> contracts with Greeks (widen DTE or omit expiration if empty)
option_greeks: {symbol, expiration, strike, right} -> delta, gamma, theta, vega, IV for specific contract
position_greeks: {symbol?} -> Greeks for all option positions (or filtered by symbol)

=== SIZING ===
calculate_size: {symbol, side, stop_distance_pct?, risk_per_trade_pct?=1.5, max_position_pct?=20} -> recommended qty + plan_order_params

=== DISCOVERY ===
instrument_selector: {symbol?, outlook?='bullish'|'bearish'|'neutral'|'volatile', regime?, iv_dte_min?, iv_dte_max?, iv_strike_pct?} -> available instruments/strategies

=== OBSERVABILITY ===
stats: {} -> comprehensive performance stats (P&L, win rate, positions, LLM costs, action breakdown)
daily_summary: {} -> generate + persist daily summary to logs/daily_summary.json
review_trades: {days?=3, sort?='efficiency', symbol?} -> closed trades with efficiency ranking
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Structured result from tool execution."""
    action: str
    data: Any
    success: bool
    raw_json: str  # JSON string for backward compat (agent message injection)

    def __str__(self) -> str:
        return self.raw_json


# Build unified handler registry from submodules
from tools.tools_research import HANDLERS as _RESEARCH
from tools.tools_account import HANDLERS as _ACCOUNT
from tools.tools_orders import HANDLERS as _ORDERS
from tools.tools_options import HANDLERS as _OPTIONS, _normalize_expiration
from tools.tools_stats import HANDLERS as _STATS
from tools.tools_sizing import HANDLERS as _SIZING
from tools.tools_instruments import HANDLERS as _INSTRUMENTS
from tools.tools_scan import HANDLERS as _SCAN

_REGISTRY: dict[str, Any] = {}
_REGISTRY.update(_RESEARCH)
_REGISTRY.update(_ACCOUNT)
_REGISTRY.update(_ORDERS)
_REGISTRY.update(_OPTIONS)
_REGISTRY.update(_STATS)
_REGISTRY.update(_SIZING)
_REGISTRY.update(_INSTRUMENTS)
_REGISTRY.update(_SCAN)

# ── Tool Aliases (common LLM misspellings) ───────────────────
_ALIASES: dict[str, str] = {
    "options_chain": "option_chain",
    "options": "option_chain",
    "get_options": "option_chain",
    "get_option_chain": "option_chain",
    "get_quote": "quote",
    "get_candles": "candles",
    "get_atr": "atr",
    "get_account": "account",
    "get_positions": "positions",
    "scan": "market_scan",
    "get_news": "news",
    "econ_calendar": "economic_calendar",
    "calendar": "economic_calendar",
    "macro_events": "economic_calendar",
    "fundamentals_extended": "extended_fundamentals",
    "buy_stock": "market_order",
    "sell_stock": "market_order",
    "close": "close_position",
    "bull_call_spread": "vertical_spread",
    "bear_put_spread": "vertical_spread",
    "bear_call_spread": "vertical_spread",
    "bull_put_spread": "vertical_spread",
    "call_spread": "vertical_spread",
    "put_spread": "vertical_spread",
    "debit_spread": "vertical_spread",
    "credit_spread": "vertical_spread",
}

# OCC symbol pattern: e.g. SMCI260220C00031000 or AAPL260321P00250000
_OCC_RE = re.compile(r'^([A-Z]{1,6})(\d{6})([CP])(\d{8})$')

def _parse_occ_symbol(sym: str) -> dict | None:
    """Parse an OCC option symbol into components, or None if not OCC."""
    m = _OCC_RE.match((sym or "").upper().strip())
    if not m:
        return None
    underlying, date_str, right_char, strike_raw = m.groups()
    # date_str = YYMMDD -> YYYYMMDD
    expiration = f"20{date_str}"
    # strike_raw = 00031000 -> 31.0  (8 digits, last 3 are decimal)
    strike = int(strike_raw) / 1000.0
    right = "C" if right_char == "C" else "P"
    return {"underlying": underlying, "expiration": expiration, "strike": strike, "right": right}

# Inline order action names (replaces deleted tool_registry dependency)
_ORDER_ACTIONS = {
    "market_order", "limit_order", "stop_order", "stop_limit",
    "trailing_stop", "bracket_order", "modify_stop", "oca_order",
    "flatten_limits", "moc_order", "loc_order", "moo_order", "loo_order",
    "trailing_stop_limit", "adaptive_order", "midprice_order", "relative_order",
    "gtd_order", "fok_order", "ioc_order", "vwap_order", "twap_order",
    "iceberg_order", "snap_mid_order", "close_position",
    "buy_option", "covered_call", "cash_secured_put", "protective_put",
    "vertical_spread", "iron_condor", "iron_butterfly", "straddle",
    "strangle", "collar", "calendar_spread", "diagonal_spread",
    "butterfly", "ratio_spread", "jade_lizard", "close_option", "roll_option",
    "plan_order", "enter_option",
}


def get_valid_actions() -> list[str]:
    """Return sorted list of all valid action names from the tool registry."""
    return sorted(_REGISTRY.keys())


class ToolExecutor:

    _OPTION_STRATEGY_DEFAULTS = {
        "long_call":        (0.40, 21, 45),
        "long_put":         (0.40, 21, 45),
        "bull_call_spread": (0.40, 21, 45),
        "bear_put_spread":  (0.40, 21, 45),
        "iron_condor":      (0.16, 30, 60),
        "covered_call":     (0.30, 14, 30),
        "cash_secured_put": (0.30, 14, 30),
        "protective_put":   (0.30, 30, 60),
        "straddle":         (0.50, 21, 45),
        "strangle":         (0.25, 21, 45),
    }

    def __init__(self, gateway, data_provider, market_hours_provider=None):
        self.gateway = gateway
        self.data_provider = data_provider
        self.market_hours_provider = market_hours_provider
        self._protective_order_actions = {"stop_order", "stop_limit", "trailing_stop", "trailing_stop_limit"}
        self._deferred_stops = []
        self._recent_orders: dict[str, float] = {}  # fingerprint → timestamp for idempotency
        # Order action names for structured trade logging
        self._order_actions = _ORDER_ACTIONS

        # Load cash_only flag from env
        self.cash_only = os.environ.get("CASH_ONLY", "true").lower() == "true"

        if self.market_hours_provider is None:
            from data.market_hours import get_market_hours_provider
            self.market_hours_provider = get_market_hours_provider()

        # Deferred stops removed in minimal build

    async def _refresh_state(self):
        """Refresh positions/account from broker after order placement."""
        try:
            await self.gateway.refresh_positions()
        except Exception as e:
            logger.debug(f"State refresh failed: {e}")

    async def _close_position(self, symbol: str, quantity: int | None = None, reason: str = "") -> dict:
        """Close a single position: cancel its stops then market-close.

        Handles stock (long/short) and option positions.  If *quantity* is
        ``None`` the full position size is used.  Returns a result dict.
        """
        symbol = symbol.upper()
        pos = await self.gateway.get_position(symbol)
        if pos is None:
            return {"error": f"No open position for {symbol}"}

        pos_qty = pos.get("quantity", 0)
        qty = quantity if quantity is not None else abs(int(pos_qty))
        if qty <= 0:
            return {"error": f"Invalid quantity {qty} for {symbol}"}

        is_long = pos_qty > 0
        is_option = pos.get("sec_type") == "OPT"

        # 1. Cancel protective stops for this symbol
        try:
            underlying = symbol.split("_")[0]
            await self.gateway.cancel_stops(underlying)
        except Exception as e:
            logger.warning(f"_close_position: cancel_stops failed for {symbol}: {e}")

        # 2. Place closing order
        try:
            if is_option:
                parts = symbol.split("_")  # SYMBOL_RIGHT_STRIKE_EXPIRY
                close_params: dict = {"symbol": parts[0]}
                if len(parts) >= 4:
                    close_params["right"] = parts[1]
                    close_params["strike"] = float(parts[2])
                    close_params["expiration"] = parts[3]
                close_params["quantity"] = qty
                close_params["reason"] = reason or "close_position"
                result = await self.gateway.close_option_position(**close_params)
            else:
                side = "SELL" if is_long else "BUY"
                result = await self.gateway.place_market_order(symbol, side, qty)
        except Exception as e:
            logger.error(f"_close_position: close order failed for {symbol}: {e}")
            return {"error": f"Close order failed: {e}", "symbol": symbol}

        await self._refresh_state()

        logger.info(f"_close_position: closed {qty} {symbol} — {reason}")
        return {
            "success": True,
            "symbol": symbol,
            "quantity": qty,
            "side": "SELL" if is_long else "BUY",
            "reason": reason,
            "result": result,
        }

    def _check_pdt(self, side: str):
        """Block BUY-side orders when PDT restricted. Returns error dict or None."""
        if side.upper() != 'BUY':
            return None
        if not self.gateway:
            return None
        if (self.gateway.day_trades_remaining <= 0
                and self.gateway.net_liquidation < 25_000):
            return {
                "error": f"PDT BLOCKED: 0 day trades remaining and account under $25k. "
                         f"Day trades left: {self.gateway.day_trades_remaining}, "
                         f"Net liquidation: ${self.gateway.net_liquidation:,.2f}"
            }
        return None

    def _check_cash_only(self, side: str, symbol: str, intent: str = "entry") -> dict | None:
        """STRICT cash-only guardrail: block any order that would create a short stock position.

        Returns error dict if blocked, None if allowed.
        Rules:
        - BUY side → always allowed (buying stock with cash)
        - SELL side → only allowed if a long position exists in the symbol
          (regardless of intent label — no naked shorts in a cash account)
        """
        if not self.cash_only:
            return None
        if side.upper() != "SELL":
            return None  # BUY always OK in cash-only
        # SELL is only allowed if we hold a long position in this symbol.
        # Check cached portfolio synchronously to avoid async in this guard.
        portfolio = self.gateway.get_cached_portfolio() if self.gateway else []
        for item in portfolio:
            if (item.contract.symbol.upper() == (symbol or "").upper()
                    and item.position > 0):
                return None  # selling shares we own — allowed
        return {
            "error": f"CASH-ONLY BLOCKED: Cannot SELL {symbol} — no long position held. "
                     f"Cash accounts cannot open short stock positions. "
                     f"For bearish views, use long puts or bear put spreads instead."
        }

    def _check_cash(self, estimated_cost: float):
        """Block if estimated cost exceeds available funds. Returns error dict or None.
        
        Uses AvailableFunds (what you can actually spend) instead of
        TotalCashValue (which can be misleading in margin accounts).
        CASH-ONLY: Never use BuyingPower.
        """
        if not self.gateway:
            return None
        # Prefer AvailableFunds over TotalCashValue
        cash = getattr(self.gateway, 'available_funds', 0) or self.gateway.cash_value
        try:
            for av in self.gateway.get_cached_account_values():
                if av.currency == 'USD':
                    if av.tag == 'AvailableFunds':
                        cash = float(av.value)
                        self.gateway.available_funds = cash
                        break
        except Exception:
            pass
        if cash <= 0:
            # Fallback to TotalCashValue if AvailableFunds not available
            try:
                for av in self.gateway.get_cached_account_values():
                    if av.tag == 'TotalCashValue' and av.currency == 'USD':
                        cash = float(av.value)
                        break
            except Exception:
                pass
        if estimated_cost > cash:
            return {
                "error": f"INSUFFICIENT CASH: Order requires ~${estimated_cost:,.2f} "
                         f"but only ${cash:,.2f} available (AvailableFunds). "
                         f"This is a CASH-ONLY account - no margin."
            }
        return None

    def _plan_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        urgency: str = "normal",
        intent: str = "entry",
        # --- Agent-controlled overrides ---
        stop_distance_pct: float = None,  # Override auto stop distance (e.g. 5.0 = 5%)
        stop_type: str = None,            # "trailing", "fixed", "none" — override auto stop type
        trail_pct: float = None,          # Override trailing % as decimal (0.05 = 5%)
        order_type: str = None,           # Override entry order type (e.g. "limit_order", "market_order")
        limit_price: float = None,        # For limit order override
    ) -> dict:
        """
        Order-type recommender with agent overrides.

        If the agent passes stop/order params, those are used.
        Otherwise falls back to the auto decision tree.
        """
        # Fetch quote, ATR, and earnings concurrently (all independent)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as pool:
            quote_fut = pool.submit(self.data_provider.get_quote, symbol)
            atr_fut = pool.submit(self.data_provider.get_atr, symbol)
            earnings_fut = pool.submit(self.data_provider.get_earnings_info, symbol)
            quote = quote_fut.result()
            atr_result = atr_fut.result()
            earnings = earnings_fut.result()

        session_info = self.market_hours_provider.get_session_info()
        session = session_info.get("session", "closed")

        iv_info = None
        # IV info requires agent-provided DTE params; skip in auto order planning

        price = None
        bid = None
        ask = None
        spread = None
        spread_pct = None
        volume = None

        if quote:
            price = quote.last
            bid = quote.bid
            ask = quote.ask
            volume = quote.volume
            mid = quote.mid
            if bid and ask and bid > 0 and ask > 0 and mid and mid > 0:
                spread = round(ask - bid, 4)
                spread_pct = round(spread / mid * 100, 4)

        atr_value = atr_result.value if atr_result else None
        atr_pct = None
        if atr_value and price and price > 0:
            atr_pct = round(atr_value / price * 100, 2)

        days_to_earnings = None
        if earnings and earnings.days_until_earnings is not None:
            days_to_earnings = earnings.days_until_earnings

        # Use pre-computed values from market_hours provider
        minutes_to_open = session_info.get("minutes_to_open")
        minutes_to_close = session_info.get("minutes_to_close")

        # --- Order type override or auto decision tree ---
        if order_type and intent != "stop":
            # Agent explicitly chose the order type
            reasons = [f"Agent override: {order_type}"]
            suggested_params = {"symbol": symbol, "side": side, "quantity": quantity}
            if limit_price is not None:
                suggested_params["limit_price"] = round(float(limit_price), 2)
            if order_type == "trailing_stop":
                suggested_params["direction"] = "LONG" if side == "BUY" else "SHORT"
                if trail_pct:
                    suggested_params["trail_percent"] = float(trail_pct)
            recommended = order_type
        elif intent == "stop":
            recommended, reasons, suggested_params = self._plan_stop(
                symbol, side, quantity, price, atr_value, atr_pct,
                days_to_earnings, spread, spread_pct
            )
        elif intent == "exit":
            recommended, reasons, suggested_params = self._plan_exit(
                symbol, side, quantity, price, urgency, session,
                spread, spread_pct, minutes_to_close
            )
        else:
            recommended, reasons, suggested_params = self._plan_entry(
                symbol, side, quantity, price, urgency, session,
                spread, spread_pct, minutes_to_open, minutes_to_close,
                atr_pct, limit_price=limit_price
            )

        # --- Stop recommendation: agent override or auto ---
        stop_rec = None
        if intent not in ("stop", "exit") and price:
            if stop_type == "none":
                # Agent explicitly says no stop
                stop_rec = None
            elif stop_type or stop_distance_pct or trail_pct:
                # Agent provided stop params — build stop from those
                stop_rec = self._build_agent_stop(
                    price=price, side=side, atr_value=atr_value, atr_pct=atr_pct,
                    stop_distance_pct=stop_distance_pct, stop_type=stop_type,
                    trail_pct=trail_pct,
                )
            elif atr_value:
                # Auto mode — existing logic
                if side == "BUY":
                    stop_rec = self._recommend_stop(price, atr_value, atr_pct, days_to_earnings)
                else:
                    stop_rec = self._recommend_stop_short(price, atr_value, atr_pct, days_to_earnings)

        data_snapshot = {
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "atr": atr_value,
            "atr_pct": atr_pct,
            "days_to_earnings": days_to_earnings,
            "iv_current": iv_info.iv_current if iv_info else None,
            "iv_rank": iv_info.iv_rank if iv_info else None,
            "session": session,
            "minutes_to_open": minutes_to_open,
            "minutes_to_close": minutes_to_close,
            "volume": volume,
        }

        return {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "intent": intent,
            "urgency": urgency,
            "recommendation": recommended,
            "reasoning": reasons,
            "suggested_params": suggested_params,
            "stop_recommendation": stop_rec,
            "data_snapshot": data_snapshot,
            "agent_overrides": {
                "order_type": order_type,
                "stop_type": stop_type,
                "stop_distance_pct": stop_distance_pct,
                "trail_pct": trail_pct,
                "limit_price": limit_price,
            },
        }

    def _plan_entry(self, symbol, side, quantity, price, urgency,
                    session, spread, spread_pct, minutes_to_open,
                    minutes_to_close, atr_pct, limit_price=None):
        """Entry order decision tree. Returns (order_type, reasons, params)."""
        reasons = []
        params = {"symbol": symbol, "side": side, "quantity": quantity}

        if session == "premarket":
            if limit_price is not None:
                reasons.append(f"Premarket: LOO (limit-on-open) at ${limit_price:.2f} for opening auction")
                params["limit_price"] = round(limit_price, 2)
                return ("loo_order", reasons, params)
            else:
                reasons.append("Premarket: MOO (market-on-open) for opening auction fill")
                return ("moo_order", reasons, params)

        if session == "postmarket":
            reasons.append("Extended hours (postmarket): only limit orders supported")
            if price:
                params["limit_price"] = round(price, 2)
            return ("limit_order", reasons, params)

        if session == "closed":
            reasons.append("Market closed: cannot place order now")
            return ("WAIT", reasons, {})

        if minutes_to_open is not None and 0 < minutes_to_open <= 5:
            reasons.append(f"Within {minutes_to_open} min of open: opening auction order")
            return ("moo_order", reasons, params)

        if minutes_to_close is not None and 0 < minutes_to_close <= 10:
            reasons.append(f"Within {minutes_to_close} min of close: closing auction order")
            return ("moc_order", reasons, params)

        if price and quantity * price > 100_000:
            reasons.append(f"Very large order (${quantity * price:,.0f}): TWAP for time distribution")
            return ("twap_order", reasons, params)
        if price and quantity * price > 50_000:
            reasons.append(f"Large order (${quantity * price:,.0f}): VWAP for volume matching")
            return ("vwap_order", reasons, params)

        spread_tight = spread_pct is not None and spread_pct < 0.10
        if urgency == "high" and spread_tight:
            reasons.append(f"High urgency + tight spread ({spread_pct:.3f}%): market order")
            return ("market_order", reasons, params)

        if urgency == "high":
            spread_str = f"{spread_pct:.3f}%" if spread_pct else "unknown"
            reasons.append(f"High urgency, spread {spread_str}: adaptive with urgent priority")
            params["priority"] = "Urgent"
            return ("adaptive_order", reasons, params)

        if spread_pct is not None and spread_pct > 0.30:
            reasons.append(f"Wide spread ({spread_pct:.3f}%): midprice order to capture mid")
            if price:
                params["price_cap"] = round(price * 1.002, 2)
            return ("midprice_order", reasons, params)

        if spread_pct is not None and spread_pct > 0.10 and urgency == "low":
            reasons.append(f"Moderate spread ({spread_pct:.3f}%) + low urgency: adaptive patient")
            params["priority"] = "Patient"
            return ("adaptive_order", reasons, params)

        reasons.append("Normal conditions: adaptive order (IBKR algo for best execution)")
        params["priority"] = "Normal"
        return ("adaptive_order", reasons, params)

    def _plan_exit(self, symbol, side, quantity, price, urgency,
                   session, spread, spread_pct, minutes_to_close):
        """Exit order decision tree. Returns (order_type, reasons, params)."""
        reasons = []
        params = {"symbol": symbol, "side": side, "quantity": quantity}

        if session in ("premarket", "postmarket"):
            reasons.append(f"Extended hours: limit order only")
            if price:
                params["limit_price"] = round(price, 2)
            return ("limit_order", reasons, params)

        if session == "closed":
            reasons.append("Market closed")
            return ("WAIT", reasons, {})

        if urgency == "high":
            reasons.append("High urgency exit: market order for guaranteed fill")
            return ("market_order", reasons, params)

        if minutes_to_close is not None and 0 < minutes_to_close <= 10:
            reasons.append(f"Near close ({minutes_to_close} min): MOC for closing auction")
            return ("moc_order", reasons, params)

        if spread_pct is not None and spread_pct > 0.30:
            reasons.append(f"Wide spread ({spread_pct:.3f}%): midprice for better fill")
            return ("midprice_order", reasons, params)

        reasons.append("Standard exit: adaptive order")
        params["priority"] = "Normal"
        return ("adaptive_order", reasons, params)

    def _plan_stop(self, symbol, side, quantity, price, atr_value,
                   atr_pct, days_to_earnings, spread, spread_pct):
        """Stop order type decision tree. Returns (order_type, reasons, params)."""
        reasons = []
        params = {"symbol": symbol, "side": side, "quantity": quantity}

        if not price or not atr_value:
            reasons.append("Insufficient data: defaulting to basic stop")
            return ("stop_order", reasons, params)

        if days_to_earnings is not None and days_to_earnings <= 3:
            reasons.append(f"Earnings in {days_to_earnings} days: stop_limit for gap protection")
            if side == "SELL":
                stop_price = round(price - atr_value, 2)
                limit_price = round(stop_price - atr_value * 0.5, 2)
            else:
                stop_price = round(price + atr_value, 2)
                limit_price = round(stop_price + atr_value * 0.5, 2)
            params["stop_price"] = stop_price
            params["limit_price"] = limit_price
            reasons.append(f"Stop: ${stop_price}, Limit: ${limit_price} (0.5 ATR slippage buffer)")
            return ("stop_limit", reasons, params)

        if atr_pct and atr_pct > 3.0:
            reasons.append(f"High volatility (ATR {atr_pct}%): trailing stop to ride momentum")
            trail_pct = round(min(atr_pct * 1.5, 15.0) / 100, 4)
            direction = "LONG" if side == "SELL" else "SHORT"
            params = {
                "symbol": symbol,
                "quantity": quantity,
                "direction": direction,
                "trail_percent": trail_pct,
            }
            reasons.append(f"Trail: {trail_pct * 100:.1f}% (1.5x ATR%)")
            return ("trailing_stop", reasons, params)

        if spread_pct and spread_pct > 0.50:
            reasons.append(f"Wide spread ({spread_pct:.2f}%): stop_limit to control fill price")
            if side == "SELL":
                stop_price = round(price - atr_value, 2)
                limit_price = round(stop_price - (spread * 2 if spread else price * 0.005), 2)
            else:
                stop_price = round(price + atr_value, 2)
                limit_price = round(stop_price + (spread * 2 if spread else price * 0.005), 2)
            params["stop_price"] = stop_price
            params["limit_price"] = limit_price
            return ("stop_limit", reasons, params)

        reasons.append("Standard conditions: basic stop order")
        if side == "SELL":
            stop_price = round(price - atr_value, 2)
        else:
            stop_price = round(price + atr_value, 2)
        params["stop_price"] = stop_price
        reasons.append(f"Stop: ${stop_price} (1 ATR from current)")
        return ("stop_order", reasons, params)

    def _build_agent_stop(self, price, side, atr_value, atr_pct,
                          stop_distance_pct=None, stop_type=None, trail_pct=None):
        """Build stop recommendation from agent-provided parameters.
        
        The agent controls the stop. We just do the math.
        Falls back to ATR-based defaults only for fields the agent didn't specify.
        """
        is_long = side == "BUY"

        # Determine effective stop type
        effective_type = stop_type or "fixed"
        if trail_pct and not stop_type:
            effective_type = "trailing"

        # Determine effective stop distance
        if stop_distance_pct is not None:
            eff_dist_pct = float(stop_distance_pct)
        elif trail_pct and effective_type == "trailing":
            eff_dist_pct = float(trail_pct) * 100 if trail_pct > 1 else float(trail_pct) * 100
        elif atr_pct:
            eff_dist_pct = round(min(atr_pct * 1.5, 15.0), 2)
        else:
            eff_dist_pct = 5.0

        risk_pct = round(eff_dist_pct, 2)

        if effective_type == "trailing":
            # Trail percent: agent may pass as decimal (0.08) or whole (8.0)
            if trail_pct:
                eff_trail = float(trail_pct)
                if eff_trail > 1:
                    eff_trail = eff_trail / 100  # Convert 8.0 -> 0.08
            else:
                eff_trail = eff_dist_pct / 100

            return {
                "stop_type": "trailing_stop",
                "trail_percent": round(eff_trail, 4),
                "direction": "LONG" if is_long else "SHORT",
                "risk_pct": risk_pct,
                "note": f"Agent: trailing {eff_trail * 100:.1f}%",
            }
        else:
            # Fixed stop
            if is_long:
                stop_price = round(price * (1 - eff_dist_pct / 100), 2)
            else:
                stop_price = round(price * (1 + eff_dist_pct / 100), 2)

            return {
                "stop_type": "stop_order",
                "stop_price": stop_price,
                "side": "SELL" if is_long else "BUY",
                "risk_pct": risk_pct,
                "note": f"Agent: fixed stop {eff_dist_pct:.1f}% away",
            }

    def _recommend_stop(self, price, atr_value, atr_pct, days_to_earnings):
        """Companion stop recommendation for long entry.
        
        Uses plain stop orders (not stop_limit) to avoid Error 202
        'Limit Price too far through Stop Price' rejections from IBKR.
        For earnings plays, uses trailing stop instead of stop_limit.
        """
        stop_price = round(price - atr_value * 1.5, 2)
        risk_pct = round((price - stop_price) / price * 100, 2)

        if atr_pct and atr_pct > 3.0:
            trail = round(min(atr_pct * 1.5, 15.0) / 100, 4)
            return {
                "stop_type": "trailing_stop",
                "trail_percent": trail,
                "direction": "LONG",
                "risk_pct": risk_pct,
                "note": f"Trailing {trail * 100:.1f}% (1.5x ATR%) | high volatility",
            }

        if days_to_earnings is not None and days_to_earnings <= 3:
            # Tighter stop near earnings — but still plain stop to avoid IBKR rejection
            tight_stop = round(price - atr_value * 1.0, 2)
            tight_risk = round((price - tight_stop) / price * 100, 2)
            return {
                "stop_type": "stop_order",
                "stop_price": tight_stop,
                "side": "SELL",
                "risk_pct": tight_risk,
                "note": f"1.0 ATR below entry | EARNINGS in {days_to_earnings}d: tight stop",
            }

        return {
            "stop_type": "stop_order",
            "stop_price": stop_price,
            "side": "SELL",
            "risk_pct": risk_pct,
            "note": "1.5 ATR below entry",
        }

    def _recommend_stop_short(self, price, atr_value, atr_pct, days_to_earnings):
        """Companion stop recommendation for short entry.
        
        Uses plain stop orders to avoid Error 202 stop_limit rejections.
        """
        stop_price = round(price + atr_value * 1.5, 2)
        risk_pct = round((stop_price - price) / price * 100, 2)

        if days_to_earnings is not None and days_to_earnings <= 3:
            tight_stop = round(price + atr_value * 1.0, 2)
            tight_risk = round((tight_stop - price) / price * 100, 2)
            return {
                "stop_type": "stop_order",
                "stop_price": tight_stop,
                "side": "BUY",
                "risk_pct": tight_risk,
                "note": f"1.0 ATR above entry (BUY stop) | EARNINGS in {days_to_earnings}d: tight stop",
            }

        return {
            "stop_type": "stop_order",
            "stop_price": stop_price,
            "side": "BUY",
            "risk_pct": risk_pct,
            "note": "1.5 ATR above entry (BUY stop for short cover)",
        }

    def _enter_option(
        self,
        symbol: str,
        strategy: str,
        quantity: int = 1,
        dte_target: int = 30,
        delta_target: float = None,
        max_spread_pct: float = 15.0,
    ) -> dict:
        """
        Deterministic option contract selector.

        Gathers market data, picks the optimal contract(s) for the requested
        strategy, and returns a plan with dispatch_action + dispatch_params
        ready for _safe_dispatch().
        """
        if strategy not in self._OPTION_STRATEGY_DEFAULTS:
            return {
                "error": f"Unknown strategy '{strategy}'. Valid: {list(self._OPTION_STRATEGY_DEFAULTS.keys())}"
            }

        defaults = self._OPTION_STRATEGY_DEFAULTS[strategy]
        delta = delta_target if delta_target is not None else defaults[0]
        dte_min = defaults[1]
        dte_max = defaults[2]

        quote = self.data_provider.get_quote(symbol)
        if not quote or not quote.last:
            return {"error": f"No quote data for {symbol}"}
        price = quote.last

        iv_info = None
        try:
            iv_info = self.data_provider.get_iv_info(
                symbol, dte_min=dte_min, dte_max=dte_max
            )
        except Exception:
            pass

        atr_result = None
        try:
            atr_result = self.data_provider.get_atr(symbol)
        except Exception:
            pass

        earnings = None
        try:
            earnings = self.data_provider.get_earnings_info(symbol)
        except Exception:
            pass
        days_to_earnings = earnings.days_until_earnings if earnings else None

        earnings_warning = None
        if days_to_earnings is not None and days_to_earnings <= 7:
            earnings_warning = f"Earnings in {days_to_earnings} days - elevated IV/risk"

        chain = self.data_provider.get_option_chain(
            symbol, dte_range=(dte_min, dte_max)
        )
        if not chain or not chain.contracts:
            return {"error": f"No option chain for {symbol} in DTE range {dte_min}-{dte_max}"}

        data_snapshot = {
            "price": price,
            "iv_current": iv_info.iv_current if iv_info else None,
            "iv_rank": iv_info.iv_rank if iv_info else None,
            "atr": round(atr_result.value, 2) if atr_result else None,
            "days_to_earnings": days_to_earnings,
            "chain_contracts": len(chain.contracts),
        }

        if strategy in ("long_call", "long_put"):
            return self._select_single_leg(
                symbol, strategy, chain, price, delta, dte_target,
                max_spread_pct, quantity, iv_info, earnings_warning, data_snapshot
            )

        elif strategy in ("bull_call_spread", "bear_put_spread"):
            return self._select_vertical_spread(
                symbol, strategy, chain, price, delta, dte_target,
                max_spread_pct, quantity, iv_info, earnings_warning, data_snapshot
            )

        elif strategy == "iron_condor":
            return self._select_iron_condor(
                symbol, chain, price, delta, dte_target,
                max_spread_pct, quantity, iv_info, earnings_warning, data_snapshot
            )

        elif strategy in ("covered_call", "cash_secured_put", "protective_put"):
            return self._select_income_protection(
                symbol, strategy, chain, price, delta, dte_target,
                max_spread_pct, quantity, iv_info, earnings_warning, data_snapshot
            )

        elif strategy in ("straddle", "strangle"):
            return self._select_straddle_strangle(
                symbol, strategy, chain, price, delta, dte_target,
                max_spread_pct, quantity, iv_info, earnings_warning, data_snapshot
            )

        return {"error": f"Strategy '{strategy}' not yet implemented"}

    @staticmethod
    def _best_contract(candidates, delta_target, max_spread_pct):
        """Pick best contract: filter spread, sort by delta proximity."""
        valid = [c for c in candidates if c.spread_pct is not None and c.spread_pct <= max_spread_pct]
        if not valid:
            valid = candidates
        valid.sort(key=lambda c: abs((abs(c.delta) if c.delta else 0) - delta_target))
        return valid[0] if valid else None

    @staticmethod
    def _contract_dict(c):
        """Format an OptionContract for output."""
        return {
            "strike": c.strike,
            "expiration": c.expiration,
            "right": "C" if c.side == "call" else "P",
            "delta": c.delta,
            "gamma": c.gamma,
            "theta": c.theta,
            "vega": c.vega,
            "iv": c.iv,
            "dte": c.dte,
            "bid": c.bid,
            "ask": c.ask,
            "mid": c.mid,
            "spread_pct": round(c.spread_pct, 1) if c.spread_pct else None,
        }

    def _select_single_leg(self, symbol, strategy, chain, price, delta,
                           dte_target, max_spread_pct, quantity,
                           iv_info, earnings_warning, data_snapshot):
        """Select contract for long_call or long_put."""
        is_call = strategy == "long_call"
        contracts = chain.calls() if is_call else chain.puts()

        contracts = [c for c in contracts
                     if c.dte is not None and abs(c.dte - dte_target) <= 15]
        contracts = [c for c in contracts
                     if c.delta is not None and abs(abs(c.delta) - delta) <= 0.15]

        if not contracts:
            return {"error": f"No {'call' if is_call else 'put'} contracts near delta {delta} and DTE {dte_target}"}

        best = self._best_contract(contracts, delta, max_spread_pct)
        if not best:
            return {"error": "No contracts pass spread filter"}

        cost = (best.ask or best.mid or 0) * 100 * quantity

        if self.gateway and self.gateway.cash_value > 0 and cost > self.gateway.cash_value:
            return {
                "error": f"Insufficient cash: need ${cost:.2f}, have ${self.gateway.cash_value:.2f}",
                "selected_contract": self._contract_dict(best),
            }

        return {
            "strategy": strategy,
            "symbol": symbol,
            "selected_contract": self._contract_dict(best),
            "estimated_cost": round(cost, 2),
            "max_loss": round(cost, 2),
            "iv_context": {
                "iv_current": iv_info.iv_current if iv_info else None,
                "iv_rank": iv_info.iv_rank if iv_info else None,
            },
            "earnings_warning": earnings_warning,
            "data_snapshot": data_snapshot,
            "dispatch_action": "buy_option",
            "dispatch_params": {
                "symbol": symbol,
                "expiration": best.expiration,
                "strike": best.strike,
                "right": "C" if is_call else "P",
                "quantity": quantity,
            },
        }

    def _select_vertical_spread(self, symbol, strategy, chain, price, delta,
                                dte_target, max_spread_pct, quantity,
                                iv_info, earnings_warning, data_snapshot):
        """Select contracts for bull_call_spread or bear_put_spread."""
        is_bull_call = strategy == "bull_call_spread"
        contracts = chain.calls() if is_bull_call else chain.puts()

        contracts = [c for c in contracts
                     if c.dte is not None and abs(c.dte - dte_target) <= 15]

        if len(contracts) < 2:
            return {"error": f"Not enough contracts for {strategy}"}

        expirations = {}
        for c in contracts:
            expirations.setdefault(c.expiration, []).append(c)

        best_exp = min(expirations.keys(),
                       key=lambda e: abs((expirations[e][0].dte or dte_target) - dte_target))
        exp_contracts = expirations[best_exp]
        exp_contracts.sort(key=lambda c: c.strike)

        short_delta = delta * 0.5
        long_candidates = [c for c in exp_contracts
                           if c.delta and abs(abs(c.delta) - delta) <= 0.15]
        short_candidates = [c for c in exp_contracts
                            if c.delta and abs(abs(c.delta) - short_delta) <= 0.15]

        if not long_candidates or not short_candidates:
            return {"error": f"Cannot find suitable long/short legs for {strategy}"}

        long_leg = self._best_contract(long_candidates, delta, max_spread_pct * 2)
        short_leg = self._best_contract(short_candidates, short_delta, max_spread_pct * 2)

        if not long_leg or not short_leg or long_leg.strike == short_leg.strike:
            return {"error": "Could not form valid spread — legs have same strike"}

        if is_bull_call:
            long_strike = min(long_leg.strike, short_leg.strike)
            short_strike = max(long_leg.strike, short_leg.strike)
            long_c = next((c for c in exp_contracts if c.strike == long_strike), long_leg)
            short_c = next((c for c in exp_contracts if c.strike == short_strike), short_leg)
        else:
            long_strike = max(long_leg.strike, short_leg.strike)
            short_strike = min(long_leg.strike, short_leg.strike)
            long_c = next((c for c in exp_contracts if c.strike == long_strike), long_leg)
            short_c = next((c for c in exp_contracts if c.strike == short_strike), short_leg)

        width = abs(long_strike - short_strike)
        debit = ((long_c.ask or long_c.mid or 0) - (short_c.bid or short_c.mid or 0))
        max_loss = round(debit * 100 * quantity, 2)

        return {
            "strategy": strategy,
            "symbol": symbol,
            "long_leg": self._contract_dict(long_c),
            "short_leg": self._contract_dict(short_c),
            "spread_width": width,
            "estimated_debit": round(debit, 2),
            "max_loss": max_loss,
            "max_profit": round((width - debit) * 100 * quantity, 2),
            "iv_context": {
                "iv_current": iv_info.iv_current if iv_info else None,
                "iv_rank": iv_info.iv_rank if iv_info else None,
            },
            "earnings_warning": earnings_warning,
            "data_snapshot": data_snapshot,
            "dispatch_action": "vertical_spread",
            "dispatch_params": {
                "symbol": symbol,
                "expiration": best_exp,
                "long_strike": long_c.strike,
                "short_strike": short_c.strike,
                "right": "C" if is_bull_call else "P",
                "quantity": quantity,
            },
        }

    def _select_iron_condor(self, symbol, chain, price, delta, dte_target,
                            max_spread_pct, quantity,
                            iv_info, earnings_warning, data_snapshot):
        """Select 4 legs for iron condor."""
        calls = chain.calls()
        puts = chain.puts()

        calls = [c for c in calls if c.dte is not None and abs(c.dte - dte_target) <= 20]
        puts = [c for c in puts if c.dte is not None and abs(c.dte - dte_target) <= 20]

        if not calls or not puts:
            return {"error": "Not enough contracts for iron condor"}

        all_dtes = {c.expiration: c.dte for c in calls + puts if c.dte}
        if not all_dtes:
            return {"error": "No DTE data available"}
        best_exp = min(all_dtes.keys(), key=lambda e: abs(all_dtes[e] - dte_target))

        exp_calls = sorted([c for c in calls if c.expiration == best_exp], key=lambda c: c.strike)
        exp_puts = sorted([c for c in puts if c.expiration == best_exp], key=lambda c: c.strike)

        short_call_candidates = [c for c in exp_calls if c.delta and abs(abs(c.delta) - delta) <= 0.10]
        short_put_candidates = [c for c in exp_puts if c.delta and abs(abs(c.delta) - delta) <= 0.10]

        if not short_call_candidates or not short_put_candidates:
            return {"error": f"Cannot find short legs near delta {delta}"}

        short_call = self._best_contract(short_call_candidates, delta, max_spread_pct * 3)
        short_put = self._best_contract(short_put_candidates, delta, max_spread_pct * 3)

        long_call = next((c for c in exp_calls if c.strike > short_call.strike), None)
        long_put = next((c for c in reversed(exp_puts) if c.strike < short_put.strike), None)

        if not long_call or not long_put:
            return {"error": "Cannot find long wings for iron condor"}

        credit = ((short_call.bid or 0) + (short_put.bid or 0)
                  - (long_call.ask or 0) - (long_put.ask or 0))
        call_width = long_call.strike - short_call.strike
        put_width = short_put.strike - long_put.strike
        max_width = max(call_width, put_width)
        max_loss = round((max_width - credit) * 100 * quantity, 2)

        return {
            "strategy": "iron_condor",
            "symbol": symbol,
            "short_call": self._contract_dict(short_call),
            "long_call": self._contract_dict(long_call),
            "short_put": self._contract_dict(short_put),
            "long_put": self._contract_dict(long_put),
            "estimated_credit": round(credit, 2),
            "max_loss": max_loss,
            "max_profit": round(credit * 100 * quantity, 2),
            "iv_context": {
                "iv_current": iv_info.iv_current if iv_info else None,
                "iv_rank": iv_info.iv_rank if iv_info else None,
            },
            "earnings_warning": earnings_warning,
            "data_snapshot": data_snapshot,
            "dispatch_action": "iron_condor",
            "dispatch_params": {
                "symbol": symbol,
                "expiration": best_exp,
                "put_long_strike": long_put.strike,
                "put_short_strike": short_put.strike,
                "call_short_strike": short_call.strike,
                "call_long_strike": long_call.strike,
                "quantity": quantity,
            },
        }

    def _select_income_protection(self, symbol, strategy, chain, price, delta,
                                  dte_target, max_spread_pct, quantity,
                                  iv_info, earnings_warning, data_snapshot):
        """Select contract for covered_call, cash_secured_put, or protective_put."""
        if strategy == "covered_call":
            contracts = chain.calls()
            right = "C"
        else:
            contracts = chain.puts()
            right = "P"

        contracts = [c for c in contracts
                     if c.dte is not None and abs(c.dte - dte_target) <= 15]
        contracts = [c for c in contracts
                     if c.delta is not None and abs(abs(c.delta) - delta) <= 0.15]

        if not contracts:
            return {"error": f"No contracts near delta {delta} and DTE {dte_target} for {strategy}"}

        best = self._best_contract(contracts, delta, max_spread_pct)
        if not best:
            return {"error": "No contracts pass spread filter"}

        if strategy == "covered_call":
            dispatch_action = "covered_call"
            dispatch_params = {
                "symbol": symbol,
                "expiration": best.expiration,
                "strike": best.strike,
                "shares": 100 * quantity,
            }
            est_credit = (best.bid or best.mid or 0) * 100 * quantity
        elif strategy == "cash_secured_put":
            dispatch_action = "cash_secured_put"
            dispatch_params = {
                "symbol": symbol,
                "expiration": best.expiration,
                "strike": best.strike,
                "contracts": quantity,
            }
            est_credit = (best.bid or best.mid or 0) * 100 * quantity
            collateral = best.strike * 100 * quantity
            if self.gateway and self.gateway.cash_value > 0 and collateral > self.gateway.cash_value:
                return {
                    "error": f"Insufficient cash for CSP collateral: need ${collateral:.2f}, have ${self.gateway.cash_value:.2f}",
                    "selected_contract": self._contract_dict(best),
                }
        else:  # protective_put
            dispatch_action = "protective_put"
            dispatch_params = {
                "symbol": symbol,
                "expiration": best.expiration,
                "strike": best.strike,
                "shares": 100 * quantity,
            }
            est_credit = -((best.ask or best.mid or 0) * 100 * quantity)

        return {
            "strategy": strategy,
            "symbol": symbol,
            "selected_contract": self._contract_dict(best),
            "estimated_premium": round(abs(est_credit), 2),
            "premium_type": "credit" if est_credit > 0 else "debit",
            "iv_context": {
                "iv_current": iv_info.iv_current if iv_info else None,
                "iv_rank": iv_info.iv_rank if iv_info else None,
            },
            "earnings_warning": earnings_warning,
            "data_snapshot": data_snapshot,
            "dispatch_action": dispatch_action,
            "dispatch_params": dispatch_params,
        }

    def _select_straddle_strangle(self, symbol, strategy, chain, price, delta,
                                  dte_target, max_spread_pct, quantity,
                                  iv_info, earnings_warning, data_snapshot):
        """Select contracts for straddle or strangle."""
        calls = chain.calls()
        puts = chain.puts()

        calls = [c for c in calls if c.dte is not None and abs(c.dte - dte_target) <= 15]
        puts = [c for c in puts if c.dte is not None and abs(c.dte - dte_target) <= 15]

        if not calls or not puts:
            return {"error": f"Not enough contracts for {strategy}"}

        if strategy == "straddle":
            all_strikes = sorted(set(c.strike for c in calls + puts))
            atm_strike = min(all_strikes, key=lambda s: abs(s - price))

            atm_calls = [c for c in calls if c.strike == atm_strike]
            atm_puts = [c for c in puts if c.strike == atm_strike]

            if not atm_calls or not atm_puts:
                return {"error": "No ATM contracts found for straddle"}

            call = min(atm_calls, key=lambda c: abs((c.dte or dte_target) - dte_target))
            put = next((p for p in atm_puts if p.expiration == call.expiration), atm_puts[0])

            cost = ((call.ask or call.mid or 0) + (put.ask or put.mid or 0)) * 100 * quantity

            return {
                "strategy": "straddle",
                "symbol": symbol,
                "call": self._contract_dict(call),
                "put": self._contract_dict(put),
                "strike": atm_strike,
                "estimated_cost": round(cost, 2),
                "max_loss": round(cost, 2),
                "iv_context": {
                    "iv_current": iv_info.iv_current if iv_info else None,
                    "iv_rank": iv_info.iv_rank if iv_info else None,
                },
                "earnings_warning": earnings_warning,
                "data_snapshot": data_snapshot,
                "dispatch_action": "straddle",
                "dispatch_params": {
                    "symbol": symbol,
                    "expiration": call.expiration,
                    "strike": atm_strike,
                    "quantity": quantity,
                },
            }

        else:  # strangle
            call_candidates = [c for c in calls
                               if c.delta and abs(abs(c.delta) - delta) <= 0.10]
            put_candidates = [c for c in puts
                              if c.delta and abs(abs(c.delta) - delta) <= 0.10]

            if not call_candidates or not put_candidates:
                return {"error": f"Cannot find OTM legs near delta {delta} for strangle"}

            call = self._best_contract(call_candidates, delta, max_spread_pct * 2)
            put_same_exp = [p for p in put_candidates if p.expiration == call.expiration]
            put = self._best_contract(put_same_exp or put_candidates, delta, max_spread_pct * 2)

            cost = ((call.ask or call.mid or 0) + (put.ask or put.mid or 0)) * 100 * quantity

            return {
                "strategy": "strangle",
                "symbol": symbol,
                "call": self._contract_dict(call),
                "put": self._contract_dict(put),
                "estimated_cost": round(cost, 2),
                "max_loss": round(cost, 2),
                "iv_context": {
                    "iv_current": iv_info.iv_current if iv_info else None,
                    "iv_rank": iv_info.iv_rank if iv_info else None,
                },
                "earnings_warning": earnings_warning,
                "data_snapshot": data_snapshot,
                "dispatch_action": "strangle",
                "dispatch_params": {
                    "symbol": symbol,
                    "expiration": call.expiration,
                    "put_strike": put.strike,
                    "call_strike": call.strike,
                    "quantity": quantity,
                },
            }

    def _get_pending_entry_summary(self) -> list:
        """Get summary of pending entry orders (unfilled orders with no position).
        
        Helps prevent the agent from re-ordering symbols it already has
        pending orders for.
        """
        # Get open orders that don't correspond to any position (orphan entries)
        try:
            orders = self.gateway.get_cached_trades() if self.gateway else []
            positions = self.gateway.get_cached_portfolio() if self.gateway else []
            pos_symbols = {item.contract.symbol.upper() for item in positions if item.position != 0}
            entry_types = {'LMT', 'MKT', 'MIDPRICE', 'REL', 'MOC', 'MOO', 'LOC', 'LOO'}
            pending = []
            for t in orders:
                sym = t.contract.symbol.upper()
                otype = t.order.orderType
                if sym not in pos_symbols and otype in entry_types:
                    pending.append({
                        "symbol": sym,
                        "order_id": t.order.orderId,
                        "action": t.order.action,
                        "quantity": int(t.order.totalQuantity),
                        "order_type": otype,
                        "limit_price": t.order.lmtPrice if t.order.lmtPrice < 1e300 else None,
                    })
            return pending
        except Exception:
            return []

    def _order_fingerprint(self, action: str, params: dict) -> str:
        """Build a dedup fingerprint for an order action."""
        parts = [
            action,
            str(params.get("symbol", "")).upper(),
            str(params.get("side", params.get("direction", ""))).upper(),
            str(params.get("quantity", params.get("total_quantity", 0))),
            str(params.get("intent", "")),
        ]
        # For option spreads include strategy-identifying params
        for key in ("strike", "expiry", "right", "strategy",
                    "long_strike", "short_strike", "call_strike", "put_strike"):
            val = params.get(key)
            if val is not None:
                parts.append(f"{key}={val}")
        return "|".join(parts)

    def _check_idempotency(self, action: str, params: dict) -> dict | None:
        """Return rejection dict if this order was submitted within 60s, else None."""
        import time
        if action not in self._order_actions:
            return None
        fp = self._order_fingerprint(action, params)
        now = time.time()
        # Prune stale entries
        self._recent_orders = {
            k: v for k, v in self._recent_orders.items() if now - v < 60
        }
        if fp in self._recent_orders:
            elapsed = round(now - self._recent_orders[fp], 1)
            logger.warning(f"Duplicate order suppressed ({elapsed}s ago): {fp}")
            return {
                "error": "duplicate_suppressed",
                "reason": f"Identical order submitted {elapsed}s ago (60s cooldown)",
                "fingerprint": fp,
            }
        # Record this order
        self._recent_orders[fp] = now
        return None

    async def _safe_dispatch(self, action: str, params: dict) -> tuple:
        """Dispatch with exception safety. Returns (result_dict, success_bool)."""
        try:
            # Idempotency guard — block duplicate orders within 60s
            dup = self._check_idempotency(action, params)
            if dup is not None:
                return dup, False

            result = await self._dispatch(action, params)
            success = isinstance(result, dict) and "error" not in result

            if success and action in self._protective_order_actions and isinstance(result, dict):
                order_id = result.get("order_id")
                if order_id:
                    verify_result, verify_success = await self._verify_order_active(
                        int(order_id),
                        symbol=str(params.get("symbol", result.get("symbol", ""))).upper() or None,
                    )
                    if not verify_success:
                        return verify_result, False
                    result["post_submit_status"] = verify_result.get("status")

            return result, success
        except Exception as e:
            logger.error(f"Gateway dispatch failed: {action} - {e}")
            return {"error": str(e)}, False

    async def _verify_order_active(self, order_id: int, symbol: str | None = None, wait_seconds: float = 0.6) -> tuple[dict, bool]:
        """Verify newly-submitted order remains active after a short broker-processing window."""
        if not self.gateway:
            return {"verified": False, "reason": "no_gateway"}, True

        try:
            await asyncio.sleep(wait_seconds)
            orders = await self.gateway.get_open_orders()
            for order in orders:
                if int(order.get("order_id", 0)) == int(order_id):
                    status = order.get("status")
                    return {
                        "verified": True,
                        "order_id": int(order_id),
                        "status": status,
                    }, True

            return {
                "error": "order_cancelled_immediately",
                "order_id": int(order_id),
                "symbol": symbol,
                "reason": "Order missing from active open orders shortly after placement",
            }, False
        except Exception as exc:
            logger.warning(f"Post-submit verify failed for order {order_id}: {exc}")
            # Don't fail hard on verification transport issues.
            return {
                "verified": False,
                "order_id": int(order_id),
                "reason": f"verification_unavailable: {exc}",
            }, True

    async def execute(self, action: str, params: dict) -> 'ToolResult':
        """Execute a tool and return structured ToolResult."""
        try:
            # Resolve aliases before validation
            action = _ALIASES.get(action, action)

            # Basic validation: check action exists in registry
            if action not in _REGISTRY and action not in ("wait", "think", "feedback", "done"):
                err = {"error": f"Unknown action: {action}", "valid_actions": get_valid_actions()[:25]}
                return ToolResult(
                    action=action, data=err, success=False,
                    raw_json=json.dumps(err, indent=2, default=str),
                )

            # Check broker connection for order actions
            if action in _ORDER_ACTIONS and not self.gateway:
                err = {"error": f"Tool '{action}' requires broker connection"}
                return ToolResult(
                    action=action, data=err, success=False,
                    raw_json=json.dumps(err),
                )

            result = await self._dispatch(action, params)

            return ToolResult(
                action=action,
                data=result,
                success='error' not in (result if isinstance(result, dict) else {}),
                raw_json=json.dumps(result, indent=2, default=str),
            )
        except Exception as e:
            logger.error(f"Tool error: {action} - {e}")
            return ToolResult(
                action=action,
                data={"error": str(e)},
                success=False,
                raw_json=json.dumps({"error": str(e)}),
            )

    async def _dispatch(self, action: str, params: dict) -> Any:
        """Route action to handler via registry."""
        # Alias common LLM verbs to valid tools to prevent no-op failures.
        if action in ("buy", "sell"):
            symbol = params.get("symbol")
            quantity = params.get("quantity")
            if not symbol or quantity is None:
                return {"error": f"Invalid {action} action: symbol and quantity required"}
            side = "BUY" if action == "buy" else "SELL"
            # Auto-detect intent: if selling a LONG or buying a SHORT, it's an exit
            intent = params.get("intent", None)
            if intent is None:
                # Check cached portfolio synchronously for intent detection
                _portfolio = self.gateway.get_cached_portfolio() if self.gateway else []
                _pos_match = None
                for _item in _portfolio:
                    if _item.contract.symbol.upper() == symbol.upper() and _item.position != 0:
                        _pos_match = _item
                        break
                if _pos_match and side == "SELL" and _pos_match.position > 0:
                    intent = "exit"
                elif _pos_match and side == "BUY" and _pos_match.position < 0:
                    intent = "exit"
                else:
                    intent = "entry"
            elif intent in ("rotation", "close", "trim"):
                # LLM explicitly says it's closing — ensure exit-like behavior
                pass  # Keep the intent as-is but it's already in stop_skipped list
            orig = params  # Save original before rebuilding
            params = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "intent": intent,
                "execute": orig.get("execute", True),
            }
            # Pass through any agent overrides from original params
            for key in ("stop_distance_pct", "stop_type", "trail_pct",
                        "order_type", "limit_price"):
                val = orig.get(key)
                if val is not None:
                    params[key] = val
            action = "plan_order"

        # Normalize side: BUY_TO_OPEN/BUY_TO_CLOSE → BUY, SELL_TO_OPEN/SELL_TO_CLOSE → SELL
        _side_raw = params.get("side", "")
        if isinstance(_side_raw, str) and ("_TO_" in _side_raw.upper()):
            _norm = _side_raw.upper().split("_")[0]  # BUY or SELL
            params["side"] = _norm

        # STRICT cash-only guardrail at dispatch level — catches direct order
        # calls (market_order, limit_order, etc.) that bypass plan_order.
        if self.cash_only and action in self._order_actions:
            side = params.get("side", "").upper()
            symbol = (params.get("symbol") or "").upper()
            intent = params.get("intent", "entry").lower()
            cash_only_err = self._check_cash_only(side, symbol, intent)
            if cash_only_err:
                return cash_only_err

        # Normalize expiration formats (Unix timestamp / YYYY-MM-DD → YYYYMMDD)
        # so IBKR never receives raw timestamps from MarketData API.
        for _ek in ("expiration", "near_expiration", "far_expiration",
                    "old_expiration", "new_expiration"):
            _ev = params.get(_ek)
            if _ev:
                params[_ek] = _normalize_expiration(str(_ev))

        # ── OCC symbol redirect ─────────────────────────────────
        # If LLM passes an OCC options symbol (e.g. SMCI260220C00031000) to a
        # stock order tool, parse it and redirect to buy_option / close_option.
        if action in ("limit_order", "market_order", "plan_order"):
            _sym = (params.get("symbol") or "")
            _occ = _parse_occ_symbol(_sym)
            if _occ:
                side = params.get("side", "BUY").upper().replace("_TO_OPEN", "").replace("_TO_CLOSE", "")
                qty = params.get("quantity", 1)
                logger.info(f"OCC redirect: {action}({_sym}) → buy_option({_occ['underlying']}, {_occ['expiration']}, {_occ['strike']}, {_occ['right']})")
                if side == "SELL":
                    action = "close_option"
                    params = {
                        "symbol": _occ["underlying"],
                        "expiration": _occ["expiration"],
                        "strike": _occ["strike"],
                        "right": _occ["right"],
                        "quantity": int(qty),
                    }
                else:
                    action = "buy_option"
                    params = {
                        "symbol": _occ["underlying"],
                        "expiration": _occ["expiration"],
                        "strike": _occ["strike"],
                        "right": _occ["right"],
                        "quantity": int(qty),
                    }

        handler = _REGISTRY.get(action)
        if handler is None:
            valid = get_valid_actions()
            return {
                "error": f"Unknown action: {action}",
                "hint": "Use plan_order for entries/exits. Use market_order/limit_order for direct execution.",
                "valid_actions": valid,
            }
        return await handler(self, params)
