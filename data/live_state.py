"""
LiveState - Unified real-time portfolio state streaming.

This is HOME PLATE - everything the agent needs to know, updated continuously.
Injected into every agent turn so it never flies blind.

Just FACTS - no "protection" logic:
- Positions (with their associated orders)
- Orphan orders (orders for symbols with no position)
- Broker events (fills, errors, rejections)

The AGENT decides what action to take based on performance and risk.
"""

import threading
import logging
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A portfolio position with time tracking."""
    symbol: str
    quantity: float  # negative = short
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    market_price: float = 0.0
    entry_time: Optional[datetime] = None  # When position was opened
    
    # Option greeks (populated for option positions only)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None
    dte: Optional[int] = None
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_option(self) -> bool:
        """True if this position key looks like an option (SYMBOL_RIGHT_STRIKE_EXPIRY)."""
        return self.symbol.count('_') >= 3
    
    @property
    def greeks_str(self) -> str:
        """Format greeks for display. Only shown for option positions."""
        if not self.is_option or self.delta is None:
            return ""
        parts = [f"Δ{self.delta:+.2f}"]
        if self.theta is not None:
            parts.append(f"Θ{self.theta:+.2f}")
        if self.iv is not None:
            parts.append(f"IV:{self.iv:.0f}%")
        if self.dte is not None:
            parts.append(f"DTE:{self.dte}")
        return " | " + " ".join(parts)
    
    @property
    def direction(self) -> str:
        return "SHORT" if self.is_short else "LONG"
    
    @property
    def shares(self) -> int:
        """Absolute number of shares."""
        return abs(int(self.quantity))
    
    @property
    def pnl_percent(self) -> float:
        """P&L as percentage of cost basis."""
        cost_basis = abs(self.quantity * self.avg_cost)
        if cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / cost_basis) * 100
    
    @property
    def hold_time_hours(self) -> Optional[float]:
        """How long position has been held in hours."""
        if self.entry_time is None:
            return None
        now = datetime.now(timezone.utc)
        entry = self.entry_time if self.entry_time.tzinfo else self.entry_time.replace(tzinfo=timezone.utc)
        delta = now - entry
        return delta.total_seconds() / 3600
    
    @property
    def pnl_per_hour(self) -> Optional[float]:
        """P&L per hour held (absolute dollar efficiency)."""
        hours = self.hold_time_hours
        if hours is None or hours < 0.017:  # ~1 minute minimum
            return None
        return self.unrealized_pnl / hours

    @property
    def efficiency(self) -> Optional[float]:
        """Return on capital per hour (%/hr). Comparable across positions and closed trades."""
        hours = self.hold_time_hours
        if hours is None or hours < 0.017:
            return None
        return self.pnl_percent / hours
    
    def format_hold_time(self) -> str:
        """Human-readable hold time."""
        hours = self.hold_time_hours
        if hours is None:
            return "?"
        if hours < 1:
            return f"{int(hours * 60)}m"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            days = hours / 24
            return f"{days:.1f}d"


@dataclass  
class Order:
    """An open order."""
    order_id: int
    symbol: str  # underlying symbol (e.g., 'AAPL') — always the underlying, even for options
    action: str  # BUY or SELL
    quantity: float
    order_type: str  # MKT, LMT, STP, TRAIL, etc.
    status: str
    aux_price: float = 0.0  # stop price
    limit_price: float = 0.0
    sec_type: str = 'STK'  # STK, OPT, BAG — for matching orders to positions
    filled_qty: float = 0.0  # partial fill tracking
    remaining_qty: float = 0.0  # partial fill tracking
    
    def format_brief(self) -> str:
        """Brief description for display."""
        price_str = ""
        # TRAIL orders from IBKR report aux_price as float('inf') / sys.float_info.max
        # when trailing reference hasn't been established yet — suppress these.
        if self.order_type == 'TRAIL' and self.aux_price > 1e30:
            price_str = " (trailing)"  # show it's a trail, skip absurd price
        elif self.aux_price > 0 and self.aux_price < 1e30:
            price_str = f" @ ${self.aux_price:.2f}"
        elif self.limit_price > 0 and self.limit_price < 1e30:
            price_str = f" @ ${self.limit_price:.2f}"
        return f"{self.action} {int(self.quantity)} {self.order_type}{price_str}"


@dataclass
class BrokerEvent:
    """A broker event (fill, rejection, error)."""
    event_type: str  # FILLED, REJECTED, CANCELLED, ERROR
    symbol: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[int] = None
    is_critical: bool = False
    
    def format(self) -> str:
        icon = {
            "FILLED": "✅",
            "PARTIAL_FILL": "⏳",
            "REJECTED": "❌", 
            "CANCELLED": "🚫",
            "ERROR": "🔴",
            "SUBMITTED": "📤",
        }.get(self.event_type, "📌")
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {icon} {self.event_type}: {self.message}"


@dataclass
class SpreadGroup:
    """Tracks multi-leg option spreads so the agent sees them as one position."""
    group_id: str          # e.g. "AAPL_IC_20260320_1" (unique)
    underlying: str        # e.g. "AAPL"
    strategy: str          # e.g. "Iron Condor", "Bull Call Spread"
    legs: list = field(default_factory=list)  # position keys like ["AAPL_C_195_20260320", ...]
    entry_time: Optional[datetime] = None
    order_id: Optional[int] = None  # original combo order ID


class LiveState:
    """
    Real-time portfolio state that streams to the agent.
    
    Just tracks FACTS - no "protection" logic.
    Agent sees this every turn via format_for_agent().
    """
    
    # Market regime signals — fetched and cached every 120s
    _REGIME_SYMBOLS = ['SPY', 'IWM', 'GLD', 'TLT', 'HYG', 'UUP']
    _REGIME_CACHE_TTL = 120  # seconds — was 30s, widened to reduce whipsaw

    def __init__(self):
        self._lock = threading.RLock()
        
        # Live state
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[int, Order] = {}  # order_id -> Order
        self._events: List[BrokerEvent] = []
        self._max_events = 50
        
        # Spread grouping — maps position_key -> SpreadGroup
        self._spread_groups: Dict[str, SpreadGroup] = {}   # group_id -> SpreadGroup
        self._leg_to_group: Dict[str, str] = {}             # position_key -> group_id
        
        # Account
        self._cash: float = 0.0
        self._available_funds: float = 0.0  # AvailableFunds — what you can spend
        self._net_liq: float = 0.0
        self._daily_pnl: float = 0.0  # Total daily P&L (realized + unrealized)
        self._realized_pnl: float = 0.0  # Realized P&L from closed trades
        self._unrealized_pnl: float = 0.0  # Unrealized P&L from open positions
        
        # Market regime cache
        self._regime_cache: Dict[str, dict] = {}  # symbol -> {last, change_pct}
        self._regime_cache_ts: float = 0.0
        self._vix_level: Optional[float] = None
        self._last_regime: Optional[str] = None  # stability: last classified regime
        self._pending_regime: Optional[str] = None  # stability: candidate if different from _last_regime
        self._pending_regime_count: int = 0  # how many consecutive reads at pending
        
        # Closed trade journal (today's session) — reload from disk to survive restarts
        self._closed_trades_today: List[dict] = self._load_closed_trades()
        
        # Pending order symbols — grace period before orphan classification
        # Maps symbol -> expiry timestamp. Orders for these symbols are NOT
        # orphans even if no position exists yet (MKT fill lag on paper).
        self._pending_symbols: Dict[str, float] = {}

        # IBKR farm connectivity tracking (state-change dedup)
        self._farm_connected: Dict[str, bool] = {}  # farm_name -> connected?
        self._farm_events: List[dict] = []  # [{timestamp, code, message, state}]
        
        # Broker reference (set when wired)
        self._broker = None
        
        # Entry time persistence — survive restarts
        self._entry_times_file = self._get_entry_times_path()
        self._persisted_entry_times: Dict[str, datetime] = self._load_entry_times()

        # Expected entry prices — for slippage tracking.
        # Stored as FIFO queues per symbol so repeated entries don't overwrite.
        # Example: {"AAPL": deque([{"price": 201.2, "order_id": 123, "basis": "limit"}, ...])}
        self._expected_entry_prices: Dict[str, deque] = {}

        # Screen dedup across cycles — tracks symbols already surfaced
        # by prepare_session so the agent can prioritize novel candidates.
        # Capped at MAX_SCREENED to prevent unbounded growth that biases
        # late-day screens toward illiquid names.
        self.MAX_SCREENED_SYMBOLS = 200
        self._session_screened_symbols: set = set()

        # Profit protection trackers — HWM per position + session equity peak
        from data.profit_metrics import PositionHighWaterTracker, SessionEquityTracker
        self._hwm_tracker = PositionHighWaterTracker()
        self._equity_tracker = SessionEquityTracker()

    # =========================================================================
    # SLIPPAGE TRACKING
    # =========================================================================

    def set_expected_entry_price(
        self,
        symbol: str,
        price: float,
        order_id: Optional[int] = None,
        basis: str = "quote",
    ):
        """Store expected entry price for later slippage attribution.

        Uses per-symbol FIFO queues to avoid overwriting when multiple entries
        occur in the same symbol during a session.
        """
        if not price or price <= 0:
            return

        key = symbol.upper()
        with self._lock:
            queue = self._expected_entry_prices.setdefault(key, deque())
            queue.append({
                "price": float(price),
                "order_id": int(order_id) if order_id is not None else None,
                "basis": basis or "quote",
                "timestamp": datetime.now().isoformat(),
                "vix_at_entry": self._vix_level,
            })

    # =========================================================================
    # FARM CONNECTIVITY TRACKING
    # =========================================================================

    def _record_farm_event(self, code: int, message: str, connected: bool) -> None:
        """Record a farm connectivity state change with dedup."""
        # Extract farm name from message (e.g. "Market data farm connection is broken:usfarm.nj")
        farm_name = message.rsplit(":", 1)[-1].strip() if ":" in message else f"code_{code}"
        prev = self._farm_connected.get(farm_name)
        # Only record state *changes*, not repeats
        if prev == connected:
            return
        self._farm_connected[farm_name] = connected
        event = {
            "timestamp": datetime.now().isoformat(),
            "code": code,
            "farm": farm_name,
            "connected": connected,
            "message": message,
        }
        self._farm_events.append(event)
        # Cap list to prevent unbounded growth
        if len(self._farm_events) > 100:
            self._farm_events = self._farm_events[-50:]
        state = "CONNECTED" if connected else "DISCONNECTED"
        logger.info(f"Farm {farm_name}: {state} (code {code})")

    @property
    def farm_health(self) -> dict:
        """Current farm connectivity status and recent events."""
        disconnected = [f for f, c in self._farm_connected.items() if not c]
        return {
            "all_connected": len(disconnected) == 0,
            "farms": dict(self._farm_connected),
            "disconnected": disconnected,
            "recent_events": self._farm_events[-10:],
            "total_events": len(self._farm_events),
        }
    
    # =========================================================================
    # ENTRY TIME PERSISTENCE
    # =========================================================================
    
    @staticmethod
    def _get_entry_times_path():
        """Deprecated — entry_times now in-memory only."""
        return None
    
    def _load_entry_times(self) -> Dict[str, datetime]:
        """No persistence — entry times are in-memory only."""
        return {}
    
    def _save_entry_times(self):
        """No-op — in-memory only, no persistence."""
        pass

    @staticmethod
    def _load_closed_trades() -> List[dict]:
        """No persistence — closed trades are in-memory only."""
        return []

    # =========================================================================
    # STATE REFRESH (solves race between tool call and event delivery)
    # =========================================================================
    
    async def refresh_after_order(self, timeout: float = 3.0):
        """
        Re-poll broker state after an order is placed.
        
        The agent reads LiveState immediately after a tool call, but ib_insync
        event callbacks may not have arrived yet. This method bridges the gap
        by polling portfolio + orders directly and updating LiveState.
        
        Called by tools_executor after any order-placing action.
        """
        import asyncio
        
        if not self._broker:
            return
        
        if not self._broker.is_connected:
            return
        
        # Wait for IBKR to process the order and send callbacks
        await asyncio.sleep(1.0)
        
        # Force-request fresh account data (not just cached)
        try:
            self._broker.ib.reqAccountUpdates(subscribe=True, account=self._broker.account_id)
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.debug(f"reqAccountUpdates failed: {e}")
        
        # Re-poll account values FIRST (cash/available_funds update after fills)
        try:
            for av in self._broker.get_cached_account_values():
                self.on_account_value(av)
        except Exception as e:
            logger.debug(f"Refresh account failed: {e}")
        
        # Re-poll portfolio (positions + P&L)
        try:
            for item in self._broker.get_cached_portfolio():
                self.on_portfolio_update(item)
        except Exception as e:
            logger.debug(f"Refresh portfolio failed: {e}")
        
        # Force-request fresh open orders to clear cancelled/filled
        try:
            await self._broker.ib.reqAllOpenOrdersAsync()
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.debug(f"reqAllOpenOrders failed: {e}")
        
        # Sync orders: remove any orders from tracking that IBKR no longer reports
        try:
            active_statuses = {'PreSubmitted', 'Submitted', 'PendingSubmit', 'PendingCancel'}
            ibkr_active_ids = set()
            for trade in self._broker.ib.openTrades():
                if trade.orderStatus.status in active_statuses:
                    ibkr_active_ids.add(trade.order.orderId)
                self.on_order_status(trade)
            
            # Remove stale orders from our tracking that IBKR no longer has
            with self._lock:
                stale_ids = [oid for oid in self._orders if oid not in ibkr_active_ids]
                for oid in stale_ids:
                    logger.info(f"Removing stale order {oid} (no longer in IBKR open orders)")
                    del self._orders[oid]
        except Exception as e:
            logger.debug(f"Refresh orders failed: {e}")
    
    async def refresh_option_greeks(self):
        """
        Refresh greeks for all option positions via the market data provider.
        
        Called periodically (e.g., every cycle) to keep greeks fresh.
        Parses the composite key (SYMBOL_RIGHT_STRIKE_EXPIRY) to look up contracts.
        Uses data_provider (market data app) instead of direct IBKR calls.
        """
        from datetime import datetime as dt
        
        try:
            from data.data_provider import get_data_provider
            dp = get_data_provider()
        except Exception:
            return
        
        with self._lock:
            option_positions = [(k, p) for k, p in self._positions.items() if p.is_option]
        
        if not option_positions:
            return
        
        for key, pos in option_positions:
            try:
                # Parse composite key: SYMBOL_RIGHT_STRIKE_EXPIRY
                parts = key.split('_')
                if len(parts) < 4:
                    continue
                underlying = parts[0]
                right = parts[1]  # C or P
                strike = float(parts[2])
                expiry = parts[3]  # YYYYMMDD
                
                # Calculate DTE
                dte = None
                try:
                    exp_date = dt.strptime(expiry, '%Y%m%d').date()
                    dte = (exp_date - dt.now().date()).days
                except (ValueError, TypeError):
                    pass
                
                # Format expiration for data_provider: YYYY-MM-DD
                exp_formatted = None
                try:
                    exp_formatted = dt.strptime(expiry, '%Y%m%d').strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    pass
                
                if exp_formatted:
                    side = 'call' if right == 'C' else 'put'
                    chain = dp.get_option_chain(
                        underlying,
                        expiration=exp_formatted,
                        side=side,
                        strike_range=(strike - 0.01, strike + 0.01)
                    )
                    if chain and chain.contracts:
                        # Find our specific contract
                        match = None
                        for c in chain.contracts:
                            if abs(c.strike - strike) < 0.01:
                                match = c
                                break
                        if match:
                            with self._lock:
                                if key in self._positions:
                                    p = self._positions[key]
                                    p.delta = match.delta
                                    p.gamma = match.gamma
                                    p.theta = match.theta
                                    p.vega = match.vega
                                    p.iv = round(match.iv * 100, 1) if match.iv and match.iv < 10 else match.iv
                                    p.dte = dte
                                    continue  # Done for this position
                
                # At least set DTE even without full greeks
                if dte is not None:
                    with self._lock:
                        if key in self._positions:
                            self._positions[key].dte = dte
                            
            except Exception as e:
                logger.debug(f"Greek refresh failed for {key}: {e}")
    
    # =========================================================================
    # STATE UPDATES
    # =========================================================================
    
    def update_position(self, symbol: str, quantity: float, avg_cost: float,
                       market_value: float = 0.0, unrealized_pnl: float = 0.0,
                       market_price: float = 0.0, entry_time: Optional[datetime] = None):
        """Update a position. Preserves entry_time and greeks if not provided.
        When quantity goes to 0, captures the closed trade in the journal."""
        with self._lock:
            if quantity == 0:
                if symbol in self._positions:
                    # === CAPTURE CLOSED TRADE before deleting ===
                    closed_pos = self._positions[symbol]
                    self._record_closed_trade(closed_pos, market_price)
                    del self._positions[symbol]
                    # Clean up persisted entry time
                    self._persisted_entry_times.pop(symbol, None)
                    self._save_entry_times()
            else:
                # Position arrived — clear pending grace if set
                underlying = self._underlying_from_key(symbol)
                self._pending_symbols.pop(underlying, None)
                
                # Preserve existing entry_time and greeks if not provided
                existing = self._positions.get(symbol)
                existing_entry_time = existing.entry_time if existing and entry_time is None else None
                if entry_time is None and existing_entry_time is None:
                    # Check persisted entry times (survives restarts)
                    persisted = self._persisted_entry_times.get(symbol)
                    if persisted:
                        entry_time = persisted
                        logger.debug(f"Restored persisted entry_time for {symbol}: {persisted}")
                    else:
                        # Fallback to first-seen timestamp so hold time is never unknown.
                        entry_time = datetime.now(timezone.utc)
                
                pos = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    market_price=market_price,
                    entry_time=entry_time or existing_entry_time
                )
                # Preserve greeks from previous update
                if existing and existing.delta is not None:
                    pos.delta = existing.delta
                    pos.gamma = existing.gamma
                    pos.theta = existing.theta
                    pos.vega = existing.vega
                    pos.iv = existing.iv
                    pos.dte = existing.dte
                
                self._positions[symbol] = pos
                
                # Update profit-protection high-water mark for this position
                try:
                    self._hwm_tracker.update(
                        symbol, unrealized_pnl, pos.pnl_percent,
                        market_price, avg_cost,
                    )
                except Exception:
                    pass
                
                # Persist entry times after every position update
                self._save_entry_times()
    
    def update_order(self, order_id: int, symbol: str, action: str, 
                    quantity: float, order_type: str, status: str,
                    aux_price: float = 0.0, limit_price: float = 0.0,
                    sec_type: str = 'STK', filled_qty: float = 0.0,
                    remaining_qty: float = 0.0):
        """Update an order. Tracks all statuses including PreSubmitted, Submitted, etc."""
        with self._lock:
            # Terminal statuses - order is done, remove from tracking
            if status in ('Filled', 'Cancelled', 'ApiCancelled', 'Inactive'):
                if order_id in self._orders:
                    del self._orders[order_id]
            else:
                # Track all other statuses: PreSubmitted, Submitted, PendingSubmit, PendingCancel
                self._orders[order_id] = Order(
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    order_type=order_type,
                    status=status,
                    aux_price=aux_price,
                    limit_price=limit_price,
                    sec_type=sec_type,
                    filled_qty=filled_qty,
                    remaining_qty=remaining_qty
                )
    
    def update_account(self, cash: float, net_liq: float):
        """Update account values."""
        with self._lock:
            self._cash = cash
            self._net_liq = net_liq
    
    def push_event(self, event_type: str, symbol: str, message: str,
                  order_id: Optional[int] = None, is_critical: bool = False):
        """Push a broker event."""
        event = BrokerEvent(
            event_type=event_type,
            symbol=symbol,
            message=message,
            order_id=order_id,
            is_critical=is_critical
        )
        
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
        
        if is_critical:
            logger.warning(f"[CRITICAL EVENT] {event.format()}")
    
    def pop_events(self) -> List[BrokerEvent]:
        """Get and clear pending events."""
        with self._lock:
            events = self._events.copy()
            self._events.clear()
            return events
    
    # =========================================================================
    # CLOSED TRADE JOURNAL
    # =========================================================================
    
    def _record_closed_trade(self, pos: 'Position', exit_price: float = 0.0):
        """Record a closed trade to memory. Called inside _lock."""
        import json
        
        entry_price = pos.avg_cost
        if exit_price <= 0:
            exit_price = pos.market_price or entry_price
        
        qty = abs(pos.quantity)
        direction = pos.direction
        
        # Calculate P&L
        if direction == "LONG":
            pnl = (exit_price - entry_price) * qty
        else:  # SHORT
            pnl = (entry_price - exit_price) * qty
        
        pnl_pct = ((pnl / (entry_price * qty)) * 100) if entry_price * qty > 0 else 0.0
        
        hold_hours = pos.hold_time_hours
        
        # Efficiency metrics: persist at close time for historical comparison
        eff_pnl_hr = round(pnl / hold_hours, 2) if hold_hours and hold_hours > 0.017 else None
        eff_pct_hr = round(pnl_pct / hold_hours, 2) if hold_hours and hold_hours > 0.017 else None
        
        # Slippage: compare actual entry fill to expected entry price at order time.
        # Discard stale entries (>4h old) to prevent orphan attribution when a
        # position is held overnight or closed manually.
        expected_entry = None
        slippage_basis = None
        vix_at_entry = None
        queue = self._expected_entry_prices.get(pos.symbol)
        if queue:
            # Drain expired entries first
            now = datetime.now()
            while queue:
                front = queue[0]
                try:
                    ts = datetime.fromisoformat(front.get("timestamp", ""))
                    if (now - ts).total_seconds() > 4 * 3600:
                        queue.popleft()  # expired
                        continue
                except (ValueError, TypeError):
                    pass
                break  # valid entry found
            if queue:
                item = queue.popleft()
                expected_entry = item.get("price")
                slippage_basis = item.get("basis")
                vix_at_entry = item.get("vix_at_entry")
            if not queue:
                self._expected_entry_prices.pop(pos.symbol, None)

        slippage_pct = None
        if expected_entry and expected_entry > 0 and entry_price > 0:
            if direction == "LONG":
                # Paid more than expected → positive slippage (bad)
                slippage_pct = round((entry_price - expected_entry) / expected_entry * 100, 3)
            else:
                # Sold for less than expected → positive slippage (bad)
                slippage_pct = round((expected_entry - entry_price) / expected_entry * 100, 3)
        
        # Capture high-water mark for profit retention tracking
        hwm_record = self._hwm_tracker.pop_peak(pos.symbol)
        peak_unrealized_pnl = None
        profit_retention = None
        if hwm_record and hwm_record.get("peak_pnl") is not None:
            peak_unrealized_pnl = round(hwm_record["peak_pnl"], 2)
            if pnl > 0 and peak_unrealized_pnl > 0:
                profit_retention = round(pnl / peak_unrealized_pnl, 2)
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": pos.symbol,
            "direction": direction,
            "quantity": qty,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "hold_hours": round(hold_hours, 2) if hold_hours else None,
            "pnl_per_hour": eff_pnl_hr,
            "efficiency": eff_pct_hr,
            "expected_entry_price": round(expected_entry, 4) if expected_entry else None,
            "slippage_basis": slippage_basis,
            "slippage_pct": slippage_pct,
            "is_option": pos.is_option,
            "vix_at_entry": vix_at_entry,
            "peak_unrealized_pnl": peak_unrealized_pnl,
            "profit_retention": profit_retention,
        }
        
        # In-memory for this session
        self._closed_trades_today.append(record)
        
        win_loss = "WIN" if pnl > 0 else "LOSS"
        hold_str = f"{hold_hours:.1f}h" if hold_hours else "?"
        log_msg = (
            f"CLOSED {win_loss}: {direction} {qty} {pos.symbol} "
            f"entry=${entry_price:.2f} exit=${exit_price:.2f} "
            f"P&L=${pnl:+.2f} ({pnl_pct:+.1f}%) hold={hold_str}"
        )
        logger.info(log_msg)

        # Write to agent memory (observation log)
        try:
            from core.agent import append_memory
            emoji = "WIN" if pnl > 0 else "LOSS"
            eff_str = f" eff={eff_pct_hr:+.1f}%/hr" if eff_pct_hr is not None else ""
            append_memory(
                f"{emoji} TRADE CLOSED {win_loss}: {direction} {qty} {pos.symbol} "
                f"entry=${entry_price:.2f} exit=${exit_price:.2f} "
                f"P&L=${pnl:+.2f} ({pnl_pct:+.1f}%) hold={hold_str}{eff_str}"
            )
            # Trade already persisted to trades.jsonl by append_trade() above
        except Exception:
            pass  # Don't crash on memory write failure
    
    def _format_closed_trades_summary(self) -> List[str]:
        """Format today's closed trades summary for agent context.
        
        Shows aggregate stats PLUS the last 5 individual trades so the
        agent gets immediate feedback on what just worked/failed.
        """
        if not self._closed_trades_today:
            return []
        
        trades = self._closed_trades_today
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        flat = [t for t in trades if t['pnl'] == 0]
        total_pnl = sum(t['pnl'] for t in trades)
        
        avg_win = (sum(t['pnl'] for t in wins) / len(wins)) if wins else 0
        avg_loss = (sum(t['pnl'] for t in losses) / len(losses)) if losses else 0
        # Expectancy = avg_win * win_rate - avg_loss * loss_rate
        win_rate_frac = len(wins) / len(trades) if trades else 0
        loss_rate_frac = 1 - win_rate_frac
        expectancy = (avg_win * win_rate_frac + avg_loss * loss_rate_frac) if trades else 0
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and wins else 0
        
        lines = []
        sign = '+' if total_pnl >= 0 else ''
        lines.append("=== TODAY'S CLOSED TRADES ===")
        lines.append(
            f"TOTALS: {len(wins)}W/{len(losses)}L/{len(flat)}F | "
            f"Net {sign}${total_pnl:,.2f} | "
            f"Expectancy ${expectancy:+,.2f}/trade | "
            f"W:L ratio {wl_ratio:.1f}:1 | PF {profit_factor:.2f}"
        )
        
        # Show last 5 trades so the agent sees what just happened
        recent = trades[-5:]
        if recent:
            lines.append("RECENT:")
            for t in reversed(recent):
                tag = 'WIN' if t['pnl'] > 0 else ('FLAT' if t['pnl'] == 0 else 'LOSS')
                hold = f"{t.get('hold_hours', 0) or 0:.1f}h" if t.get('hold_hours') is not None else '?'
                eff = t.get('efficiency')
                eff_str = f" | {eff:+.1f}%/hr" if eff is not None else ""
                slippage_str = ""
                if t.get('slippage_pct') is not None:
                    slippage_str = f" | slip {t['slippage_pct']:+.2f}%"
                lines.append(
                    f"  [{tag}] {t['direction']} {t.get('quantity', '?')} {t['symbol']} "
                    f"${t.get('entry_price', 0):.2f}→${t.get('exit_price', 0):.2f} "
                    f"P&L ${t['pnl']:+.2f} ({t.get('pnl_pct', 0):+.1f}%) "
                    f"hold={hold}{eff_str}{slippage_str}"
                )
        
        return lines
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    @staticmethod
    def _underlying_from_key(symbol_key: str) -> str:
        """Extract underlying symbol from position key.
        'AAPL_C_150_20260320' -> 'AAPL', 'AAPL' -> 'AAPL'
        """
        return symbol_key.split('_')[0] if '_' in symbol_key else symbol_key

    def get_orders_for(self, symbol_key: str) -> List[Order]:
        """Get all open orders for a position (handles options composite keys).
        
        For stock positions (key='AAPL'): matches orders with symbol='AAPL'
        For option positions (key='AAPL_C_150_20260320'): matches orders with
            symbol='AAPL' that are OPT or BAG type orders.
        """
        with self._lock:
            underlying = self._underlying_from_key(symbol_key)
            is_option_position = '_' in symbol_key
            
            if is_option_position:
                # Option position: match orders on same underlying that are options/combos
                return [o for o in self._orders.values() 
                        if o.symbol == underlying and o.sec_type in ('OPT', 'BAG')]
            else:
                # Stock position: match stock orders only
                return [o for o in self._orders.values() 
                        if o.symbol == underlying and o.sec_type == 'STK']
    
    def register_pending_order(self, symbol: str, ttl: float = 90.0):
        """Mark a symbol as having a recently-placed entry order.
        
        Orders for this symbol will NOT be classified as orphans
        until the TTL expires. This prevents the race condition where
        a MKT entry order + its stop are both flagged as dangerous
        orphans before the fill arrives and creates the position.
        
        Args:
            symbol: Underlying symbol (e.g. 'AAOI')
            ttl: Grace period in seconds (default 90s)
        """
        with self._lock:
            self._pending_symbols[symbol.upper()] = time.time() + ttl
            logger.info(f"Registered pending order for {symbol} (grace {ttl}s)")
    
    def get_pending_symbols(self) -> set:
        """Get symbols currently in the pending grace period."""
        now = time.time()
        with self._lock:
            # Clean expired
            expired = [s for s, exp in self._pending_symbols.items() if now >= exp]
            for s in expired:
                del self._pending_symbols[s]
            return set(self._pending_symbols.keys())
    
    # =========================================================================
    # SPREAD GROUPING — track multi-leg option positions as unified spreads
    # =========================================================================

    def register_spread(self, underlying: str, strategy: str,
                        leg_keys: list, order_id: int = None):
        """Register a multi-leg option spread so legs display as one position.

        Called by tools_options handlers after a successful spread placement.
        Leg keys use the position key format: SYMBOL_RIGHT_STRIKE_EXPIRY
        (e.g. ['AAPL_C_195_20260320', 'AAPL_C_200_20260320']).

        Args:
            underlying: e.g. 'AAPL'
            strategy: e.g. 'Bull Call Spread', 'Iron Condor'
            leg_keys: list of position keys for each leg
            order_id: original combo order ID (optional)
        """
        with self._lock:
            # Generate a unique group ID
            existing_count = sum(
                1 for g in self._spread_groups.values()
                if g.underlying == underlying
            )
            group_id = f"{underlying}_{strategy.replace(' ', '_')}_{existing_count + 1}"

            group = SpreadGroup(
                group_id=group_id,
                underlying=underlying,
                strategy=strategy,
                legs=list(leg_keys),
                entry_time=datetime.now(timezone.utc),
                order_id=order_id,
            )
            self._spread_groups[group_id] = group
            for key in leg_keys:
                self._leg_to_group[key] = group_id

            logger.info(f"Registered spread: {group_id} ({strategy}) legs={leg_keys}")

    def get_spread_for_leg(self, position_key: str) -> Optional[SpreadGroup]:
        """Return the SpreadGroup a position belongs to, or None."""
        with self._lock:
            gid = self._leg_to_group.get(position_key)
            if gid:
                return self._spread_groups.get(gid)
            return None

    def is_spread_leg(self, position_key: str) -> bool:
        """Check if a position is part of a registered spread."""
        return position_key in self._leg_to_group

    def cleanup_closed_spreads(self):
        """Remove spread groups whose legs are all closed (no longer in positions)."""
        with self._lock:
            to_remove = []
            for gid, group in self._spread_groups.items():
                alive = [k for k in group.legs if k in self._positions]
                if not alive:
                    to_remove.append(gid)
            for gid in to_remove:
                group = self._spread_groups.pop(gid)
                for key in group.legs:
                    self._leg_to_group.pop(key, None)
                logger.info(f"Cleaned up closed spread: {gid}")
    
    def get_orphan_orders(self) -> List[Order]:
        """Get orders for symbols that have NO position (orphans).
        
        Respects the pending-order grace period: symbols with a recently
        placed entry order (via plan_order) are NOT considered orphans,
        because the fill hasn't arrived yet to create the position.
        """
        now = time.time()
        with self._lock:
            # Clean expired pending symbols
            expired = [s for s, exp in self._pending_symbols.items() if now >= exp]
            for s in expired:
                del self._pending_symbols[s]
            
            # Build set of symbols that should NOT be orphans
            position_underlyings = set(
                self._underlying_from_key(key) for key in self._positions.keys()
            )
            pending = set(self._pending_symbols.keys())
            known_symbols = position_underlyings | pending
            
            return [o for o in self._orders.values() 
                    if o.symbol not in known_symbols]
    
    # =========================================================================
    # MARKET REGIME
    # =========================================================================
    
    def _refresh_regime_cache(self):
        """Fetch SPY, IWM, GLD, TLT quotes + VIX level. Cached for 30s."""
        now = time.time()
        if now - self._regime_cache_ts < self._REGIME_CACHE_TTL and self._regime_cache:
            return  # cache still fresh
        try:
            from data.data_provider import get_data_provider
            dp = get_data_provider()
            quotes = dp.get_quotes_bulk(self._REGIME_SYMBOLS)
            for sym, q in quotes.items():
                if q and q.last:
                    self._regime_cache[sym] = {
                        'last': q.last,
                        'change_pct': q.change_pct or 0.0,
                    }
            # VIX: MarketData API always 404s for VIX — go straight to yfinance
            # to avoid wasting 5-7s on failed API calls every refresh.
            try:
                import yfinance as yf
                vix_ticker = yf.Ticker('^VIX')
                vix_data = vix_ticker.history(period='1d')
                if not vix_data.empty:
                    self._vix_level = float(vix_data['Close'].iloc[-1])
            except Exception:
                pass  # VIX is informational — missing is acceptable
            self._regime_cache_ts = now
        except Exception as e:
            logger.debug(f"Regime cache refresh failed: {e}")
    
    def _classify_regime(self) -> tuple:
        """
        Derive market regime from 7 signals (SPY, IWM, GLD, TLT, HYG, UUP, VIX).
        Returns (regime_label, emoji, one_line_interpretation).

        Improvements over v1:
        - Added HYG (credit risk appetite) and UUP (dollar strength)
        - Widened thresholds to 0.4% (was 0.2%) to reduce whipsaw
        - Added regime stability: must confirm for 2 consecutive reads before flipping
        - Actionable guidance is cash-only aware (puts/hedges, not shorts)
        """
        spy = self._regime_cache.get('SPY', {})
        iwm = self._regime_cache.get('IWM', {})
        gld = self._regime_cache.get('GLD', {})
        tlt = self._regime_cache.get('TLT', {})
        hyg = self._regime_cache.get('HYG', {})
        uup = self._regime_cache.get('UUP', {})
        vix = self._vix_level

        def _pct(d):
            v = d.get('change_pct', 0) or 0
            return v * 100 if abs(v) < 1 and v != 0 else v

        spy_chg = _pct(spy)
        iwm_chg = _pct(iwm)
        gld_chg = _pct(gld)
        tlt_chg = _pct(tlt)
        hyg_chg = _pct(hyg)
        uup_chg = _pct(uup)

        # Thresholds widened from 0.2% to 0.4% to reduce intraday noise
        spy_up = spy_chg > 0.4
        spy_dn = spy_chg < -0.4
        iwm_up = iwm_chg > 0.4
        iwm_dn = iwm_chg < -0.4
        gld_up = gld_chg > 0.4
        tlt_up = tlt_chg > 0.3
        hyg_dn = hyg_chg < -0.2  # credit stress
        uup_up = uup_chg > 0.3  # dollar strengthening
        vix_low = vix is not None and vix < 16
        vix_elevated = vix is not None and vix > 22
        vix_high = vix is not None and vix > 30

        # Composite signals
        credit_stress = hyg_dn and tlt_up  # bonds bid, credit selling = risk off
        dollar_surge = uup_up and spy_dn  # strong dollar crushing equities
        broad_weakness = spy_dn and iwm_dn
        broad_strength = spy_up and iwm_up

        # Decision tree — ordered by severity
        if broad_weakness and vix_high:
            raw = ("LIQUIDATION", "🔴",
                   "Broad selloff + VIX >30 — raise cash, buy protective puts on remaining longs, no new entries")
        elif broad_weakness and credit_stress:
            raw = ("CREDIT STRESS", "🔴",
                   "Equities and HY bonds selling together — systemic risk rising, hedge with puts or stand aside")
        elif spy_dn and gld_up and tlt_up and vix_elevated:
            raw = ("RISK-OFF", "🟠",
                   "Flight to safety — gold/bonds bid, tighten stops, consider long puts on weak holdings")
        elif dollar_surge and vix_elevated:
            raw = ("DOLLAR SQUEEZE", "🟠",
                   "Strong dollar crushing risk assets — favor domestic/defensive names, avoid EM-exposed")
        elif broad_weakness:
            raw = ("SELLING", "🟠",
                   "Equities weak across board — tighten stops, screen for relative strength longs or buy puts for protection")
        elif spy_up and gld_up and tlt_up and vix_elevated:
            raw = ("FRAGILE RALLY", "🟡",
                   "Equities up but hedges also bid — size smaller, use limit/adaptive orders, favor defined-risk spreads")
        elif spy_up and iwm_dn:
            raw = ("NARROW RALLY", "🟡",
                   "Mega-cap only, breadth weak — stick to large-cap leaders, avoid small caps, use limit entries")
        elif not spy_up and not spy_dn and gld_up and vix_elevated:
            raw = ("INFLATION FEAR", "🟡",
                   "Gold bid, equities stalling — favor commodity/value names, avoid growth/duration-sensitive")
        elif broad_strength and vix_low:
            raw = ("RISK-ON", "🟢",
                   "Broad rally, low fear, credit healthy — deploy capital aggressively, momentum/trend-following")
        elif broad_strength:
            raw = ("RISK-ON", "🟢",
                   "Broad rally with participation — favorable for entries, use adaptive/midprice orders")
        elif spy_up:
            raw = ("CAUTIOUS BULL", "🟢",
                   "Equities up, mixed signals — normal sizing, standard stops")
        elif spy_dn:
            raw = ("CAUTIOUS BEAR", "🟠",
                   "Equities soft — reduce exposure, tighten stops, consider protective puts on longs")
        else:
            raw = ("NEUTRAL", "⚪",
                   "No strong directional signal — trade setups not direction, use limit orders for entries")

        # Regime stability: require 2 consecutive identical reads before flipping
        candidate = raw[0]
        if self._last_regime is None:
            self._last_regime = candidate
            return raw
        if candidate == self._last_regime:
            self._pending_regime = None
            self._pending_regime_count = 0
            return raw
        # Different from current regime — count confirmations
        if candidate == self._pending_regime:
            self._pending_regime_count += 1
        else:
            self._pending_regime = candidate
            self._pending_regime_count = 1
        if self._pending_regime_count >= 2:
            # Confirmed flip
            self._last_regime = candidate
            self._pending_regime = None
            self._pending_regime_count = 0
            return raw
        # Not yet confirmed — keep old regime
        return self._classify_regime_by_label(self._last_regime)

    def _classify_regime_by_label(self, label: str) -> tuple:
        """Return (label, emoji, interpretation) for a known regime label."""
        _REGIME_META = {
            "LIQUIDATION": ("🔴", "Broad selloff + VIX >30 — raise cash, buy protective puts, no new entries"),
            "CREDIT STRESS": ("🔴", "Equities and HY bonds selling together — hedge with puts or stand aside"),
            "RISK-OFF": ("🟠", "Flight to safety — gold/bonds bid, tighten stops, consider long puts"),
            "DOLLAR SQUEEZE": ("🟠", "Strong dollar crushing risk — favor domestic/defensive, avoid EM-exposed"),
            "SELLING": ("🟠", "Equities weak — tighten stops, screen for relative strength or buy puts"),
            "FRAGILE RALLY": ("🟡", "Equities up but hedges bid — size smaller, use limit/adaptive orders"),
            "NARROW RALLY": ("🟡", "Mega-cap only, breadth weak — stick to leaders, use limit entries"),
            "INFLATION FEAR": ("🟡", "Gold bid, equities stalling — favor commodity/value"),
            "RISK-ON": ("🟢", "Broad rally — deploy capital, momentum works"),
            "CAUTIOUS BULL": ("🟢", "Equities up, mixed signals — normal sizing"),
            "CAUTIOUS BEAR": ("🟠", "Equities soft — reduce exposure, tighten stops, consider puts"),
            "NEUTRAL": ("⚪", "No strong signal — trade setups not direction"),
        }
        emoji, interp = _REGIME_META.get(label, ("⚪", "Unknown regime"))
        return (label, emoji, interp)
    
    def _format_market_regime(self) -> List[str]:
        """Format market regime section for agent context."""
        self._refresh_regime_cache()
        
        if not self._regime_cache:
            return []
        
        regime, emoji, interpretation = self._classify_regime()
        
        lines = []
        lines.append(f"=== MARKET REGIME: {regime} {emoji} ===")
        
        # Signal line: SPY $502.31 (+0.8%) | IWM $198.40 (-0.3%) | ...
        parts = []
        for sym in self._REGIME_SYMBOLS:
            data = self._regime_cache.get(sym, {})
            price = data.get('last')
            chg = data.get('change_pct', 0) or 0
            # Normalize to percentage display
            if abs(chg) < 1 and chg != 0:
                chg *= 100
            if price:
                parts.append(f"{sym} ${price:,.2f} ({chg:+.1f}%)")
        if self._vix_level:
            parts.append(f"VIX {self._vix_level:.1f}")
        lines.append(" | ".join(parts))
        lines.append(interpretation)

        # Regime-aware order type hint for the agent
        order_hints = {
            "LIQUIDATION": "Order guidance: NO new entries. Buy protective puts on longs. Use market orders to exit losers.",
            "CREDIT STRESS": "Order guidance: Buy puts for protection. Use market orders to reduce. Stand aside on new entries.",
            "RISK-OFF": "Order guidance: Use limit orders for defensives (utilities/staples). Buy puts on vulnerable longs. Avoid market orders for entries.",
            "DOLLAR SQUEEZE": "Order guidance: Limit orders on domestic/defensive names. Avoid chasing. Consider protective puts.",
            "SELLING": "Order guidance: Use limit orders at support for relative-strength names. Buy puts on weak holdings. Use adaptive_algo for exits.",
            "FRAGILE RALLY": "Order guidance: Use limit/adaptive_algo/midprice for entries. Consider vertical spreads over naked stock.",
            "NARROW RALLY": "Order guidance: Limit orders on large-cap leaders only. Use adaptive_algo. Avoid market orders on small-caps.",
            "INFLATION FEAR": "Order guidance: Limit orders on commodity/value names. Consider call spreads for defined risk.",
            "RISK-ON": "Order guidance: Use adaptive_algo or midprice for best fills. Brackets for entries with auto-stops. Market orders acceptable for momentum.",
            "CAUTIOUS BULL": "Order guidance: Limit or adaptive orders. Standard bracket entries with stops.",
            "CAUTIOUS BEAR": "Order guidance: Limit orders only for new longs. Tighten existing stops. Consider protective puts.",
            "NEUTRAL": "Order guidance: Limit orders at support/resistance levels. No market orders for entries. Use adaptive_algo for size.",
        }
        hint = order_hints.get(regime, "")
        if hint:
            lines.append(hint)
        lines.append("")
        
        return lines

    def _format_position_heat(self) -> List[str]:
        """Format position heat: how many moving with/against, sector concentration."""
        positions = list(self._positions.values())
        if not positions:
            return []

        winners = []  # positions with positive unrealized P&L
        losers = []   # positions with negative unrealized P&L
        flat = []     # ~breakeven

        for p in positions:
            pnl_pct = p.pnl_percent
            if pnl_pct is not None and pnl_pct > 0.1:
                winners.append(p)
            elif pnl_pct is not None and pnl_pct < -0.1:
                losers.append(p)
            else:
                flat.append(p)

        total = len(positions)
        lines = []
        lines.append("=== POSITION HEAT ===")

        # Directional heat
        w_pct = (len(winners) / total * 100) if total else 0
        l_pct = (len(losers) / total * 100) if total else 0
        total_winner_pnl = sum(p.unrealized_pnl for p in winners)
        total_loser_pnl = sum(p.unrealized_pnl for p in losers)

        if len(losers) >= len(winners) * 2 and total >= 3:
            heat_emoji = "🔥"
            heat_label = "HIGH HEAT"
        elif len(losers) > len(winners) and total >= 3:
            heat_emoji = "⚠️"
            heat_label = "WARM"
        elif len(winners) >= len(losers) * 2 and total >= 3:
            heat_emoji = "❄️"
            heat_label = "COOL — portfolio winning broadly"
        else:
            heat_emoji = "➖"
            heat_label = "MIXED"

        lines.append(
            f"{heat_emoji} {heat_label}: "
            f"{len(winners)} winning (+${total_winner_pnl:,.2f}) | "
            f"{len(losers)} losing (${total_loser_pnl:,.2f}) | "
            f"{len(flat)} flat"
        )

        # Concentration: find if multiple positions share same underlying
        underlying_counts: Dict[str, List[str]] = {}
        for p in positions:
            # Extract underlying from option keys like AAPL_C_200_20260320
            underlying = p.symbol.split('_')[0] if '_' in p.symbol else p.symbol
            underlying_counts.setdefault(underlying, []).append(p.symbol)

        concentrated = {k: v for k, v in underlying_counts.items() if len(v) > 1}
        if concentrated:
            conc_parts = [f"{k}({len(v)}x)" for k, v in sorted(concentrated.items(), key=lambda x: -len(x[1]))]
            lines.append(f"CONCENTRATION: {', '.join(conc_parts)}")

        # Correlated loss warning: if 3+ losers, flag it
        if len(losers) >= 3:
            worst = sorted(losers, key=lambda p: p.unrealized_pnl)[:3]
            worst_str = ", ".join(f"{p.symbol} ${p.unrealized_pnl:+.2f}" for p in worst)
            lines.append(f"⚠️ {len(losers)} positions red — worst: {worst_str}")

        lines.append("")
        return lines

    def _format_holding_risk_context(self, session_info: Dict[str, Any]) -> List[str]:
        """Format time-of-day and holding-risk context for the agent.

        Focus:
        - Near-close behavior (when overnight risk is about to begin)
        - Outside-regular-hours behavior (reduced liquidity / deferred stops)
        - Weekend / long-weekend gap risk awareness
        """
        lines: List[str] = []
        total_positions = len(self._positions)
        if total_positions <= 0:
            return lines

        session = str(session_info.get('session', 'unknown'))
        minutes_to_close = session_info.get('minutes_to_close')

        # Determine days until next regular session to detect weekend/holiday gaps.
        days_to_next_session: Optional[int] = None
        next_session_date = None
        try:
            from data.market_hours import get_market_hours_provider
            mh = get_market_hours_provider()
            now_et = mh._to_eastern(datetime.now(timezone.utc))
            if mh.calendar is not None:
                next_session = mh.calendar.next_session(now_et.date())
                if hasattr(next_session, 'date'):
                    next_session_date = next_session.date()
                else:
                    next_session_date = datetime.fromisoformat(str(next_session)).date()
                days_to_next_session = (next_session_date - now_et.date()).days
        except Exception:
            pass

        lines.append("=== HOLDING RISK CONTEXT ===")
        lines.append(f"Open positions: {total_positions} — overnight gap risk applies to all holds.")

        if session == 'regular' and isinstance(minutes_to_close, int):
            if minutes_to_close <= 30:
                lines.append("🔔 <30m to close: avoid new swing entries unless intentional; tighten stops and reduce weakest names.")
            elif minutes_to_close <= 90:
                lines.append("⏳ Last 90m: prioritize trade management over fresh risk; prefer exits/reductions over new adds.")

        if session in ('postmarket', 'closed'):
            lines.append("🌙 Outside regular hours: liquidity is thinner and stop execution quality can degrade.")

        if days_to_next_session is not None:
            if days_to_next_session >= 3:
                lines.append(
                    f"📅 LONG-BREAK RISK: next regular session in {days_to_next_session} days"
                    + (f" ({next_session_date})" if next_session_date else "")
                    + ". Reduce gross exposure or hedge remaining longs with puts/collars."
                )
            elif days_to_next_session == 2:
                lines.append("📅 Weekend gap ahead: review every hold for event/news risk before close.")

        lines.append("")
        return lines
    
    # =========================================================================
    # AGENT INTERFACE
    # =========================================================================
    
    def format_for_agent(self) -> str:
        """
        Format current state for agent context.
        Injected into EVERY agent turn.
        
        Structure:
        1. Time context (session, key times)
        2. Market regime (SPY, IWM, GLD, TLT, VIX — multi-signal)
        3. Broker events (just happened)
        4. SHORT stock positions (with orders, hold time, %/hr)
        5. LONG stock positions (with orders, hold time, %/hr)
        6. OPTION positions (long = defined risk, short = flag unprotected)
        7. ORPHAN orders (no matching position - need cleanup)
        8. Account summary (split P&L, exposure, deployed %)
        """
        with self._lock:
            lines = []
            
            # === TIME CONTEXT (session awareness) ===
            try:
                from data.market_hours import get_market_hours_provider
                mh = get_market_hours_provider()
                session_info = mh.get_session_info()
                session = session_info.get('session', 'unknown')
                et_time = session_info.get('current_time_et', '')
                next_trans = session_info.get('next_transition', '')
                
                session_emoji = {
                    'premarket': '🌅',
                    'regular': '☀️',
                    'postmarket': '🌆',
                    'closed': '🌙'
                }.get(session, '⏰')
                
                lines.append(f"=== TIME: {et_time} ET | {session_emoji} {session.upper()} | Next: {next_trans} ===")
                lines.append("")

                # Time-of-day and hold-risk implications (near close / overnight / long weekend)
                risk_lines = self._format_holding_risk_context(session_info)
                if risk_lines:
                    lines.extend(risk_lines)
            except Exception as e:
                logger.debug(f"Failed to get market hours: {e}")
            
            # === MARKET REGIME (multi-signal assessment) ===
            regime_lines = self._format_market_regime()
            if regime_lines:
                lines.extend(regime_lines)

            # === POSITION HEAT (portfolio-level risk) ===
            heat_lines = self._format_position_heat()
            if heat_lines:
                lines.extend(heat_lines)

            # === ECONOMIC CALENDAR (macro event awareness) ===
            try:
                from data.economic_calendar import format_calendar_for_agent
                cal_lines = format_calendar_for_agent()
                if cal_lines:
                    lines.extend(cal_lines)
            except Exception as e:
                logger.debug(f"Economic calendar failed: {e}")

            # === CRITICAL EVENTS FIRST (rejections, errors) ===
            # Only show events from the last 30 minutes to avoid stale noise
            cutoff = datetime.now() - timedelta(minutes=30)
            recent_events = [e for e in self._events if e.timestamp > cutoff][-10:]
            critical = [e for e in recent_events if e.event_type in ('REJECTED', 'ERROR', 'CANCELLED')]
            normal = [e for e in recent_events if e.event_type not in ('REJECTED', 'ERROR', 'CANCELLED')][-5:]
            
            if critical:
                lines.append("🚨🚨🚨 CRITICAL BROKER EVENTS - ACTION REQUIRED 🚨🚨🚨")
                for event in critical:
                    lines.append(f"🚨 {event.format()}")
                lines.append("")
            
            if normal:
                lines.append("=== RECENT BROKER EVENTS ===")
                for event in normal:
                    lines.append(event.format())
                lines.append("")
            
            # Prune events older than 30 min to prevent unbounded growth
            self._events = [e for e in self._events if e.timestamp > cutoff]
            
            # === SEPARATE BY TYPE ===
            shorts = [p for p in self._positions.values() if p.is_short]
            longs = [p for p in self._positions.values() if p.is_long]
            
            stock_shorts = [p for p in shorts if not p.is_option]
            stock_longs = [p for p in longs if not p.is_option]
            opt_shorts = [p for p in shorts if p.is_option]
            opt_longs = [p for p in longs if p.is_option]
            
            # === SHORT STOCK POSITIONS (with orders) ===
            # Count unprotected shorts for alerts (stocks only — short options handled separately)
            unprotected_shorts = []
            for p in stock_shorts:
                orders = self.get_orders_for(p.symbol)
                has_buy_order = any(o.action == "BUY" for o in orders)
                if not has_buy_order:
                    unprotected_shorts.append(p.symbol)
            
            if stock_shorts:
                lines.append("=== SHORT STOCK POSITIONS ===")
                
                for p in sorted(stock_shorts, key=lambda x: x.unrealized_pnl):
                    orders = self.get_orders_for(p.symbol)
                    has_buy_order = any(o.action == "BUY" for o in orders)
                    pnl_pct = p.pnl_percent
                    pnl_str = f"{'+' if p.unrealized_pnl >= 0 else ''}${p.unrealized_pnl:.2f} ({pnl_pct:+.1f}%)"
                    
                    # Add hold time and efficiency (%/hr)
                    hold_str = p.format_hold_time()
                    eff = p.efficiency
                    efficiency_str = f" | {eff:+.1f}%/hr" if eff is not None else ""
                    
                    if orders:
                        orders_str = " | ".join(o.format_brief() for o in orders)
                        if has_buy_order:
                            lines.append(f"  ✅ {p.symbol}: {p.shares} @ ${p.avg_cost:.2f} | P&L: {pnl_str} | Hold: {hold_str}{efficiency_str}")
                        else:
                            lines.append(f"  ⚠️ {p.symbol}: {p.shares} @ ${p.avg_cost:.2f} | P&L: {pnl_str} | Hold: {hold_str}{efficiency_str}")
                        lines.append(f"    Orders: {orders_str}")
                    else:
                        lines.append(f"  🚨 {p.symbol}: {p.shares} @ ${p.avg_cost:.2f} | P&L: {pnl_str} | Hold: {hold_str}{efficiency_str}")
                        lines.append(f"    Orders: NONE")
                lines.append("")
            
            # === LONG STOCK POSITIONS (with orders) ===
            # Count unprotected longs (stocks only — long options have defined risk)
            unprotected_longs = []
            for p in stock_longs:
                orders = self.get_orders_for(p.symbol)
                has_sell_order = any(o.action == "SELL" for o in orders)
                if not has_sell_order:
                    unprotected_longs.append(p.symbol)
            
            if stock_longs:
                lines.append("=== LONG STOCK POSITIONS ===")
                
                # Sort by P&L (worst first)
                for p in sorted(stock_longs, key=lambda x: x.unrealized_pnl):
                    orders = self.get_orders_for(p.symbol)
                    has_sell_order = any(o.action == "SELL" for o in orders)
                    pnl_pct = p.pnl_percent
                    pnl_str = f"{'+' if p.unrealized_pnl >= 0 else ''}${p.unrealized_pnl:.2f} ({pnl_pct:+.1f}%)"
                    
                    # Add hold time and efficiency (%/hr)
                    hold_str = p.format_hold_time()
                    eff = p.efficiency
                    efficiency_str = f" | {eff:+.1f}%/hr" if eff is not None else ""
                    
                    if orders:
                        orders_str = " | ".join(o.format_brief() for o in orders)
                        if has_sell_order:
                            lines.append(f"  ✅ {p.symbol}: {p.shares} @ ${p.avg_cost:.2f} | P&L: {pnl_str} | Hold: {hold_str}{efficiency_str}")
                        else:
                            lines.append(f"  ⚠️ {p.symbol}: {p.shares} @ ${p.avg_cost:.2f} | P&L: {pnl_str} | Hold: {hold_str}{efficiency_str}")
                        lines.append(f"    Orders: {orders_str}")
                    else:
                        lines.append(f"  🚨 {p.symbol}: {p.shares} @ ${p.avg_cost:.2f} | P&L: {pnl_str} | Hold: {hold_str}{efficiency_str}")
                        lines.append(f"    Orders: NONE")
                lines.append("")
            
            # === OPTION POSITIONS ===
            # Long options = defined risk (max loss = premium). No stop needed.
            # Short options = undefined risk UNLESS they're a spread leg.
            self.cleanup_closed_spreads()
            
            unprotected_short_opts = []
            for p in opt_shorts:
                if self.is_spread_leg(p.symbol):
                    continue  # Protected by the spread — not naked
                orders = self.get_orders_for(p.symbol)
                has_buy_order = any(o.action == "BUY" for o in orders)
                if not has_buy_order:
                    unprotected_short_opts.append(p.symbol)
            
            all_opts = opt_longs + opt_shorts
            if all_opts:
                lines.append("=== OPTION POSITIONS ===")
                
                # --- Spread groups (unified display) ---
                displayed_spread_keys = set()
                displayed_groups = set()
                for p in all_opts:
                    group = self.get_spread_for_leg(p.symbol)
                    if group and group.group_id not in displayed_groups:
                        displayed_groups.add(group.group_id)
                        legs = [self._positions[k] for k in group.legs if k in self._positions]
                        if not legs:
                            continue
                        
                        net_pnl = sum(lp.unrealized_pnl for lp in legs)
                        total_cost = sum(abs(lp.avg_cost * lp.quantity) for lp in legs)
                        net_pnl_pct = (net_pnl / total_cost * 100) if total_cost else 0.0
                        pnl_str = f"{'+' if net_pnl >= 0 else ''}${net_pnl:.2f} ({net_pnl_pct:+.1f}%)"
                        
                        hold_leg = min(legs, key=lambda l: (l.entry_time or datetime.max.replace(tzinfo=timezone.utc)))
                        hold_str = hold_leg.format_hold_time()
                        
                        leg_parts = []
                        for lp in legs:
                            d = "S" if lp.is_short else "L"
                            leg_parts.append(f"{lp.symbol}({d}{lp.shares})")
                        
                        lines.append(
                            f"  📦 [{group.strategy}] {group.underlying}: "
                            f"Net P&L: {pnl_str} | Hold: {hold_str}"
                        )
                        lines.append(f"    Legs: {', '.join(leg_parts)}")
                        lines.append(f"    ⚠️  SPREAD — do NOT close individual legs")
                        
                        for k in group.legs:
                            displayed_spread_keys.add(k)
                
                # --- Standalone (non-spread) options ---
                standalone = [p for p in all_opts if p.symbol not in displayed_spread_keys]
                for p in sorted(standalone, key=lambda x: x.unrealized_pnl):
                    pnl_pct = p.pnl_percent
                    pnl_str = f"{'+' if p.unrealized_pnl >= 0 else ''}${p.unrealized_pnl:.2f} ({pnl_pct:+.1f}%)"
                    hold_str = p.format_hold_time()
                    eff = p.efficiency
                    efficiency_str = f" | {eff:+.1f}%/hr" if eff is not None else ""
                    greeks = p.greeks_str
                    direction = "SHORT" if p.is_short else "LONG"
                    
                    if p.is_short and p.symbol in unprotected_short_opts:
                        icon = "🚨"
                    elif p.is_short:
                        icon = "✅"
                    else:
                        icon = "📋"
                    
                    lines.append(
                        f"  {icon} {p.symbol} ({direction}): {p.shares} @ ${p.avg_cost:.2f} | "
                        f"P&L: {pnl_str} | Hold: {hold_str}{efficiency_str}{greeks}"
                    )
                    orders = self.get_orders_for(p.symbol)
                    if orders:
                        orders_str = " | ".join(o.format_brief() for o in orders)
                        lines.append(f"    Orders: {orders_str}")
                lines.append("")
            
            # === PENDING ENTRY ORDERS (awaiting fill — do NOT cancel or re-order) ===
            # === ORPHAN ORDERS (need cleanup) ===
            orphans = self.get_orphan_orders()
            if orphans:
                # Entry-type orders (LMT/MIDPRICE/MKT etc.) for symbols with no position
                # are intentional pending fills — NOT dangerous orphans.
                entry_types = {'LMT', 'MKT', 'MIDPRICE', 'REL', 'MOC', 'MOO', 'LOC', 'LOO'}
                pending_entries = [o for o in orphans
                                   if o.order_type in entry_types or o.sec_type in ('OPT', 'BAG')]
                stale_orphans = [o for o in orphans
                                  if o not in pending_entries]

                if pending_entries:
                    by_symbol = {}
                    for o in pending_entries:
                        by_symbol.setdefault(o.symbol, []).append(o)
                    lines.append(f"=== PENDING ENTRY ORDERS ({len(pending_entries)} awaiting fill) ===")
                    lines.append("These are YOUR orders waiting to fill. Do NOT cancel or re-order these symbols.")
                    for sym, ords in by_symbol.items():
                        for o in ords:
                            lines.append(f"  📋 Order #{o.order_id}: {o.format_brief()} for {sym}")
                    lines.append("")

                if stale_orphans:
                    # Stop/trailing orders for symbols with no position — leftover from closed trades
                    lines.append(f"=== ORPHAN STOP ORDERS ({len(stale_orphans)} total - consider cancelling) ===")
                    for o in stale_orphans[:5]:
                        lines.append(f"  Order #{o.order_id}: {o.format_brief()} for {o.symbol}")
                    if len(stale_orphans) > 5:
                        lines.append(f"  ... +{len(stale_orphans) - 5} more orphan stops (use cancel_all_orphans to clean up)")
                    lines.append("")
            
            # === CLOSED TRADES TODAY ===
            closed_summary = self._format_closed_trades_summary()
            if closed_summary:
                lines.extend(closed_summary)
                lines.append("")
            
            # === PROFIT PROTECTION SCORECARD ===
            try:
                from data.profit_metrics import format_profit_metrics_for_agent
                pm_lines = format_profit_metrics_for_agent(
                    self._closed_trades_today, self._hwm_tracker, self._equity_tracker,
                )
                if pm_lines:
                    lines.extend(pm_lines)
            except Exception as e:
                logger.debug(f"Profit metrics format failed: {e}")
            
            # === ACCOUNT (split P&L + exposure + deployed %) ===
            # Actual cash (TotalCashValue) vs margin buying power (AvailableFunds)
            actual_cash = self._cash
            margin_buying_power = self._available_funds
            net_liq = self._net_liq
            realized_pnl = self._realized_pnl
            if self._broker:
                try:
                    if hasattr(self._broker, 'cash_value') and self._broker.cash_value != 0:
                        actual_cash = self._broker.cash_value
                    if hasattr(self._broker, 'available_funds') and self._broker.available_funds > 0:
                        margin_buying_power = self._broker.available_funds
                    if hasattr(self._broker, 'net_liquidation') and self._broker.net_liquidation > 0:
                        net_liq = self._broker.net_liquidation
                    # Fetch realized P&L from IBKR account values
                    if hasattr(self._broker, 'get_cached_account_values'):
                        for av in self._broker.get_cached_account_values():
                            if av.currency == 'USD' and av.tag == 'RealizedPnL':
                                realized_pnl = float(av.value)
                except Exception as e:
                    logger.debug(f"Failed to fetch account values: {e}")
            
            # Compute unrealized P&L from positions (always accurate)
            unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
            daily_pnl = unrealized_pnl + realized_pnl
            
            # P&L line — split unrealized / realized / day total
            day_sign = '+' if daily_pnl >= 0 else ''
            day_pct = (daily_pnl / net_liq * 100) if net_liq > 0 else 0
            pnl_emoji = '📈' if daily_pnl >= 0 else '📉'
            unreal_sign = '+' if unrealized_pnl >= 0 else ''
            real_sign = '+' if realized_pnl >= 0 else ''
            lines.append(f"{pnl_emoji} P&L: Unrealized {unreal_sign}${unrealized_pnl:,.2f} | Realized {real_sign}${realized_pnl:,.2f} | Day {day_sign}${daily_pnl:,.2f} ({day_sign}{day_pct:.1f}%)")
            
            # Exposure breakdown — net exposure (long - short) for deployed %
            long_exposure = sum(abs(p.market_value) for p in longs)
            short_exposure = sum(abs(p.market_value) for p in shorts)
            net_exposure = long_exposure - short_exposure
            deployed_pct = (net_exposure / net_liq * 100) if net_liq > 0 else 0
            
            # Largest single position as % of net liq
            largest_sym = ""
            largest_pct = 0.0
            for p in list(self._positions.values()):
                pos_pct = (abs(p.market_value) / net_liq * 100) if net_liq > 0 else 0
                if pos_pct > largest_pct:
                    largest_pct = pos_pct
                    largest_sym = p.symbol
            
            long_pct = (long_exposure / net_liq * 100) if net_liq > 0 else 0
            short_pct = (short_exposure / net_liq * 100) if net_liq > 0 else 0
            lines.append(f"EXPOSURE: Long ${long_exposure:,.0f} ({long_pct:.0f}%) | Short ${short_exposure:,.0f} ({short_pct:.0f}%) | Net Deployed {deployed_pct:.0f}%")
            if largest_sym:
                lines.append(f"LARGEST: {largest_sym} {largest_pct:.1f}% of NL")
            cash_pct = (actual_cash / net_liq * 100) if net_liq > 0 else 0
            lines.append(f"ACCOUNT: Cash ${actual_cash:,.2f} ({cash_pct:.0f}% NL) | Margin Buying Power ${margin_buying_power:,.2f} | Net Liq ${net_liq:,.2f}")
            
            # === CAPITAL EFFICIENCY — objective quantiles ===
            all_positions = list(self._positions.values())
            if all_positions:
                import statistics
                hold_hours_list = [p.hold_time_hours for p in all_positions if p.hold_time_hours is not None]
                eff_list = [p.efficiency for p in all_positions if p.efficiency is not None]
                
                eff_parts = [f"{len(all_positions)} positions"]
                
                if hold_hours_list:
                    median_hrs = statistics.median(hold_hours_list)
                    median_days = round(median_hrs / 24, 1)
                    eff_parts.append(f"median hold {median_days}d")
                
                # Capital in longest 25% holds
                if hold_hours_list:
                    positions_sorted = sorted(all_positions, key=lambda p: p.hold_time_hours or 0, reverse=True)
                    num_longest = max(1, int(len(all_positions) * 0.25))
                    longest_25pct = positions_sorted[:num_longest]
                    long_cap = sum(abs(p.market_value) for p in longest_25pct)
                    long_pct = round((long_cap / net_liq * 100), 1) if net_liq > 0 else 0
                    eff_parts.append(f"longest 25% holds ${long_cap:,.0f} ({long_pct}%)")
                
                if eff_list:
                    median_eff = statistics.median(eff_list)
                    eff_parts.append(f"median {median_eff:+.1f}%/hr")
                
                lines.append(f"CAPITAL EFFICIENCY: {' | '.join(eff_parts)}")
            
            # === ROTATION CANDIDATES — positions bleeding capital ===
            rotation = []
            for p in all_positions if all_positions else []:
                eff = p.efficiency
                if eff is not None and eff < 0:
                    rotation.append(p)
            if rotation:
                rotation.sort(key=lambda p: p.efficiency)  # worst first
                lines.append("")
                lines.append("=== ROTATION CANDIDATES (negative %/hr) ===")
                for p in rotation:
                    lines.append(
                        f"  {p.symbol}: {p.format_hold_time()} held | "
                        f"{p.efficiency:+.1f}%/hr | "
                        f"P&L ${p.unrealized_pnl:+.2f} ({p.pnl_percent:+.1f}%)"
                    )
                lines.append("")
            
            # Summary with alerts — break out stocks vs options
            summary_parts = []
            if stock_longs or stock_shorts:
                summary_parts.append(f"{len(stock_longs)} stock longs, {len(stock_shorts)} stock shorts")
            if opt_longs or opt_shorts:
                summary_parts.append(f"{len(opt_longs)} option longs, {len(opt_shorts)} option shorts")
            if not summary_parts:
                summary_parts.append("0 positions")
            summary_parts.append(f"{len(orphans)} orphan orders")
            if unprotected_shorts:
                summary_parts.append(f"🚨 {len(unprotected_shorts)} UNPROTECTED SHORT STOCKS")
            if unprotected_longs:
                summary_parts.append(f"🚨 {len(unprotected_longs)} UNPROTECTED LONG STOCKS")
            if unprotected_short_opts:
                summary_parts.append(f"🚨 {len(unprotected_short_opts)} NAKED SHORT OPTIONS")
            lines.append(f"SUMMARY: {', '.join(summary_parts)}")
            
            return "\n".join(lines)
    
    # =========================================================================
    # IBKR EVENT HANDLERS
    # =========================================================================
    
    def on_position(self, position):
        """Handle IBKR positionEvent."""
        try:
            contract = position.contract
            symbol = contract.symbol
            
            # Create unique key for options vs stocks
            if hasattr(contract, 'right') and contract.right in ('C', 'P'):
                # It's an option
                expiry = getattr(contract, 'lastTradeDateOrContractMonth', '')
                strike = getattr(contract, 'strike', 0)
                symbol_key = f"{symbol}_{contract.right}_{strike}_{expiry}"
            else:
                symbol_key = symbol
            
            qty = position.position
            avg_cost = position.avgCost
            self.update_position(symbol_key, qty, avg_cost)
        except Exception as e:
            logger.debug(f"Position event error: {e}")
    
    def on_portfolio_update(self, item):
        """Handle IBKR updatePortfolioEvent."""
        try:
            contract = item.contract
            symbol = contract.symbol
            
            # Create unique key for options vs stocks
            # Options: "AES_C_17_20260220" for AES Call $17 Feb 20
            # Stocks: "AES"
            if hasattr(contract, 'right') and contract.right in ('C', 'P'):
                # It's an option
                expiry = getattr(contract, 'lastTradeDateOrContractMonth', '')
                strike = getattr(contract, 'strike', 0)
                symbol_key = f"{symbol}_{contract.right}_{strike}_{expiry}"
            else:
                symbol_key = symbol
            
            qty = item.position
            avg_cost = item.averageCost
            market_value = item.marketValue
            unrealized_pnl = item.unrealizedPNL
            market_price = item.marketPrice
            self.update_position(symbol_key, qty, avg_cost, market_value, unrealized_pnl, market_price)
        except Exception as e:
            logger.debug(f"Portfolio update error: {e}")
    
    def on_order_status(self, trade):
        """Handle IBKR orderStatusEvent."""
        try:
            order = trade.order
            status = trade.orderStatus.status
            symbol = trade.contract.symbol
            sec_type = getattr(trade.contract, 'secType', 'STK') or 'STK'
            filled = trade.orderStatus.filled or 0
            remaining = trade.orderStatus.remaining or 0
            
            # Sanitize absurd prices from IBKR (TRAIL orders report
            # auxPrice as float('inf') / sys.float_info.max before reference set)
            raw_aux = order.auxPrice or 0
            raw_lmt = order.lmtPrice or 0
            aux_price = raw_aux if raw_aux < 1e30 else 0.0
            limit_price = raw_lmt if raw_lmt < 1e30 else 0.0

            self.update_order(
                order_id=order.orderId,
                symbol=symbol,
                action=order.action,
                quantity=order.totalQuantity,
                order_type=order.orderType,
                status=status,
                aux_price=aux_price,
                limit_price=limit_price,
                sec_type=sec_type,
                filled_qty=filled,
                remaining_qty=remaining
            )
            
            # Push events for important status changes
            if status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                self.push_event(
                    "FILLED", symbol,
                    f"{order.orderType} {order.action} {order.totalQuantity} {symbol} @ ${fill_price:.2f}",
                    order_id=order.orderId,
                    is_critical=True
                )
            elif filled > 0 and remaining > 0:
                # Partial fill — keep order active, notify agent
                fill_price = trade.orderStatus.avgFillPrice
                self.push_event(
                    "PARTIAL_FILL", symbol,
                    f"Order {order.orderId}: {int(filled)}/{int(filled + remaining)} filled @ ${fill_price:.2f}",
                    order_id=order.orderId
                )
            elif status in ('Cancelled', 'ApiCancelled'):
                self.push_event(
                    "CANCELLED", symbol,
                    f"Order {order.orderId} cancelled",
                    order_id=order.orderId
                )
            elif status == 'Rejected':
                self.push_event(
                    "REJECTED", symbol,
                    f"Order {order.orderId} REJECTED!",
                    order_id=order.orderId,
                    is_critical=True
                )
        except Exception as e:
            logger.debug(f"Order status error: {e}")
    
    def on_error(self, req_id: int, error_code: int, error_string: str, contract):
        """Handle IBKR errorEvent."""
        try:
            symbol = contract.symbol if contract else "UNKNOWN"
            
            # Farm connectivity restored messages — record state change
            if error_code in (2104, 2106, 2158):
                self._record_farm_event(error_code, error_string, connected=True)
                return

            # Farm connectivity BROKEN messages — record + alert
            if error_code in (2103, 2105):
                self._record_farm_event(error_code, error_string, connected=False)
                logger.info(f"Farm connectivity issue {error_code}: {error_string}")
                return
            
            # Error 10147: Order not found - remove stale order from tracking
            if error_code == 10147 and req_id > 0:
                with self._lock:
                    if req_id in self._orders:
                        logger.info(f"Removing stale order {req_id} (not found on IBKR)")
                        del self._orders[req_id]
                # Don't log as critical - it's just cleanup
                return
            
            # Order-related errors are critical
            if error_code in (201, 202, 203, 110, 2102):
                message = f"Error {error_code}: {error_string}"
                critical = True

                if error_code == 202 and req_id and req_id > 0 and self._broker is not None:
                    try:
                        attribution = self._broker.get_cancel_attribution(req_id)
                        if attribution.get("kind") == "self_cancel":
                            source = attribution.get("source", "unknown")
                            age_s = attribution.get("age_seconds", "?")
                            message = f"{message} [self_cancel source={source} age={age_s}s]"
                            critical = False
                        else:
                            message = f"{message} [broker_cancel]"
                    except Exception:
                        pass

                self.push_event(
                    "ERROR", symbol,
                    message,
                    order_id=req_id if req_id > 0 else None,
                    is_critical=critical
                )
        except Exception as e:
            logger.debug(f"Error event error: {e}")
    
    def on_account_value(self, account_value):
        """Handle IBKR accountValueEvent."""
        try:
            if account_value.currency == 'USD':
                if account_value.tag == 'CashBalance':
                    self._cash = float(account_value.value)
                elif account_value.tag == 'AvailableFunds':
                    self._available_funds = float(account_value.value)
                elif account_value.tag == 'NetLiquidation':
                    self._net_liq = float(account_value.value)
                elif account_value.tag == 'RealizedPnL':
                    self._realized_pnl = float(account_value.value)
                elif account_value.tag == 'UnrealizedPnL':
                    self._unrealized_pnl = float(account_value.value)
            # Update session equity tracker for giveback %
            if self._net_liq > 0:
                try:
                    self._equity_tracker.update(self._net_liq)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Account value error: {e}")


# Global instance
_live_state: Optional[LiveState] = None


def get_live_state() -> LiveState:
    """Get or create the global LiveState."""
    global _live_state
    if _live_state is None:
        _live_state = LiveState()
    return _live_state


async def wire_to_broker(broker) -> None:
    """
    Wire LiveState to IBKR broker events.
    Call this after broker connection is established.
    Must be awaited from an async context.
    """
    import asyncio
    
    state = get_live_state()
    state._broker = broker
    ib = broker.ib  # The ib_insync.IB instance
    
    # Subscribe to events FIRST (before requesting data)
    ib.positionEvent += state.on_position
    ib.updatePortfolioEvent += state.on_portfolio_update
    ib.orderStatusEvent += state.on_order_status
    ib.errorEvent += state.on_error
    ib.accountValueEvent += state.on_account_value
    
    # IBKR sends portfolio updates during initial connection sync
    # We just need to wait for all data to arrive and then read it
    try:
        # Wait for data to stabilize - IBKR streams it during connection
        await asyncio.sleep(0.5)
        last_count = 0
        for _ in range(10):  # Max 10 retries (5 seconds total)
            current_count = len(ib.portfolio())
            if current_count > 0 and current_count == last_count:
                break  # Count stabilized
            last_count = current_count
            await asyncio.sleep(0.5)
        
        # Now ib.portfolio() has the full data with P&L
        portfolio_items = ib.portfolio()
        for item in portfolio_items:
            state.on_portfolio_update(item)
        
        # Fetch entry times for each position
        for symbol, pos in list(state._positions.items()):
            if pos.entry_time is None:
                try:
                    # Only for stock positions (not options with _ in symbol)
                    if '_' not in symbol:
                        entry_time = await broker.get_entry_time_for_symbol(symbol, pos.direction)
                        if entry_time:
                            state._positions[symbol].entry_time = entry_time
                            logger.debug(f"  Entry time for {symbol}: {entry_time.isoformat()}")
                except Exception as e:
                    logger.debug(f"  Could not get entry time for {symbol}: {e}")
        
        # Log each position we synced
        for item in portfolio_items:
            if item.position != 0:
                symbol = item.contract.symbol
                hold_str = ""
                if symbol in state._positions and state._positions[symbol].entry_time:
                    hours = state._positions[symbol].hold_time_hours
                    if hours is not None:
                        if hours < 24:
                            hold_str = f" (held {hours:.1f}h)"
                        else:
                            hold_str = f" (held {hours/24:.1f}d)"
                logger.info(f"  SYNCED: {item.contract.symbol} qty={item.position} P&L=${item.unrealizedPNL:.2f}{hold_str}")
        
        logger.info(f"Synced {len(portfolio_items)} portfolio items with P&L (non-zero: {len([p for p in portfolio_items if p.position != 0])})")
        
        # Get account values (cash, net liq) - broker should have subscribed already
        try:
            account_values = ib.accountValues()
            for av in account_values:
                if av.tag == 'CashBalance' and av.currency == 'USD':
                    state._cash = float(av.value)
                elif av.tag == 'TotalCashValue' and av.currency == 'USD':
                    # Fallback if CashBalance not available
                    if state._cash == 0:
                        state._cash = float(av.value)
                elif av.tag == 'AvailableFunds' and av.currency == 'USD':
                    state._available_funds = float(av.value)
                elif av.tag == 'NetLiquidation':
                    state._net_liq = float(av.value)
            logger.info(f"Account: Available ${state._available_funds:,.2f}, Cash ${state._cash:,.2f}, Net Liq ${state._net_liq:,.2f}")
        except Exception as e:
            logger.warning(f"Failed to get account values: {e}")
        
        # Orders - use reqAllOpenOrdersAsync to get orders from ALL client sessions
        # This is critical for seeing stops placed by previous agent sessions.
        # Clear stale orders first so we only track what IBKR actually reports.
        state._orders.clear()
        await ib.reqAllOpenOrdersAsync()
        await asyncio.sleep(0.5)  # Give time for orders to arrive
        for trade in ib.openTrades():
            state.on_order_status(trade)
        
        logger.info(f"LiveState wired: {len(state._positions)} positions, {len(state._orders)} orders")
    except Exception as e:
        logger.warning(f"Initial sync error: {e}")
    
    logger.info("LiveState wired to IBKR broker events")
