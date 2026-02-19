"""
IBKR Core Connector - Connection Management and Base Class

This module provides the core IBKRConnector class with:
- Singleton pattern for connection management
- Thread-safe connection/disconnection
- Event handler registration
- Real-time market data streaming
- Base infrastructure for order and query operations

The IBKRConnector class imports mixins from:
- ibkr_orders.py: Order placement and management
- ibkr_options.py: Options chains and spreads
- ibkr_queries.py: Account and position queries
"""

import logging
import asyncio
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from ib_insync import IB, Order, Trade, Fill
from ib_insync.contract import Stock, Option

from execution.ibkr_utils import resolve_ibkr_endpoint
from execution.order_types import IBKROrderType

logger = logging.getLogger(__name__)


# ============================================================
# ORDER STATE TRACKING
# ============================================================

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    INACTIVE = "inactive"


@dataclass
class OrderState:
    """
    Tracks state of an individual order.

    Used for monitoring bracket groups and OCA orders.
    """
    order_id: int
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'LMT', 'STP', 'TRAIL', etc.
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    oca_group: Optional[str] = None
    bracket_group: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'status': self.status.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'trail_amount': self.trail_amount,
            'trail_percent': self.trail_percent,
            'filled_qty': self.filled_qty,
            'avg_fill_price': self.avg_fill_price,
            'oca_group': self.oca_group,
            'bracket_group': self.bracket_group,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class BracketGroup:
    """
    Tracks a complete bracket order group (entry + stop + target).
    """
    group_id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    target_order_id: Optional[int] = None
    oca_group: Optional[str] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    quantity: int = 0
    status: str = "pending"  # pending, active, closed_profit, closed_loss, cancelled
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    realized_pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'group_id': self.group_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_order_id': self.entry_order_id,
            'stop_order_id': self.stop_order_id,
            'target_order_id': self.target_order_id,
            'oca_group': self.oca_group,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'quantity': self.quantity,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'realized_pnl': self.realized_pnl
        }


# Import mixins after defining base classes to avoid circular imports
from execution.ibkr_orders import IBKROrdersMixin
from execution.ibkr_options import IBKROptionsMixin
from execution.ibkr_queries import IBKRQueriesMixin


class IBKRConnector(IBKROrdersMixin, IBKROptionsMixin, IBKRQueriesMixin):
    """
    IBKR connector with essential trading functionality.
    Thread-safe singleton pattern.

    Inherits from:
    - IBKROrdersMixin: Order placement and management methods
    - IBKROptionsMixin: Options chains and spreads methods
    - IBKRQueriesMixin: Account and position query methods
    """

    _instance: Optional['IBKRConnector'] = None
    _lock: Lock = Lock()
    _async_lock: Optional[asyncio.Lock] = None  # Created lazily per event loop

    def __new__(cls) -> 'IBKRConnector':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @property
    def async_lock(self) -> asyncio.Lock:
        """Get async lock, creating if needed for current event loop."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def __init__(self):
        if self._initialized:
            return

        # Resolve endpoint from environment
        self.host, self.port, self.mode = resolve_ibkr_endpoint()
        # Fixed client ID from env var (default 1) — MUST be consistent across restarts
        # so the agent can cancel orders from prior sessions
        self.client_id = int(os.environ.get('IBKR_CLIENT_ID', '1'))

        # Connection state
        self.ib = IB()
        self._connected = False
        self.account_id: Optional[str] = None
        self.net_liquidation: float = 0.0
        self.cash_value: float = 0.0  # TotalCashValue - actual cash
        self.available_funds: float = 0.0  # AvailableFunds - what you can actually spend (CASH ONLY)
        self.day_trades_remaining: int = 3  # PDT tracking - updated on connect

        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Active streaming subscriptions: symbol -> ticker
        self._tickers: Dict[str, Any] = {}

        # Background heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Execution tracking - stores ALL fills with actual prices
        # Key: symbol, Value: list of execution records
        self._executions: Dict[str, List[Dict[str, Any]]] = {}
        self._execution_lock = Lock()

        # Order state tracking
        self._order_states: Dict[int, OrderState] = {}  # order_id -> OrderState
        self._bracket_groups: Dict[str, BracketGroup] = {}  # group_id -> BracketGroup
        self._order_state_lock = Lock()
        self._order_status_listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._local_cancel_requests: Dict[int, Dict[str, Any]] = {}
        self._local_cancel_lock = Lock()

        # Store strong references to event handlers (prevents weakref issues)
        self._disconnect_handler = self._on_disconnect
        self._execution_handler = self._on_execution
        self._order_status_handler = self._on_order_status

        # Register event handlers
        self._register_handlers()

        self._initialized = True
        logger.info(f"IBKRConnector initialized ({self.mode} mode, {self.host}:{self.port})")

    def _register_handlers(self):
        """Register event handlers on the current IB instance."""
        self.ib.disconnectedEvent += self._disconnect_handler
        self.ib.execDetailsEvent += self._execution_handler
        self.ib.orderStatusEvent += self._order_status_handler

    def is_connected(self) -> bool:
        """Check if connected to IBKR TWS/Gateway."""
        return self._connected and self.ib.isConnected()

    def _record_local_cancel_request(self, order_id: int, source: str = "unknown") -> None:
        """Track a local cancel request so later Error 202 can be attributed."""
        if not order_id:
            return
        now = datetime.now(timezone.utc)
        with self._local_cancel_lock:
            self._local_cancel_requests[int(order_id)] = {
                "timestamp": now,
                "source": source,
            }
            # Keep map small
            stale = [
                oid for oid, meta in self._local_cancel_requests.items()
                if (now - meta.get("timestamp", now)).total_seconds() > 300
            ]
            for oid in stale:
                self._local_cancel_requests.pop(oid, None)

    def _cancel_order_with_tracking(self, order: Order, source: str = "unknown") -> None:
        """Cancel order while recording local attribution metadata."""
        order_id = int(getattr(order, "orderId", 0) or 0)
        if order_id:
            self._record_local_cancel_request(order_id, source=source)
        self.ib.cancelOrder(order)

    def get_cancel_attribution(self, order_id: int, ttl_seconds: int = 30) -> Dict[str, Any]:
        """Classify cancel as self-initiated vs broker-side using a recent local cancel ledger."""
        now = datetime.now(timezone.utc)
        with self._local_cancel_lock:
            meta = self._local_cancel_requests.get(int(order_id))

        if not meta:
            return {
                "kind": "broker_cancel",
                "order_id": int(order_id),
            }

        ts = meta.get("timestamp", now)
        age_s = max((now - ts).total_seconds(), 0.0)
        if age_s <= ttl_seconds:
            return {
                "kind": "self_cancel",
                "order_id": int(order_id),
                "source": meta.get("source", "unknown"),
                "age_seconds": round(age_s, 2),
            }

        return {
            "kind": "broker_cancel",
            "order_id": int(order_id),
            "stale_local_cancel": True,
            "age_seconds": round(age_s, 2),
        }

    def _unregister_handlers(self):
        """Safely remove event handlers from the current IB instance."""
        try:
            self.ib.disconnectedEvent -= self._disconnect_handler
        except Exception:
            pass
        try:
            self.ib.execDetailsEvent -= self._execution_handler
        except Exception:
            pass
        try:
            self.ib.orderStatusEvent -= self._order_status_handler
        except Exception:
            pass

    def _on_disconnect(self) -> None:
        """Handle disconnection from TWS."""
        logger.warning("Disconnected from IBKR TWS")
        self._connected = False
        self._tickers.clear()

    def _on_execution(self, trade: Trade, fill: Fill) -> None:
        """
        Handle execution event - stores actual fill data.

        This is event-driven, so we capture EVERY fill as it happens.
        No more estimated exit prices!
        """
        try:
            symbol = trade.contract.symbol
            execution = fill.execution
            commission = fill.commissionReport.commission if fill.commissionReport else 0

            exec_record = {
                'symbol': symbol,
                'side': execution.side,  # 'BOT' or 'SLD'
                'shares': int(execution.shares),
                'price': float(execution.price),
                'avg_price': float(execution.avgPrice),
                'time': execution.time.isoformat() if execution.time else datetime.now(timezone.utc).isoformat(),
                'order_id': execution.orderId,
                'exec_id': execution.execId,
                'commission': commission,
                'order_type': trade.order.orderType,
                'oca_group': trade.order.ocaGroup or None
            }

            with self._execution_lock:
                if symbol not in self._executions:
                    self._executions[symbol] = []
                self._executions[symbol].append(exec_record)

            logger.info(f"Execution captured: {execution.side} {execution.shares} {symbol} @ ${execution.price:.2f}")

        except Exception as e:
            logger.error(f"Failed to process execution event: {e}")

    def _on_order_status(self, trade: Trade) -> None:
        """Handle order status changes."""
        try:
            status = trade.orderStatus.status
            symbol = trade.contract.symbol
            order_type = trade.order.orderType
            event = {
                "order_id": trade.order.orderId,
                "symbol": symbol,
                "status": status,
                "order_type": order_type,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
            }

            if status == 'Filled':
                logger.info(f"[OK] Order FILLED: {order_type} {symbol}")
            elif status == 'Cancelled':
                logger.info(f"[X] Order CANCELLED: {order_type} {symbol}")

            for listener in list(self._order_status_listeners):
                try:
                    listener(event)
                except Exception as e:
                    logger.debug(f"Order status listener error: {e}")
        except Exception as e:
            logger.debug(f"Order status event error: {e}")

    def register_order_status_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        if callback not in self._order_status_listeners:
            self._order_status_listeners.append(callback)

    def unregister_order_status_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        if callback in self._order_status_listeners:
            self._order_status_listeners.remove(callback)

    @property
    def connected(self) -> bool:
        """Check if actually connected (not just flag)."""
        if not self._connected:
            return False
        # Verify connection is still alive
        return self.ib.isConnected()

    @connected.setter
    def connected(self, value: bool):
        self._connected = value

    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    # ========== CONNECTION ==========

    async def connect(self, max_retries: int = 3) -> bool:
        """Connect to IBKR TWS/Gateway."""
        if self.connected:
            return True

        async with self.async_lock:
            # Double-check after acquiring lock
            if self.connected:
                return True

            for attempt in range(max_retries):
                try:
                    # Use the fixed client_id. On retry, try client_id+attempt to handle
                    # stale connection on the same ID (e.g., TWS still thinks old session is active)
                    current_client_id = self.client_id + attempt
                    logger.info(f"Connecting to IBKR ({self.host}:{self.port}, client_id={current_client_id}, attempt {attempt + 1})")

                    # Clean up old IB instance handlers before creating new one
                    self._unregister_handlers()

                    # Create fresh IB instance on each attempt
                    self.ib = IB()

                    # Re-register all event handlers on new IB instance
                    self._register_handlers()

                    await self.ib.connectAsync(
                        host=self.host,
                        port=self.port,
                        clientId=current_client_id,
                        timeout=10
                    )

                    # Brief wait for connection to stabilize
                    await asyncio.sleep(0.5)

                    if self.ib.isConnected():
                        self._connected = True
                        self.client_id = current_client_id  # Store the working client ID

                        # Force live (real-time) market data — paper accounts have it free
                        self.ib.reqMarketDataType(1)
                        logger.info("Market data type set to LIVE (1)")

                        # Get account ID
                        accounts = self.ib.managedAccounts()
                        if accounts:
                            self.account_id = accounts[0]
                            logger.info(f"Connected to IBKR account: {self.account_id}")

                            # Fetch account values
                            await self._update_account_values()

                        # Start background heartbeat
                        self._start_heartbeat()

                        return True
                    else:
                        logger.warning(f"Connection timeout on attempt {attempt + 1}")

                except Exception as e:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

            logger.error(f"Failed to connect after {max_retries} attempts")
            return False

    async def disconnect(self):
        """Disconnect from IBKR."""
        if self.connected:
            # Stop heartbeat
            self._stop_heartbeat()

            # Cancel all streaming subscriptions
            for ticker in self._tickers.values():
                try:
                    self.ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._tickers.clear()

            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

        # Shutdown thread pool executor to prevent thread leaks
        if hasattr(self, '_executor') and self._executor:
            try:
                self._executor.shutdown(wait=False)
                logger.debug("ThreadPoolExecutor shutdown complete")
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")

    async def _update_account_values(self):
        """Fetch and update account values (available funds, net liquidation, PDT status)."""
        try:
            # Use accountValues (synchronous, cached) instead of accountSummary
            account_values = self.ib.accountValues(self.account_id)

            for av in account_values:
                if av.tag == 'NetLiquidation':
                    self.net_liquidation = float(av.value)
                elif av.tag == 'TotalCashValue':
                    self.cash_value = float(av.value)
                elif av.tag == 'AvailableFunds':
                    self.available_funds = float(av.value)
                elif av.tag == 'DayTradesRemaining':
                    self.day_trades_remaining = int(float(av.value))

            if self.available_funds > 0 or self.net_liquidation > 0:
                logger.info(f"Account values - Available: ${self.available_funds:,.2f}, Cash: ${self.cash_value:,.2f}, Net Liq: ${self.net_liquidation:,.2f}, Day Trades: {self.day_trades_remaining}")
            else:
                # If still zero, request subscription
                self.ib.reqAccountUpdates(subscribe=True, account=self.account_id)
                await asyncio.sleep(0.5)  # Give time for update
                account_values = self.ib.accountValues(self.account_id)
                for av in account_values:
                    if av.tag == 'NetLiquidation':
                        self.net_liquidation = float(av.value)
                    elif av.tag == 'TotalCashValue':
                        self.cash_value = float(av.value)
                    elif av.tag == 'AvailableFunds':
                        self.available_funds = float(av.value)
                logger.info(f"Account values (subscribed) - Available: ${self.available_funds:,.2f}, Cash: ${self.cash_value:,.2f}, Net Liq: ${self.net_liquidation:,.2f}")
        except Exception as e:
            logger.warning(f"Failed to fetch account values: {e}")

    async def _run_in_executor(self, func, *args, timeout: float = 30.0):
        """Run blocking function in thread pool."""
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(self._executor, func, *args),
            timeout=timeout
        )

    async def _ensure_connected(self) -> bool:
        """Ensure we're connected, attempt reconnect if not."""
        if self.connected:
            return True

        logger.info("Connection lost or not established, attempting reconnect...")
        self._connected = False  # Reset state
        return await self.connect()

    async def _wait_for_fill(self, trade, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Wait for a trade to fill or be cancelled.

        Args:
            trade: The ib_insync Trade object to monitor
            timeout: Max seconds to wait for fill

        Returns:
            Dict with 'filled', 'status', 'avg_fill_price', 'filled_quantity'
        """
        start_time = asyncio.get_event_loop().time()
        poll_interval = 0.1  # 100ms between checks

        while True:
            # Check if we've exceeded timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Fill wait timeout after {elapsed:.1f}s, status: {trade.orderStatus.status}")
                return {
                    'filled': False,
                    'status': 'Timeout',
                    'avg_fill_price': None,
                    'filled_quantity': 0
                }

            # Check current status
            status = trade.orderStatus.status

            if status == 'Filled':
                avg_price = trade.orderStatus.avgFillPrice
                filled_qty = trade.orderStatus.filled
                logger.info(f"Order FILLED: {filled_qty} @ ${avg_price:.2f}")
                return {
                    'filled': True,
                    'status': 'Filled',
                    'avg_fill_price': avg_price,
                    'filled_quantity': int(filled_qty)
                }

            if status in ('Cancelled', 'ApiCancelled'):
                logger.warning(f"Order {status}, not filled")
                return {
                    'filled': False,
                    'status': status,
                    'avg_fill_price': None,
                    'filled_quantity': 0
                }

            # Inactive means order is valid but not immediately fillable
            # This can happen with limit orders - keep waiting for DAY orders
            if status == 'Inactive':
                # For paper trading, Inactive often means price moved -
                # give it more time before giving up
                if elapsed < timeout * 0.9:
                    await asyncio.sleep(poll_interval)
                    continue
                else:
                    logger.warning(f"Order Inactive after {elapsed:.1f}s - limit price may be stale")
                    return {
                        'filled': False,
                        'status': status,
                        'avg_fill_price': None,
                        'filled_quantity': 0
                    }

            # Still pending (PreSubmitted, Submitted, etc.)
            await asyncio.sleep(poll_interval)

    # ========== EMERGENCY OPERATIONS ==========

    async def flatten_all(self) -> dict:
        """Cancel all open orders and close all positions with market orders.

        Returns a summary dict with cancelled/closed counts.
        This is the broker-level nuclear option — caller decides when to invoke.
        """
        if not await self._ensure_connected():
            return {'success': False, 'error': 'Not connected'}

        result = {'orders_cancelled': 0, 'orders_total': 0,
                  'positions_closed': 0, 'positions_total': 0,
                  'errors': []}

        # Step 1: Cancel ALL open orders
        try:
            open_orders = await self.get_open_orders()
            result['orders_total'] = len(open_orders)
            for order in open_orders:
                try:
                    oid = order.get('order_id')
                    if oid:
                        await self.cancel_order(oid)
                        result['orders_cancelled'] += 1
                except Exception as e:
                    result['errors'].append(f'cancel order {order.get("order_id")}: {e}')
        except Exception as e:
            result['errors'].append(f'get_open_orders: {e}')

        await asyncio.sleep(1)  # Let cancellations process

        # Step 2: Close ALL positions with MKT IOC orders
        try:
            positions = await self.get_positions()
            result['positions_total'] = len(positions)
            for pos in positions:
                try:
                    symbol = pos.get('symbol', '')
                    qty = pos.get('quantity', 0)
                    if qty == 0 or not symbol:
                        continue
                    action = 'SELL' if qty > 0 else 'BUY'
                    close_qty = abs(qty)
                    order_result = await self._place_order(
                        symbol=symbol,
                        action=action,
                        quantity=close_qty,
                        order_type='MKT',
                        tif='IOC',
                        order_name='EMERGENCY_FLATTEN',
                    )
                    if order_result.get('success'):
                        result['positions_closed'] += 1
                    else:
                        result['errors'].append(f'flatten {symbol}: {order_result}')
                except Exception as e:
                    result['errors'].append(f'flatten {pos.get("symbol", "?")}: {e}')
        except Exception as e:
            result['errors'].append(f'get_positions: {e}')

        result['success'] = True
        logger.critical(f"FLATTEN ALL: cancelled {result['orders_cancelled']}/{result['orders_total']} orders, "
                        f"closed {result['positions_closed']}/{result['positions_total']} positions")
        return result

    async def flatten_limits(self) -> dict:
        """Cancel open orders and close all positions with LIMIT orders at mid.

        This is a non-emergency flatten. Uses only limit orders (no markets).
        Returns a summary dict with cancelled/closed counts and any errors.
        """
        if not await self._ensure_connected():
            return {'success': False, 'error': 'Not connected'}

        result = {
            'orders_cancelled': 0,
            'orders_total': 0,
            'positions_closed': 0,
            'positions_total': 0,
            'errors': [],
        }

        # Step 1: Cancel ALL open orders to prevent overfills/over-shorts
        try:
            open_orders = await self.get_open_orders()
            result['orders_total'] = len(open_orders)
            for order in open_orders:
                try:
                    oid = order.get('order_id')
                    if oid:
                        await self.cancel_order(oid)
                        result['orders_cancelled'] += 1
                except Exception as e:
                    result['errors'].append(f'cancel order {order.get("order_id")}: {e}')
        except Exception as e:
            result['errors'].append(f'get_open_orders: {e}')

        await asyncio.sleep(2)  # Let cancellations + any final fills settle

        # Step 2: Re-read positions AFTER cancellations (flip-guard)
        # Positions may have changed if fills arrived during cancel phase
        try:
            positions = await self.get_positions()
            result['positions_total'] = len(positions)
            for pos in positions:
                try:
                    symbol = pos.get('symbol', '')
                    qty = pos.get('quantity', 0)
                    sec_type = pos.get('sec_type', 'STK')
                    if qty == 0 or not symbol:
                        continue

                    if sec_type == 'OPT':
                        expiration = pos.get('expiration')
                        strike = pos.get('strike')
                        right = pos.get('right')
                        if not all([expiration, strike, right]):
                            result['errors'].append(f'option details missing for {symbol}')
                            continue

                        contract = Option(symbol, expiration, float(strike), right, 'SMART')
                        await self.ib.qualifyContractsAsync(contract)
                        ticker = self.ib.reqMktData(contract, '', snapshot=True, regulatorySnapshot=False)
                        await asyncio.sleep(1)
                        bid = ticker.bid or 0
                        ask = ticker.ask or 0
                        self.ib.cancelMktData(contract)
                        if bid > 0 and ask > 0:
                            limit_price = round((bid + ask) / 2, 2)
                        else:
                            result['errors'].append(f'no midpoint for option {symbol} {strike}{right} {expiration}')
                            continue

                        order_result = await self.close_option_position(
                            symbol,
                            expiration=expiration,
                            strike=float(strike),
                            right=str(right),
                            quantity=abs(int(qty)),
                            limit_price=limit_price,
                            reason='flatten_limits',
                        )
                    else:
                        # Use MarketData.app for live quotes (no IBKR subscription needed)
                        from data.marketdata_client import get_marketdata_client
                        mda = get_marketdata_client()
                        quote = await mda.get_quote(symbol)
                        mid = quote.get('mid') if quote else None
                        bid = quote.get('bid') if quote else None
                        ask = quote.get('ask') if quote else None

                        if mid and mid > 0:
                            limit_price = round(mid, 4) if mid < 1 else round(mid, 2)
                        elif bid and ask and bid > 0 and ask > 0:
                            m = (bid + ask) / 2
                            limit_price = round(m, 4) if m < 1 else round(m, 2)
                        elif pos.get('market_price') and pos['market_price'] > 0:
                            limit_price = round(pos['market_price'], 4) if pos['market_price'] < 1 else round(pos['market_price'], 2)
                            logger.warning(f"Using position market_price {limit_price} for {symbol} (MDA unavailable)")
                        else:
                            result['errors'].append(f'no price data for {symbol}')
                            continue

                        action = 'SELL' if qty > 0 else 'BUY'
                        order_result = await self.place_limit_order(
                            symbol, action, abs(int(qty)), float(limit_price)
                        )

                    if order_result.get('success'):
                        result['positions_closed'] += 1
                    else:
                        result['errors'].append(f'flatten {symbol}: {order_result}')
                except Exception as e:
                    result['errors'].append(f'flatten {pos.get("symbol", "?")}: {e}')
        except Exception as e:
            result['errors'].append(f'get_positions: {e}')

        result['success'] = len(result['errors']) == 0
        logger.warning(
            "FLATTEN LIMITS: cancelled %s/%s orders, closed %s/%s positions, errors=%s",
            result['orders_cancelled'],
            result['orders_total'],
            result['positions_closed'],
            result['positions_total'],
            len(result['errors']),
        )
        return result

    # ========== HEARTBEAT ==========

    def _start_heartbeat(self):
        """Start background heartbeat to prevent idle disconnect."""
        self._stop_heartbeat()  # Cancel any existing task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.debug("IBKR heartbeat started (60s interval)")

    def _stop_heartbeat(self):
        """Cancel background heartbeat."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
            logger.debug("IBKR heartbeat stopped")

    async def _heartbeat_loop(self):
        """Ping broker every 60s to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(60)
                if not self.connected:
                    break
                try:
                    # Use reqCurrentTime — lightweight, no subscription accumulation.
                    # reqAccountSummary creates persistent subscriptions that stack
                    # up and hit IBKR's Error 322 limit.
                    await self.ib.reqCurrentTimeAsync()
                    logger.debug("Heartbeat OK")
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")
                    self._connected = False
                    break
        except asyncio.CancelledError:
            pass

    # ========== REAL-TIME STREAMING ==========

    async def subscribe_market_data(self, symbol: str, delayed: bool = False) -> Optional[Any]:
        """
        Subscribe to real-time market data.

        Args:
            symbol: Stock symbol
            delayed: If True, request delayed data (free). If False (default), request live data.

        Returns:
            Ticker object for accessing bid/ask/last, or None on error
        """
        if not await self._ensure_connected():
            return None

        if symbol in self._tickers:
            return self._tickers[symbol]

        try:
            # Set market data type: 1=Live, 3=Delayed, 4=Delayed Frozen
            if delayed:
                self.ib.reqMarketDataType(3)  # Delayed data (free)
            else:
                self.ib.reqMarketDataType(1)  # Live data (requires subscription)

            contract = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(contract)

            ticker = self.ib.reqMktData(contract, '', False, False)
            self._tickers[symbol] = ticker

            logger.info(f"Subscribed to streaming data for {symbol}")
            return ticker

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return None

    def get_realtime_price(self, ticker) -> Dict[str, Any]:
        """Get current price data from a subscribed ticker."""
        if ticker is None:
            return {'error': 'No ticker'}

        return {
            'bid': ticker.bid if ticker.bid > 0 else None,
            'ask': ticker.ask if ticker.ask > 0 else None,
            'last': ticker.last if ticker.last > 0 else None,
            'volume': ticker.volume if ticker.volume >= 0 else 0,
            'high': ticker.high if ticker.high > 0 else None,
            'low': ticker.low if ticker.low > 0 else None,
        }

    async def unsubscribe_market_data(self, symbol: str) -> bool:
        """Unsubscribe from market data."""
        if symbol in self._tickers:
            try:
                self.ib.cancelMktData(self._tickers[symbol].contract)
                del self._tickers[symbol]
                logger.info(f"Unsubscribed from {symbol}")
                return True
            except Exception as e:
                logger.error(f"Failed to unsubscribe from {symbol}: {e}")
        return False


# ========== FACTORY FUNCTION ==========

def get_ibkr_connector() -> IBKRConnector:
    """Get singleton IBKR connector instance."""
    return IBKRConnector()


# ========== TEST FUNCTION ==========

async def test_connection():
    """Test IBKR connection."""
    connector = get_ibkr_connector()

    print("Testing IBKR Connection...")
    print("=" * 50)

    connected = await connector.connect()
    if not connected:
        print("Failed to connect")
        print("Ensure TWS/Gateway is running and API connections enabled")
        return

    print(f"Connected to account: {connector.account_id}")

    # Account summary
    summary = await connector.get_account_summary()
    if 'error' not in summary:
        print(f"Net Liquidation: ${summary.get('netliquidation', 0):,.2f}")
        print(f"Cash: ${summary.get('totalcashvalue', 0):,.2f}")
        print(f"Available Funds: ${summary.get('availablefunds', 0):,.2f}")

    # Positions
    positions = await connector.get_positions()
    print(f"Positions: {len(positions)}")
    for pos in positions[:5]:
        print(f"   {pos['symbol']}: {pos['quantity']} @ ${pos['avg_cost']:.2f}")

    # Test streaming
    print("\nTesting real-time data for SPY (delayed)...")
    ticker = await connector.subscribe_market_data('SPY', delayed=True)
    if ticker:
        await asyncio.sleep(3)  # Wait longer for delayed data
        price = connector.get_realtime_price(ticker)
        print(f"   Last: ${price.get('last', 'N/A')}")
        print(f"   Bid/Ask: ${price.get('bid', 'N/A')} / ${price.get('ask', 'N/A')}")
        await connector.unsubscribe_market_data('SPY')

    await connector.disconnect()
    print("\nTest complete")


if __name__ == "__main__":
    asyncio.run(test_connection())
