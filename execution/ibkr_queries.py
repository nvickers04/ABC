"""
IBKR Queries Mixin - Account and Position Queries

This module provides all query-related functionality as a mixin class:
- Account summary and buying power queries
- Position queries with P/L data
- Open order queries
- Order cancellation
- Execution history and P/L calculation

This mixin is imported by IBKRConnector in ibkr_core.py.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class IBKRQueriesMixin:
    """
    Mixin class providing account and position query methods.

    Must be used with IBKRConnector which provides:
    - self.ib: IB connection instance
    - self._ensure_connected(): Connection check method
    - self.async_lock: Async lock for thread safety
    - self.account_id: Account ID string
    - self._executions: Execution tracking dict
    - self._execution_lock: Threading lock for executions
    """

    # ========== CACHED DATA ACCESSORS (sync, no API call) ==========

    def get_cached_account_values(self) -> list:
        """Return cached account-value objects from IB (sync, no API call)."""
        try:
            if not self.ib.isConnected():
                return []
            return list(self.ib.accountValues())
        except Exception:
            return []

    def get_cached_portfolio(self) -> list:
        """Return cached portfolio items from IB (sync, no API call)."""
        try:
            if not self.ib.isConnected():
                return []
            return list(self.ib.portfolio())
        except Exception:
            return []

    def get_cached_trades(self) -> list:
        """Return cached open trades from IB (sync, no API call)."""
        try:
            if not self.ib.isConnected():
                return []
            return list(self.ib.openTrades())
        except Exception:
            return []

    # ========== POSITION QUERIES ==========

    async def refresh_positions(self) -> None:
        """Force refresh position data from TWS.

        The ib.positions() method returns cached data that may become stale.
        This method requests fresh data from TWS and waits for update.
        """
        if not self._connected:
            return

        try:
            async with self.async_lock:
                # Request fresh positions from TWS using ib_insync's async method
                # reqPositionsAsync() properly handles the event loop
                await self.ib.reqPositionsAsync()
                logger.debug("Position data refreshed from TWS")
        except Exception as e:
            logger.warning(f"Position refresh failed: {e}")

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions with P/L data (refreshes from TWS first).
        
        Returns both stock and options positions with full contract details.
        Use sec_type to distinguish: 'STK' for stocks, 'OPT' for options.
        """
        if not await self._ensure_connected():
            return []

        try:
            # Force refresh to get accurate data
            await self.refresh_positions()

            # Use portfolio() instead of positions() to get unrealized P/L
            portfolio_items = self.ib.portfolio()
            positions = []
            
            for item in portfolio_items:
                if item.position == 0:
                    continue
                    
                contract = item.contract
                sec_type = contract.secType or 'STK'
                
                pos_data = {
                    'symbol': contract.symbol,
                    'quantity': item.position,
                    'avg_cost': item.averageCost,
                    'market_value': item.marketValue,
                    'unrealized_pnl': item.unrealizedPNL,
                    'realized_pnl': item.realizedPNL,
                    'market_price': item.marketPrice,
                    'sec_type': sec_type,
                }
                
                # Add options-specific fields
                if sec_type == 'OPT':
                    pos_data.update({
                        'strike': contract.strike,
                        'expiration': contract.lastTradeDateOrContractMonth,
                        'right': contract.right,  # 'C' or 'P'
                        'multiplier': int(contract.multiplier or 100),
                        'local_symbol': contract.localSymbol,
                        'con_id': contract.conId,
                    })
                
                positions.append(pos_data)
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific position by symbol.
        
        Args:
            symbol: Stock symbol to look up
            
        Returns:
            Position dict if found, None otherwise
        """
        positions = await self.get_positions()
        for p in positions:
            if p.get('symbol', '').upper() == symbol.upper():
                return p
        return None

    async def get_stock_positions(self) -> List[Dict[str, Any]]:
        """Get only stock positions (filters out options)."""
        positions = await self.get_positions()
        return [p for p in positions if p.get('sec_type', 'STK') == 'STK']

    async def get_options_positions(self) -> List[Dict[str, Any]]:
        """
        Get options positions with full contract details.
        
        Returns list of options positions including:
        - symbol: Underlying symbol
        - strike, expiration, right: Option contract details
        - quantity: Number of contracts (negative = short)
        - avg_cost: Entry price per contract
        - market_price: Current price
        - unrealized_pnl: Current P/L
        """
        positions = await self.get_positions()
        return [p for p in positions if p.get('sec_type') == 'OPT']

    # ========== ACCOUNT QUERIES ==========

    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary."""
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            async with self.async_lock:
                # Use accountValues() which is already populated from connection
                account_values = self.ib.accountValues()

                result = {'account_id': self.account_id}
                target_tags = {
                    'NetLiquidation',
                    'TotalCashValue',
                    'AvailableFunds',
                    'DailyPnL',
                    'UnrealizedPnL',
                    'RealizedPnL'
                }

                for av in account_values:
                    if av.tag in target_tags and av.currency == 'USD':
                        result[av.tag.lower()] = float(av.value)

                return result

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {'error': str(e)}

    async def get_available_funds(self) -> float:
        """Get available funds (cash only, no margin)."""
        summary = await self.get_account_summary()
        return summary.get('availablefunds', 0.0)

    # ========== ORDER MANAGEMENT ==========

    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """Cancel an open order."""
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            # Find the trade by order ID
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    if hasattr(self, "_cancel_order_with_tracking"):
                        self._cancel_order_with_tracking(trade.order, source="cancel_order")
                    else:
                        self.ib.cancelOrder(trade.order)
                    return {'success': True, 'order_id': order_id}

            return {'error': f'Order {order_id} not found'}

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {'error': str(e)}

    async def cancel_stops(self, symbol: str, exclude_order_ids: Optional[set[int]] = None) -> Dict[str, Any]:
        """
        Cancel all stop orders for a specific symbol.
        
        This includes STP, STP LMT, TRAIL, and TRAIL LIMIT orders.
        Used by order_executor when adjusting stops or moving to breakeven.
        
        Handles cross-client orders: tries cancelOrder first, then verifies.
        Orders from other clientIds may fail with error 10147 (async), so
        we verify actual cancellation after a wait period.
        
        Args:
            symbol: Stock ticker to cancel stops for
            exclude_order_ids: Optional set of order IDs to preserve
            
        Returns:
            Dict with 'cancelled' count and list of 'order_ids'
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected', 'cancelled': 0}

        try:
            exclude_order_ids = exclude_order_ids or set()
            stop_order_types = {'STP', 'STP LMT', 'TRAIL', 'TRAIL LIMIT'}
            attempted_ids = []
            
            # Request ALL open orders from broker (all client sessions)
            await self.ib.reqAllOpenOrdersAsync()
            await asyncio.sleep(0.5)
            
            # Collect matching stop orders
            targets = []
            for trade in self.ib.openTrades():
                if (trade.contract.symbol == symbol and 
                    trade.order.orderType in stop_order_types):
                    if trade.order.orderId in exclude_order_ids:
                        continue
                    targets.append(trade)
            
            if not targets:
                return {'success': True, 'cancelled': 0, 'order_ids': [], 'note': f'No stop orders found for {symbol}'}
            
            # Attempt cancellation of each
            for trade in targets:
                try:
                    if hasattr(self, "_cancel_order_with_tracking"):
                        self._cancel_order_with_tracking(trade.order, source="cancel_stops")
                    else:
                        self.ib.cancelOrder(trade.order)
                    attempted_ids.append(trade.order.orderId)
                    logger.info(f"Cancel requested: {trade.order.orderType} #{trade.order.orderId} for {symbol} (clientId={trade.order.clientId})")
                except Exception as cancel_err:
                    logger.warning(f"Cancel call failed for #{trade.order.orderId}: {cancel_err}")
            
            # Wait for cancellations to process (error 10147 comes async)
            await asyncio.sleep(1.5)
            
            # Verify: re-request all orders and check which are actually gone
            await self.ib.reqAllOpenOrdersAsync()
            await asyncio.sleep(0.5)
            
            still_open = set()
            for trade in self.ib.openTrades():
                if (trade.contract.symbol == symbol and 
                    trade.order.orderType in stop_order_types and
                    trade.orderStatus.status in ('PreSubmitted', 'Submitted', 'PendingSubmit')):
                    still_open.add(trade.order.orderId)
            
            actually_cancelled = [oid for oid in attempted_ids if oid not in still_open]
            failed_ids = [oid for oid in attempted_ids if oid in still_open]
            
            result = {
                'success': len(failed_ids) == 0,
                'cancelled': len(actually_cancelled),
                'order_ids': actually_cancelled,
            }
            if failed_ids:
                result['failed_ids'] = failed_ids
                result['warning'] = f"{len(failed_ids)} orders could not be cancelled (likely from another client session). Use TWS to cancel manually or use cancel_all_orphans."
                logger.warning(f"cancel_stops({symbol}): {len(failed_ids)} orders survived: {failed_ids}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel stops for {symbol}: {e}")
            return {'error': str(e), 'cancelled': 0}

    async def cancel_all_orphans(self, exclude_symbols: set = None) -> Dict[str, Any]:
        """
        Cancel ALL orphan orders (orders for symbols with no matching position).
        
        Includes ALL order types (LMT, MKT, STP, TRAIL, MIDPRICE, etc.),
        not just stops. Orphan LMT/MIDPRICE orders are especially dangerous
        because they can fill and create unwanted positions.
        
        Handles cross-client orders with verification.
        
        Args:
            exclude_symbols: Optional set of symbols to skip (e.g. pending
                entries where the fill hasn't arrived yet).
        
        Returns:
            Dict with cancelled/failed counts and details
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected', 'cancelled': 0}
        
        exclude_symbols = exclude_symbols or set()

        try:
            # Get current positions
            positions = await self.get_positions()
            symbols_with_position = {p['symbol'] for p in positions if p.get('quantity', 0) != 0}
            
            # Combine positions + pending entries = "not orphans"
            known_symbols = symbols_with_position | exclude_symbols
            
            # Get ALL open orders
            await self.ib.reqAllOpenOrdersAsync()
            await asyncio.sleep(0.5)
            
            ACTIVE_STATUSES = {'PreSubmitted', 'Submitted', 'PendingSubmit'}
            orphan_trades = []
            
            for trade in self.ib.openTrades():
                symbol = trade.contract.symbol
                status = trade.orderStatus.status
                
                if symbol not in known_symbols and status in ACTIVE_STATUSES:
                    orphan_trades.append(trade)
            
            if not orphan_trades:
                return {'success': True, 'cancelled': 0, 'note': 'No orphan orders found'}
            
            # Cancel ONLY the orphan orders individually — NEVER use reqGlobalCancel
            # reqGlobalCancel kills ALL orders including protective stops!
            for trade in orphan_trades:
                try:
                    if hasattr(self, "_cancel_order_with_tracking"):
                        self._cancel_order_with_tracking(trade.order, source="cancel_all_orphans")
                    else:
                        self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelling orphan order #{trade.order.orderId}: "
                                f"{trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cancel orphan #{trade.order.orderId}: {e}")
            
            attempted = []
            for trade in orphan_trades:
                attempted.append({
                    'order_id': trade.order.orderId,
                    'symbol': trade.contract.symbol,
                    'order_type': trade.order.orderType,
                    'action': trade.order.action,
                    'quantity': trade.order.totalQuantity,
                    'client_id': trade.order.clientId,
                })
                logger.info(f"Cancel orphan: {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol} #{trade.order.orderId} (clientId={trade.order.clientId})")
            
            # Wait for IBKR to process (error 10147 comes async for cross-client)
            await asyncio.sleep(2.0)
            
            # Verify: re-check which are actually gone
            await self.ib.reqAllOpenOrdersAsync()
            await asyncio.sleep(0.5)
            
            still_open_ids = set()
            for trade in self.ib.openTrades():
                if trade.orderStatus.status in ACTIVE_STATUSES:
                    still_open_ids.add(trade.order.orderId)
            
            cancelled = [a for a in attempted if a['order_id'] not in still_open_ids]
            failed = [a for a in attempted if a['order_id'] in still_open_ids]
            
            result = {
                'success': True,
                'cancelled': len(cancelled),
                'failed': len(failed),
                'cancelled_orders': cancelled,
            }
            if failed:
                result['failed_orders'] = failed
                result['warning'] = f"{len(failed)} orders from other client sessions could not be cancelled. Cancel them manually in TWS."
                logger.warning(f"cancel_all_orphans: {len(failed)} orders survived cancellation")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel orphaned orders: {e}")
            return {'error': str(e), 'cancelled': 0}

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders including from other client sessions.
        
        ONLY returns truly active orders that will execute.
        Filters out: Cancelled, Filled, Inactive, PendingCancel
        """
        if not await self._ensure_connected():
            return []

        # These are the only statuses where an order can still execute
        ACTIVE_STATUSES = {'PreSubmitted', 'Submitted', 'PendingSubmit'}

        try:
            # Request all open orders (including from other sessions)
            await self.ib.reqAllOpenOrdersAsync()
            await asyncio.sleep(0.3)  # Give time for orders to arrive

            trades = self.ib.openTrades()
            orders = []
            for t in trades:
                status = t.orderStatus.status
                
                # Skip non-active orders - agent shouldn't see these
                if status not in ACTIVE_STATUSES:
                    continue
                
                # lmtPrice default is huge float (1.7976931348623157e+308) when unset
                lmt_price = t.order.lmtPrice
                if lmt_price > 1e300:  # Unset sentinel value
                    lmt_price = None
                    
                orders.append({
                    'order_id': t.order.orderId,
                    'symbol': t.contract.symbol,
                    'action': t.order.action,
                    'quantity': t.order.totalQuantity,
                    'order_type': t.order.orderType,
                    'aux_price': t.order.auxPrice,
                    'lmt_price': lmt_price,
                    'status': status
                })
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    # ========== EXECUTION QUERIES ==========

    async def get_recent_executions(self) -> List[Dict[str, Any]]:
        """
        Get recent execution fills from IBKR.

        Returns list of executions with actual fill prices.
        Use this to calculate realized P/L accurately.
        """
        if not await self._ensure_connected():
            return []

        try:
            # ib_insync stores executions from the session
            fills = self.ib.fills()

            executions = []
            for fill in fills:
                executions.append({
                    'symbol': fill.contract.symbol,
                    'side': fill.execution.side,  # 'BOT' or 'SLD'
                    'shares': fill.execution.shares,
                    'price': fill.execution.price,
                    'avg_price': fill.execution.avgPrice,
                    'time': fill.execution.time.isoformat() if fill.execution.time else None,
                    'order_id': fill.execution.orderId,
                    'exec_id': fill.execution.execId,
                    'commission': fill.commissionReport.commission if fill.commissionReport else 0
                })

            return executions
        except Exception as e:
            logger.error(f"Failed to get executions: {e}")
            return []

    def get_executions_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all captured executions for a specific symbol.

        This uses the event-driven execution cache, not IBKR API polling.
        Provides accurate fill prices for P/L calculation.
        """
        with self._execution_lock:
            return list(self._executions.get(symbol, []))

    def get_exit_price_for_symbol(self, symbol: str, direction: str) -> Optional[float]:
        """
        Get the actual exit price for a closed position.

        Args:
            symbol: Ticker symbol
            direction: 'LONG' or 'SHORT' - determines which side is the exit

        Returns:
            Weighted average exit price, or None if no exit found
        """
        executions = self.get_executions_for_symbol(symbol)
        if not executions:
            return None

        # For LONG positions, exit is 'SLD' (sold)
        # For SHORT positions, exit is 'BOT' (bought to cover)
        exit_side = 'SLD' if direction == 'LONG' else 'BOT'

        exit_execs = [e for e in executions if e['side'] == exit_side]
        if not exit_execs:
            return None

        # Calculate weighted average exit price
        total_value = sum(e['price'] * e['shares'] for e in exit_execs)
        total_shares = sum(e['shares'] for e in exit_execs)

        return total_value / total_shares if total_shares > 0 else None

    def calculate_realized_pnl(self, symbol: str, direction: str) -> Dict[str, Any]:
        """
        Calculate realized P/L for a symbol using actual execution data.

        Args:
            symbol: Ticker symbol
            direction: 'LONG' or 'SHORT'

        Returns:
            Dict with realized_pnl, entry_price, exit_price, shares, etc.
        """
        executions = self.get_executions_for_symbol(symbol)
        if not executions:
            return {'error': 'No executions found', 'symbol': symbol}

        # Separate entries and exits based on direction
        if direction == 'LONG':
            entry_side, exit_side = 'BOT', 'SLD'
        else:
            entry_side, exit_side = 'SLD', 'BOT'

        entries = [e for e in executions if e['side'] == entry_side]
        exits = [e for e in executions if e['side'] == exit_side]

        if not entries or not exits:
            return {
                'error': 'Incomplete trade data',
                'symbol': symbol,
                'entries': len(entries),
                'exits': len(exits)
            }

        # Calculate weighted average entry and exit prices
        entry_value = sum(e['price'] * e['shares'] for e in entries)
        entry_shares = sum(e['shares'] for e in entries)
        avg_entry = entry_value / entry_shares if entry_shares > 0 else 0

        exit_value = sum(e['price'] * e['shares'] for e in exits)
        exit_shares = sum(e['shares'] for e in exits)
        avg_exit = exit_value / exit_shares if exit_shares > 0 else 0

        # Calculate P/L (use the smaller of entry/exit shares for closed portion)
        closed_shares = min(entry_shares, exit_shares)

        if direction == 'LONG':
            realized_pnl = (avg_exit - avg_entry) * closed_shares
        else:
            realized_pnl = (avg_entry - avg_exit) * closed_shares

        # Total commissions
        total_commission = sum(e.get('commission', 0) for e in entries + exits)
        net_pnl = realized_pnl - total_commission

        return {
            'symbol': symbol,
            'direction': direction,
            'entry_price': round(avg_entry, 4),
            'exit_price': round(avg_exit, 4),
            'shares': closed_shares,
            'gross_pnl': round(realized_pnl, 2),
            'commission': round(total_commission, 2),
            'net_pnl': round(net_pnl, 2),
            'pnl_pct': round((avg_exit / avg_entry - 1) * 100, 2) if direction == 'LONG' and avg_entry > 0 else
                       round((avg_entry / avg_exit - 1) * 100, 2) if avg_exit > 0 else 0,
            'entry_executions': len(entries),
            'exit_executions': len(exits)
        }

    def clear_executions(self, symbol: Optional[str] = None):
        """
        Clear cached executions.

        Args:
            symbol: If provided, clear only that symbol. Otherwise clear all.
        """
        with self._execution_lock:
            if symbol:
                self._executions.pop(symbol, None)
            else:
                self._executions.clear()

    async def get_stop_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all stop orders, optionally filtered by symbol.
        
        Args:
            symbol: If provided, filter to this symbol only
            
        Returns:
            List of stop order dicts
        """
        from execution.order_types import is_stop_order
        
        orders = await self.get_open_orders()
        stop_orders = [
            o for o in orders 
            if is_stop_order(o.get('order_type', ''))
        ]
        
        if symbol:
            stop_orders = [o for o in stop_orders if o.get('symbol') == symbol]
        
        return stop_orders

    async def get_stop_order_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the stop order for a specific symbol.
        
        Returns:
            Stop order dict if found, None otherwise
        """
        stops = await self.get_stop_orders(symbol)
        return stops[0] if stops else None

    async def get_completed_trades(self, api_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get completed trades including from prior sessions.
        
        This uses reqCompletedOrders which survives session restarts,
        unlike fills() which only has current session data.
        
        Args:
            api_only: If True, only return orders placed via API (not TWS UI)
            
        Returns:
            List of completed trade dicts with fills and timestamps
        """
        if not await self._ensure_connected():
            return []

        try:
            # Request completed orders - this includes prior sessions
            # Use async API when available to avoid missing event loop in worker threads.
            if hasattr(self.ib, "reqCompletedOrdersAsync"):
                completed = await self.ib.reqCompletedOrdersAsync(api_only)
            else:
                completed = await asyncio.to_thread(
                    self.ib.reqCompletedOrders, api_only
                )
            
            trades = []
            for trade in completed:
                trade_data = {
                    'symbol': trade.contract.symbol,
                    'sec_type': trade.contract.secType,
                    'order_id': trade.order.orderId,
                    'action': trade.order.action,
                    'quantity': trade.order.totalQuantity,
                    'order_type': trade.order.orderType,
                    'status': trade.orderStatus.status,
                    'filled': trade.orderStatus.filled,
                    'avg_fill_price': trade.orderStatus.avgFillPrice,
                    'fills': [],
                    'log': []
                }
                
                # Extract fill times
                for fill in trade.fills:
                    fill_data = {
                        'time': fill.execution.time.isoformat() if fill.execution.time else None,
                        'side': fill.execution.side,
                        'shares': fill.execution.shares,
                        'price': fill.execution.price,
                        'avg_price': fill.execution.avgPrice,
                        'exec_id': fill.execution.execId
                    }
                    trade_data['fills'].append(fill_data)
                
                # Extract log entries with timestamps
                for entry in trade.log:
                    log_data = {
                        'time': entry.time.isoformat() if entry.time else None,
                        'status': entry.status,
                        'message': entry.message
                    }
                    trade_data['log'].append(log_data)
                
                trades.append(trade_data)
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get completed trades: {e}")
            return []

    async def get_entry_time_for_symbol(
        self, 
        symbol: str, 
        direction: str = 'LONG'
    ) -> Optional[datetime]:
        """
        Get the entry time for a position from IBKR execution data.
        
        Queries both completed orders (survives restarts) and current 
        session fills to find the earliest BUY (for LONG) or SELL 
        (for SHORT) execution for this symbol.
        
        Args:
            symbol: Stock ticker symbol
            direction: 'LONG' or 'SHORT' - determines entry side
            
        Returns:
            datetime of earliest entry execution, or None if not found.
            Returns timezone-aware UTC datetime.
        """
        entry_side = 'BOT' if direction == 'LONG' else 'SLD'
        entry_times: List[datetime] = []
        
        # 1. Check current session fills first (fastest)
        try:
            fills = self.ib.fills()
            for fill in fills:
                if (fill.contract.symbol == symbol and 
                    fill.execution.side == entry_side and
                    fill.execution.time):
                    entry_times.append(fill.execution.time)
        except Exception as e:
            logger.warning(f"Failed to get session fills for {symbol}: {e}")
        
        # 2. Check completed orders (includes prior sessions)
        try:
            completed_trades = await self.get_completed_trades(api_only=False)
            for trade in completed_trades:
                if trade['symbol'] != symbol:
                    continue
                    
                for fill in trade.get('fills', []):
                    if fill.get('side') == entry_side and fill.get('time'):
                        # Parse ISO format back to datetime
                        fill_time = datetime.fromisoformat(fill['time'])
                        entry_times.append(fill_time)
        except Exception as e:
            logger.warning(f"Failed to get completed trades for {symbol}: {e}")
        
        if not entry_times:
            logger.debug(f"No entry executions found for {symbol} ({direction})")
            return None
        
        # Return earliest entry time
        earliest = min(entry_times)
        logger.debug(f"Entry time for {symbol}: {earliest.isoformat()}")
        return earliest
