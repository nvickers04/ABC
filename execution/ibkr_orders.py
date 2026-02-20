"""
IBKR Orders Mixin - Order Placement and Management

This module provides all order-related functionality as a mixin class:
- Basic order types (limit, market, stop, stop-limit)
- Bracket orders with OCA protection
- Trailing stops
- Advanced order types (adaptive, midprice, relative, VWAP, TWAP, etc.)
- Order modification and cancellation
- Order state tracking

This mixin is imported by IBKRConnector in ibkr_core.py.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from ib_insync import Order, TagValue
from ib_insync.contract import Stock

logger = logging.getLogger(__name__)


class IBKROrdersMixin:
    """
    Mixin class providing order placement and management methods.

    Must be used with IBKRConnector which provides:
    - self.ib: IB connection instance
    - self._ensure_connected(): Connection check method
    - self._wait_for_fill(): Fill waiting method
    - self._update_account_values(): Account value refresh
    - self._order_states, self._bracket_groups: Order tracking dicts
    - self._order_state_lock: Threading lock for order state
    - self.net_liquidation, self.day_trades_remaining: Account values
    """

    # ========== HELPER METHODS ==========

    async def _prepare_contract(self, symbol: str) -> Optional[Stock]:
        """Prepare and qualify a stock contract. Returns None if not connected."""
        if not await self._ensure_connected():
            return None
        contract = Stock(symbol, 'SMART', 'USD')
        await self.ib.qualifyContractsAsync(contract)
        return contract

    async def _place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        tif: str = 'DAY',
        limit_price: float = None,
        aux_price: float = None,
        order_name: str = None,
        extra_result: Dict = None,
        **order_attrs
    ) -> Dict[str, Any]:
        """
        Generic order placement helper.

        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: IBKR order type string
            tif: Time in force
            limit_price: Limit price (optional)
            aux_price: Auxiliary price for stops (optional)
            order_name: Name for logging (defaults to order_type)
            extra_result: Additional fields for result dict
            **order_attrs: Additional Order attributes to set
        """
        contract = await self._prepare_contract(symbol)
        if contract is None:
            return {'error': 'Not connected'}

        # === INFO: Check for existing orders (warn but don't block) ===
        try:
            existing_orders = await self.get_open_orders()
            same_symbol_orders = [o for o in existing_orders if o.get('symbol') == symbol]
            # Just log it, don't block - agent may be intentionally replacing protection
            if same_symbol_orders:
                order_info = ", ".join([f"{o['action']} {o['quantity']} ({o['order_type']})" for o in same_symbol_orders])
                logger.info(f"Note: {symbol} already has orders: {order_info}")
        except Exception as e:
            logger.warning(f"Could not check existing orders: {e}")

        try:
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = order_type
            order.tif = tif
            order.transmit = True

            if limit_price is not None:
                order.lmtPrice = limit_price
            if aux_price is not None:
                order.auxPrice = aux_price

            for attr, value in order_attrs.items():
                setattr(order, attr, value)

            trade = self.ib.placeOrder(contract, order)

            name = order_name or order_type
            price_info = f" @ ${limit_price:.2f}" if limit_price else ""
            price_info = price_info or (f" @ ${aux_price:.2f}" if aux_price else "")
            logger.info(f"{name} order placed: {action} {quantity} {symbol}{price_info}")

            result = {
                'success': True,
                'filled': False,  # Order placed but not filled yet
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            if limit_price is not None:
                result['limit_price'] = limit_price
            if aux_price is not None:
                result['stop_price'] = aux_price
            if extra_result:
                result.update(extra_result)

            return result

        except Exception as e:
            logger.error(f"{order_type} order failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    async def _place_order_with_fill(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        tif: str,
        limit_price: float = None,
        timeout: float = 2.0,
        order_name: str = None
    ) -> Dict[str, Any]:
        """Place order and wait for fill result (for IOC/FOK)."""
        contract = await self._prepare_contract(symbol)
        if contract is None:
            return {'error': 'Not connected'}

        try:
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = 'LMT'
            order.lmtPrice = limit_price
            order.tif = tif
            order.transmit = True

            trade = self.ib.placeOrder(contract, order)
            fill_result = await self._wait_for_fill(trade, timeout=timeout)

            name = order_name or tif
            logger.info(f"{name} order completed: {action} {symbol}, filled {fill_result['filled_quantity']}/{quantity}")

            return {
                'success': True,
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': tif,
                'limit_price': limit_price,
                'filled': fill_result['filled'],
                'fill_status': fill_result['status'],
                'filled_quantity': fill_result['filled_quantity'],
                'avg_fill_price': fill_result['avg_fill_price'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"{tif} order failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # ========== BASIC ORDER TYPES ==========

    async def place_limit_order(
        self, symbol: str, action: str, quantity: int, limit_price: float, tif: str = 'DAY'
    ) -> Dict[str, Any]:
        """Place a basic limit order."""
        return await self._place_order(
            symbol, action, quantity, 'LMT', tif=tif, limit_price=limit_price,
            extra_result={'tif': tif}
        )

    # ========== BRACKET ORDERS (LONG + SHORT) ==========

    async def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        direction: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        time_bucket: str = 'short_swing'
    ) -> Dict[str, Any]:
        """
        Place entry order + OCA protective orders (stop loss and take profit).

        Uses OCA (One-Cancels-All) instead of parent-linked brackets because:
        1. IOC entry orders complete before children can be linked
        2. OCA orders don't depend on parent order status
        3. More reliable protection that persists after entry fills
        """
        from execution.ibkr_core import BracketGroup
        from execution.order_types import IBKROrderType

        contract = await self._prepare_contract(symbol)
        if contract is None:
            return {'error': 'Not connected'}

        try:

            # Determine actions based on direction
            if direction == 'LONG':
                entry_action = 'BUY'
                exit_action = 'SELL'
            else:  # SHORT
                entry_action = 'SELL'
                exit_action = 'BUY'

            # entry_price already has tolerance applied by upstream validation
            limit_price = entry_price

            # ═══════════════════════════════════════════════
            #  HARD RISK GUARD — configurable via RISK_PER_TRADE
            # ═══════════════════════════════════════════════
            from core.config import RISK_PER_TRADE
            await self._update_account_values()
            risk_per_share = abs(entry_price - stop_price)
            risk_dollars = risk_per_share * quantity
            max_risk = RISK_PER_TRADE * self.net_liquidation
            if self.net_liquidation > 0 and risk_dollars > max_risk:
                msg = (f"RISK GUARD BLOCKED: {symbol} risk ${risk_dollars:,.2f} "
                       f"> {RISK_PER_TRADE*100}% of equity ${max_risk:,.2f} "
                       f"(NLV=${self.net_liquidation:,.2f})")
                logger.error(msg)
                return {
                    'success': False,
                    'filled': False,
                    'reason': msg,
                    'symbol': symbol,
                    'direction': direction
                }

            # PDT CHECK: Account under $25k has restrictions
            # Note: PDT only applies to intraday trades (same-day round trips)
            # short_swing and swing trades (hold overnight) are NOT affected
            if self.net_liquidation < 25000:
                print(f"\n[!] PDT Account (${self.net_liquidation:,.0f} < $25k)")
                print(f"   Day trades remaining: {self.day_trades_remaining}")
                print(f"   Time bucket: {time_bucket}")

                if time_bucket == 'intraday' and self.day_trades_remaining <= 0:
                    logger.error(f"PDT BLOCKED: No day trades remaining for intraday {symbol}")
                    print("   [BLOCKED] Cannot execute intraday trade with 0 day trades")
                    print("   TIP: Use short_swing or swing trades (hold overnight)")
                    return {
                        'success': False,
                        'filled': False,
                        'reason': 'PDT restriction - no day trades remaining for intraday trade',
                        'symbol': symbol,
                        'direction': direction
                    }
                elif time_bucket in ('short_swing', 'swing'):
                    print(f"   [OK] {time_bucket} trade OK - will hold overnight (not a day trade)")

            # HARD CASH GUARD — prevent margin buying in cash-only account
            # AvailableFunds on IBKR paper includes margin; use TotalCashValue
            total_cost = entry_price * quantity
            if self.cash_value > 0 and direction == 'LONG' and total_cost > self.cash_value:
                msg = (f"CASH GUARD BLOCKED: {symbol} costs ${total_cost:,.2f} "
                       f"but only ${self.cash_value:,.2f} cash available. "
                       f"No margin buying allowed in cash-only account.")
                logger.error(msg)
                return {
                    'success': False,
                    'filled': False,
                    'reason': msg,
                    'symbol': symbol,
                    'direction': direction
                }

            # STEP 1: Place entry order
            entry_order = Order()
            entry_order.action = entry_action
            entry_order.totalQuantity = quantity
            entry_order.orderType = 'LMT'
            entry_order.lmtPrice = limit_price
            entry_order.tif = 'DAY'  # DAY order gives time to fill
            entry_order.transmit = True

            # Log order details
            print(f"\n[ORDER] {entry_action} {quantity} {symbol} @ ${limit_price:.2f}")

            logger.info(f"Placing LIMIT {entry_action} {quantity} {symbol} @ ${limit_price:.2f}")

            entry_trade = self.ib.placeOrder(contract, entry_order)
            entry_id = entry_trade.order.orderId

            # Wait for entry fill confirmation
            fill_result = await self._wait_for_fill(entry_trade, timeout=30.0)

            if not fill_result['filled']:
                # Order didn't fill in time - cancel it
                logger.warning(f"Entry NOT FILLED: {symbol} - {fill_result['status']}")
                try:
                    if hasattr(self, "_cancel_order_with_tracking"):
                        self._cancel_order_with_tracking(entry_trade.order, source="bracket_entry_timeout")
                    else:
                        self.ib.cancelOrder(entry_trade.order)
                except Exception:
                    pass  # Order may already be cancelled
                return {
                    'success': False,
                    'filled': False,
                    'reason': f"Entry order {fill_result['status']} - no fill at ${limit_price:.2f}",
                    'symbol': symbol,
                    'direction': direction
                }

            # STEP 2: Entry filled - now place OCA protective orders
            actual_fill_price = fill_result['avg_fill_price']
            filled_qty = fill_result['filled_quantity']

            # Recalculate stop/target based on ACTUAL fill price
            # Preserve the PERCENTAGE distances from the original analysis
            if entry_price <= 0:
                logger.error(f"Invalid entry_price {entry_price} - using original stop/target")
                adjusted_stop = stop_price
                adjusted_target = target_price
            elif direction == 'LONG':
                stop_pct = (entry_price - stop_price) / entry_price  # e.g., 2.2% risk
                target_pct = (target_price - entry_price) / entry_price  # e.g., 6.6% reward
                adjusted_stop = round(actual_fill_price * (1 - stop_pct), 2)
                adjusted_target = round(actual_fill_price * (1 + target_pct), 2)
            else:  # SHORT
                stop_pct = (stop_price - entry_price) / entry_price
                target_pct = (entry_price - target_price) / entry_price
                adjusted_stop = round(actual_fill_price * (1 + stop_pct), 2)
                adjusted_target = round(actual_fill_price * (1 - target_pct), 2)

            logger.info(f"Entry FILLED: {direction} {filled_qty} {symbol} @ ${actual_fill_price:.2f}")
            logger.info(f"Placing OCA protection: stop=${adjusted_stop:.2f}, target=${adjusted_target:.2f}")

            # Create unique OCA group to link stop and target
            oca_group = f"OCA_{symbol}_{entry_id}_{int(datetime.now().timestamp())}"

            # Stop loss order (GTC, OCA-linked)
            stop_order = Order()
            stop_order.action = exit_action
            stop_order.totalQuantity = filled_qty
            stop_order.orderType = IBKROrderType.STOP.value
            stop_order.auxPrice = adjusted_stop
            stop_order.tif = 'GTC'
            stop_order.ocaGroup = oca_group
            stop_order.ocaType = 1  # Cancel on fill
            stop_order.transmit = True

            stop_trade = self.ib.placeOrder(contract, stop_order)
            stop_id = stop_trade.order.orderId

            # Brief wait to ensure stop order is accepted before placing target
            await asyncio.sleep(0.1)

            # Check if stop order was rejected before placing target
            if stop_trade.orderStatus.status in ('ApiCancelled', 'Cancelled', 'Error'):
                logger.error(f"Stop order failed for bracket {symbol}: {stop_trade.orderStatus.status}")
                # Entry already filled - log critical warning
                logger.critical(f"POSITION AT RISK: Entry filled but stop failed for {symbol}!")
                return {
                    'success': False,
                    'filled': True,  # Entry DID fill
                    'entry_price': actual_fill_price,
                    'quantity': filled_qty,
                    'error': f"Stop order failed: {stop_trade.orderStatus.status}",
                    'symbol': symbol,
                    'warning': 'Position is unprotected - manual stop required!'
                }

            # Take profit order (GTC, OCA-linked)
            target_order = Order()
            target_order.action = exit_action
            target_order.totalQuantity = filled_qty
            target_order.orderType = 'LMT'
            target_order.lmtPrice = adjusted_target
            target_order.tif = 'GTC'
            target_order.ocaGroup = oca_group
            target_order.ocaType = 1  # Cancel on fill
            target_order.transmit = True

            target_trade = self.ib.placeOrder(contract, target_order)
            target_id = target_trade.order.orderId

            # Brief wait and verify target order was accepted
            await asyncio.sleep(0.1)

            if target_trade.orderStatus.status in ('ApiCancelled', 'Cancelled', 'Error'):
                # Target failed - BUT KEEP THE STOP (position is protected)
                logger.error(f"Target order failed for bracket {symbol}: {target_trade.orderStatus.status}")
                logger.warning(f"Stop order {stop_id} remains active - position is protected")
                # Continue with partial success - stop is more important than target

            logger.info(f"OCA protection placed: stop_id={stop_id}, target_id={target_id}, oca_group={oca_group}")

            # Track bracket group for monitoring
            bracket_group_obj = BracketGroup(
                group_id=f"BRK_{symbol}_{entry_id}",
                symbol=symbol,
                direction=direction,
                entry_order_id=entry_id,
                stop_order_id=stop_id,
                target_order_id=target_id,
                oca_group=oca_group,
                entry_price=actual_fill_price,
                stop_price=adjusted_stop,
                target_price=adjusted_target,
                quantity=filled_qty,
                status='active'
            )
            with self._order_state_lock:
                self._bracket_groups[bracket_group_obj.group_id] = bracket_group_obj

            return {
                'success': True,
                'filled': True,
                'bracket_order_id': entry_id,
                'bracket_group_id': bracket_group_obj.group_id,
                'stop_order_id': stop_id,
                'target_order_id': target_id,
                'oca_group': oca_group,
                'symbol': symbol,
                'quantity': filled_qty,
                'direction': direction,
                'entry_price': actual_fill_price,
                'stop_price': adjusted_stop,
                'target_price': adjusted_target,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Bracket order failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # ========== OCA PROTECTIVE ORDERS ==========

    async def place_oca(
        self,
        symbol: str,
        quantity: int,
        direction: str,
        stop_price: float,
        target_price: float
    ) -> Dict[str, Any]:
        """
        Place OCA (One-Cancels-All) protective orders.
        When stop or target fills, the other is cancelled.
        Use this to add protection to an existing position.
        """
        from execution.order_types import IBKROrderType

        contract = await self._prepare_contract(symbol)
        if contract is None:
            return {'error': 'Not connected'}

        try:
            oca_group = f"OCA_{symbol}_{int(datetime.now().timestamp())}"
            exit_action = 'SELL' if direction == 'LONG' else 'BUY'

            # Stop order
            stop_order = Order()
            stop_order.action = exit_action
            stop_order.totalQuantity = quantity
            stop_order.orderType = IBKROrderType.STOP.value
            stop_order.auxPrice = stop_price
            stop_order.tif = 'GTC'
            stop_order.ocaGroup = oca_group
            stop_order.ocaType = 1
            stop_order.transmit = True

            # Target order
            target_order = Order()
            target_order.action = exit_action
            target_order.totalQuantity = quantity
            target_order.orderType = IBKROrderType.LIMIT.value
            target_order.lmtPrice = target_price
            target_order.tif = 'GTC'
            target_order.ocaGroup = oca_group
            target_order.ocaType = 1
            target_order.transmit = True

            stop_trade = self.ib.placeOrder(contract, stop_order)
            stop_id = stop_trade.order.orderId

            # Brief wait to ensure stop order is accepted before placing target
            await asyncio.sleep(0.1)

            # Check if stop order was rejected before placing target
            if stop_trade.orderStatus.status in ('ApiCancelled', 'Cancelled', 'Error'):
                logger.error(f"Stop order failed for {symbol}: {stop_trade.orderStatus.status}")
                return {
                    'success': False,
                    'error': f"Stop order failed: {stop_trade.orderStatus.status}",
                    'symbol': symbol
                }

            target_trade = self.ib.placeOrder(contract, target_order)
            target_id = target_trade.order.orderId

            # Brief wait and verify target order was accepted
            await asyncio.sleep(0.1)

            if target_trade.orderStatus.status in ('ApiCancelled', 'Cancelled', 'Error'):
                # Target failed - MUST cancel stop to avoid orphaned order
                logger.error(f"Target order failed for {symbol}: {target_trade.orderStatus.status}")
                logger.warning(f"Cancelling orphaned stop order {stop_id} for {symbol}")
                try:
                    if hasattr(self, "_cancel_order_with_tracking"):
                        self._cancel_order_with_tracking(stop_order, source="oca_target_failed")
                    else:
                        self.ib.cancelOrder(stop_order)
                except Exception as cancel_err:
                    logger.error(f"Failed to cancel orphaned stop {stop_id}: {cancel_err}")

                return {
                    'success': False,
                    'error': f"Target order failed: {target_trade.orderStatus.status}, stop cancelled",
                    'symbol': symbol,
                    'cancelled_stop_id': stop_id
                }

            logger.info(f"OCA orders placed for {symbol}: stop=${stop_price:.2f} (id={stop_id}), target=${target_price:.2f} (id={target_id})")

            return {
                'success': True,
                'oca_group': oca_group,
                'stop_order_id': stop_trade.order.orderId,
                'target_order_id': target_trade.order.orderId,
                'symbol': symbol,
                'quantity': quantity,
                'direction': direction,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"OCA orders failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # Alias for backward compatibility
    place_oca_orders = place_oca

    # ========== TRAILING STOP ==========

    async def place_trailing_stop(
        self, symbol: str, quantity: int, direction: str,
        trail_amount: float = None, trail_percent: float = None
    ) -> Dict[str, Any]:
        """Place trailing stop order with fixed amount or percentage."""
        from execution.order_types import IBKROrderType

        if trail_amount is None and trail_percent is None:
            return {'error': 'Must provide trail_amount or trail_percent'}

        action = 'SELL' if direction == 'LONG' else 'BUY'
        order_attrs = {}
        if trail_percent:
            # Normalize: if < 1, assume decimal (0.08 = 8%); otherwise already a %
            pct = trail_percent * 100 if trail_percent < 1 else trail_percent
            pct = max(0.1, min(pct, 99.0))  # IBKR requires (0, 100)
            order_attrs['trailingPercent'] = pct

        return await self._place_order(
            symbol, action, quantity, IBKROrderType.TRAIL.value, tif='GTC',
            aux_price=trail_amount if not trail_percent else None,
            order_name='Trailing stop',
            extra_result={'direction': direction, 'trail_amount': trail_amount, 'trail_percent': trail_percent},
            **order_attrs
        )

    # ========== SIMPLE STOP ORDER ==========

    async def place_stop_order(
        self, symbol: str, action: str, quantity: int, stop_price: float
    ) -> Dict[str, Any]:
        """Place a simple stop order for position protection."""
        from execution.order_types import IBKROrderType
        return await self._place_order(
            symbol, action, quantity, IBKROrderType.STOP.value, tif='GTC', aux_price=stop_price
        )

    # ========== STOP-LIMIT ORDER ==========

    async def place_stop_limit(
        self, symbol: str, action: str, quantity: int,
        stop_price: float, limit_price: float, tif: str = 'GTC'
    ) -> Dict[str, Any]:
        """Place a stop-limit order. Triggers at stop_price, executes at limit_price."""
        from execution.ibkr_core import OrderState, OrderStatus

        contract = await self._prepare_contract(symbol)
        if contract is None:
            return {'error': 'Not connected'}

        try:
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = 'STP LMT'
            order.auxPrice = stop_price
            order.lmtPrice = limit_price
            order.tif = tif
            order.transmit = True

            trade = self.ib.placeOrder(contract, order)

            # Track order state
            with self._order_state_lock:
                self._order_states[trade.order.orderId] = OrderState(
                    order_id=trade.order.orderId, symbol=symbol, action=action,
                    quantity=quantity, order_type='STP LMT',
                    stop_price=stop_price, limit_price=limit_price, status=OrderStatus.SUBMITTED
                )

            logger.info(f"Stop-limit order placed: {action} {quantity} {symbol} stop=${stop_price:.2f} limit=${limit_price:.2f}")
            return {
                'success': True, 'order_id': trade.order.orderId, 'symbol': symbol,
                'action': action, 'quantity': quantity, 'stop_price': stop_price,
                'limit_price': limit_price, 'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Stop-limit order failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # Alias for backward compatibility
    place_stop_limit_order = place_stop_limit

    # ========== MARKET ORDER ==========

    async def place_market_order(
        self, symbol: str, action: str, quantity: int,
        wait_for_fill: bool = True, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Place a market order with optional fill waiting."""
        contract = await self._prepare_contract(symbol)
        if contract is None:
            return {'error': 'Not connected'}

        try:
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = 'MKT'
            order.tif = 'DAY'
            order.transmit = True

            logger.info(f"Placing MARKET {action} {quantity} {symbol}")
            trade = self.ib.placeOrder(contract, order)

            result = {
                'success': True, 'order_id': trade.order.orderId, 'symbol': symbol,
                'action': action, 'quantity': quantity, 'order_type': 'MKT',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            if wait_for_fill:
                fill_result = await self._wait_for_fill(trade, timeout=timeout)
                result.update({
                    'filled': fill_result['filled'], 'fill_status': fill_result['status'],
                    'avg_fill_price': fill_result['avg_fill_price'],
                    'filled_quantity': fill_result['filled_quantity']
                })
                if fill_result['filled']:
                    logger.info(f"Market order FILLED: {action} {fill_result['filled_quantity']} {symbol} @ ${fill_result['avg_fill_price']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Market order failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # ========== MODIFY ORDER ==========

    async def modify_stop_price(
        self,
        order_id: int,
        new_stop_price: float
    ) -> Dict[str, Any]:
        """
        Modify the stop price of an existing stop order.

        Use for adjusting stops as position moves in your favor.

        Args:
            order_id: ID of order to modify
            new_stop_price: New stop price

        Returns:
            Dict with modification result
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            # Find the order
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    # Modify the stop price
                    trade.order.auxPrice = new_stop_price
                    self.ib.placeOrder(trade.contract, trade.order)

                    logger.info(f"Modified stop order {order_id} to ${new_stop_price:.2f}")

                    return {
                        'success': True,
                        'order_id': order_id,
                        'new_stop_price': new_stop_price,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

            return {'error': f'Order {order_id} not found'}

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return {'error': str(e)}

    # ========== ORDER STATE TRACKING ==========

    def get_order_state(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Get tracked state for an order."""
        with self._order_state_lock:
            state = self._order_states.get(order_id)
            return state.to_dict() if state else None

    def get_bracket_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get tracked state for a bracket group."""
        with self._order_state_lock:
            group = self._bracket_groups.get(group_id)
            return group.to_dict() if group else None

    def get_active_bracket_groups(self) -> list:
        """Get all active bracket groups."""
        with self._order_state_lock:
            return [
                g.to_dict() for g in self._bracket_groups.values()
                if g.status == 'active'
            ]

    # ========== ADVANCED ORDER TYPES ==========

    async def place_trailing_stop_limit(
        self, symbol: str, quantity: int, direction: str,
        trail_amount: float = None, trail_percent: float = None, limit_offset: float = 0.10
    ) -> Dict[str, Any]:
        """Place trailing stop with limit protection to prevent slippage."""
        if trail_amount is None and trail_percent is None:
            return {'error': 'Must provide trail_amount or trail_percent'}

        action = 'SELL' if direction == 'LONG' else 'BUY'
        order_attrs = {'lmtPriceOffset': limit_offset}
        if trail_percent:
            # Normalize: if < 1, assume decimal (0.08 = 8%); otherwise already a %
            pct = trail_percent * 100 if trail_percent < 1 else trail_percent
            pct = max(0.1, min(pct, 99.0))  # IBKR requires (0, 100)
            order_attrs['trailingPercent'] = pct

        return await self._place_order(
            symbol, action, quantity, 'TRAIL LIMIT', tif='GTC',
            aux_price=trail_amount if not trail_percent else None,
            order_name='Trailing stop limit',
            extra_result={'direction': direction, 'trail_amount': trail_amount,
                         'trail_percent': trail_percent, 'limit_offset': limit_offset},
            **order_attrs
        )

    async def place_market_on_close(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Place Market-on-Close (MOC) order. Executes at closing auction."""
        return await self._place_order(symbol, action, quantity, 'MOC', order_name='MOC')

    async def place_limit_on_close(
        self, symbol: str, action: str, quantity: int, limit_price: float
    ) -> Dict[str, Any]:
        """Place Limit-on-Close (LOC) order. Executes at close if price acceptable."""
        return await self._place_order(symbol, action, quantity, 'LOC', limit_price=limit_price, order_name='LOC')

    async def place_market_on_open(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Place Market-on-Open (MOO) order. Executes at opening auction."""
        return await self._place_order(symbol, action, quantity, 'MKT', tif='OPG', order_name='MOO')

    async def place_limit_on_open(
        self, symbol: str, action: str, quantity: int, limit_price: float
    ) -> Dict[str, Any]:
        """Place Limit-on-Open (LOO) order. Participates in opening auction with price limit."""
        return await self._place_order(symbol, action, quantity, 'LMT', tif='OPG', limit_price=limit_price, order_name='LOO')

    async def place_adaptive(
        self, symbol: str, action: str, quantity: int,
        order_type: str = 'MKT', limit_price: float = None, priority: str = 'Normal'
    ) -> Dict[str, Any]:
        """Place IBKR Adaptive algo order for better execution. Priority: Patient/Normal/Urgent."""
        if order_type == 'LMT' and limit_price is None:
            return {'error': 'Limit price required for LMT adaptive orders'}

        return await self._place_order(
            symbol, action, quantity, order_type, limit_price=limit_price,
            order_name=f'Adaptive {order_type}',
            extra_result={'priority': priority},
            algoStrategy='Adaptive', algoParams=[TagValue('adaptivePriority', priority)]
        )

    # Alias for backward compatibility
    place_adaptive_order = place_adaptive

    async def place_midprice(
        self, symbol: str, action: str, quantity: int, price_cap: float = None
    ) -> Dict[str, Any]:
        """Place Midprice order. Pegs to bid/ask midpoint for better fills."""
        return await self._place_order(
            symbol, action, quantity, 'MIDPRICE',
            limit_price=price_cap, order_name='Midprice',
            extra_result={'price_cap': price_cap} if price_cap else None
        )

    # Alias for backward compatibility
    place_midprice_order = place_midprice

    async def place_relative(
        self, symbol: str, action: str, quantity: int,
        offset: float = 0.01, limit_price: float = None
    ) -> Dict[str, Any]:
        """Place Relative order. Pegs to bid/ask with offset, auto-adjusts."""
        return await self._place_order(
            symbol, action, quantity, 'REL', aux_price=offset, limit_price=limit_price,
            order_name='Relative', extra_result={'offset': offset}
        )

    # Alias for backward compatibility
    place_relative_order = place_relative

    async def place_limit_order_gtd(
        self, symbol: str, action: str, quantity: int,
        limit_price: float, good_till_date: str
    ) -> Dict[str, Any]:
        """Place Good-Till-Date limit order. Format: 'YYYYMMDD HH:MM:SS'."""
        return await self._place_order(
            symbol, action, quantity, 'LMT', tif='GTD', limit_price=limit_price,
            extra_result={'tif': 'GTD', 'good_till_date': good_till_date},
            goodTillDate=good_till_date
        )

    async def place_fill_or_kill(
        self, symbol: str, action: str, quantity: int, limit_price: float
    ) -> Dict[str, Any]:
        """Place Fill-or-Kill (FOK) order. Must fill entirely or cancels."""
        return await self._place_order_with_fill(
            symbol, action, quantity, 'FOK', 'FOK', limit_price, timeout=2.0, order_name='FOK'
        )

    async def place_immediate_or_cancel(
        self, symbol: str, action: str, quantity: int, limit_price: float
    ) -> Dict[str, Any]:
        """Place Immediate-or-Cancel (IOC) order. Fills what it can, cancels rest."""
        return await self._place_order_with_fill(
            symbol, action, quantity, 'IOC', 'IOC', limit_price, timeout=2.0, order_name='IOC'
        )

    async def place_vwap(
        self, symbol: str, action: str, quantity: int,
        start_time: str = None, end_time: str = None, max_pct_volume: float = 25.0
    ) -> Dict[str, Any]:
        """Place VWAP algo order. Executes over time to achieve volume-weighted average."""
        algo_params = [TagValue('maxPctVol', str(max_pct_volume / 100)), TagValue('noTakeLiq', '0')]
        if start_time:
            algo_params.append(TagValue('startTime', start_time))
        if end_time:
            algo_params.append(TagValue('endTime', end_time))

        return await self._place_order(
            symbol, action, quantity, 'MKT', order_name='VWAP',
            extra_result={'max_pct_volume': max_pct_volume, 'start_time': start_time, 'end_time': end_time},
            algoStrategy='Vwap', algoParams=algo_params
        )

    # Alias for backward compatibility
    place_vwap_order = place_vwap

    async def place_twap(
        self, symbol: str, action: str, quantity: int,
        start_time: str = None, end_time: str = None, randomize_pct: float = 55.0
    ) -> Dict[str, Any]:
        """Place TWAP algo order. Spreads execution evenly over time."""
        algo_params = [TagValue('strategyType', 'Marketable'), TagValue('allowPastEndTime', '1')]
        if start_time:
            algo_params.append(TagValue('startTime', start_time))
        if end_time:
            algo_params.append(TagValue('endTime', end_time))

        return await self._place_order(
            symbol, action, quantity, 'MKT', order_name='TWAP',
            extra_result={'start_time': start_time, 'end_time': end_time},
            algoStrategy='Twap', algoParams=algo_params
        )

    # Alias for backward compatibility
    place_twap_order = place_twap

    async def place_iceberg_order(
        self, symbol: str, action: str, total_quantity: int,
        display_size: int, limit_price: float
    ) -> Dict[str, Any]:
        """Place Iceberg order. Shows only display_size, hides rest."""
        return await self._place_order(
            symbol, action, total_quantity, 'LMT', limit_price=limit_price,
            order_name='Iceberg',
            extra_result={'total_quantity': total_quantity, 'display_size': display_size},
            displaySize=display_size
        )

    async def place_snap_to_midpoint(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Place Snap-to-Midpoint order. Sets limit to midpoint at submission."""
        return await self._place_order(symbol, action, quantity, 'SNAP MID', order_name='Snap-to-Mid')
