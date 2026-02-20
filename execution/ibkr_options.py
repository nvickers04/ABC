"""
IBKR Options Mixin - Options Chains and Spreads

This module provides all options-related functionality as a mixin class:
- Options chain queries
- Vertical spreads (bull/bear call/put spreads)
- Iron condors and iron butterflies
- Straddles and strangles
- Butterfly spreads
- Calendar spreads
- Covered calls and cash-secured puts
- Protective puts and collars
- Ratio spreads and jade lizards

This mixin is imported by IBKRConnector in ibkr_core.py.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from ib_insync import Order, Option, ComboLeg, Contract
from ib_insync.contract import Stock

logger = logging.getLogger(__name__)


class IBKROptionsMixin:
    """
    Mixin class providing options chains and spread methods.

    Must be used with IBKRConnector which provides:
    - self.ib: IB connection instance
    - self._ensure_connected(): Connection check method
    """

    # ========== HELPER METHODS ==========

    async def _create_options(self, symbol: str, expiration: str,
                              strikes_and_rights: List[Tuple[float, str]]) -> List[Option]:
        """
        Create and qualify multiple option contracts.

        Args:
            symbol: Underlying symbol
            expiration: Expiration date 'YYYYMMDD'
            strikes_and_rights: List of (strike, right) tuples where right is 'C' or 'P'

        Returns:
            List of qualified Option contracts
        """
        options = [Option(symbol, expiration, strike, right, 'SMART')
                   for strike, right in strikes_and_rights]
        await self.ib.qualifyContractsAsync(*options)
        # Validate all legs resolved — conId=0 means contract doesn't exist
        bad = [f"{o.strike}{o.right} {o.lastTradeDateOrContractMonth}" for o in options if o.conId == 0]
        if bad:
            raise ValueError(f"Contract(s) not found at IBKR: {', '.join(bad)}. "
                           f"Check expiration dates — the strike/expiry may not exist.")
        return options

    def _build_combo(self, symbol: str, legs: List[Tuple[int, int, str]]) -> Contract:
        """
        Build a combo (BAG) contract from legs.

        Args:
            symbol: Underlying symbol
            legs: List of (conId, ratio, action) tuples

        Returns:
            Combo Contract ready for order placement
        """
        combo = Contract()
        combo.symbol = symbol
        combo.secType = 'BAG'
        combo.currency = 'USD'
        combo.exchange = 'SMART'
        combo.comboLegs = [
            ComboLeg(conId=con_id, ratio=ratio, action=action, exchange='SMART')
            for con_id, ratio, action in legs
        ]
        return combo

    async def _place_combo_order(
        self,
        symbol: str,
        expiration: str,
        strikes_rights_actions: List[Tuple[float, str, str, int]],
        quantity: int,
        combo_action: str = 'BUY',
        limit_price: Optional[float] = None,
        strategy_name: str = 'Combo'
    ) -> Dict[str, Any]:
        """
        Generic combo order placement - all multi-leg strategies use this.

        Args:
            symbol: Underlying symbol
            expiration: Expiration date 'YYYYMMDD'
            strikes_rights_actions: List of (strike, right, action, ratio) tuples
            quantity: Number of spreads
            combo_action: 'BUY' or 'SELL' for the combo
            limit_price: Limit price (positive=debit, negative=credit)
            strategy_name: Name for logging

        Returns:
            Dict with order result
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            # Create and qualify all options
            options = await self._create_options(
                symbol, expiration,
                [(s, r) for s, r, _, _ in strikes_rights_actions]
            )

            # Build combo legs
            legs = [
                (opt.conId, ratio, action)
                for opt, (_, _, action, ratio) in zip(options, strikes_rights_actions)
            ]
            combo = self._build_combo(symbol, legs)

            # Create and place order
            order = Order()
            order.action = combo_action
            order.totalQuantity = quantity
            order.orderType = 'LMT' if limit_price is not None else 'MKT'
            if limit_price is not None:
                order.lmtPrice = abs(limit_price)
            order.tif = 'DAY'
            order.transmit = True

            trade = self.ib.placeOrder(combo, order)

            strikes_str = '/'.join(str(s) for s, _, _, _ in strikes_rights_actions)
            logger.info(f"{strategy_name}: {symbol} {strikes_str} x{quantity}")

            return {
                'success': True,
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'strategy': strategy_name,
                'expiration': expiration,
                'quantity': quantity,
                'limit_price': limit_price,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"{strategy_name} failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    async def _place_single_option(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        action: str,
        quantity: int,
        limit_price: Optional[float] = None,
        strategy_name: str = 'Option'
    ) -> Dict[str, Any]:
        """Place a single option order (covered call, CSP, protective put)."""
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            opt = Option(symbol, expiration, strike, right, 'SMART')
            await self.ib.qualifyContractsAsync(opt)
            if opt.conId == 0:
                return {'error': f'Contract not found: {symbol} {strike}{right} {expiration}. '
                                 f'Check that expiration date exists for this symbol.'}

            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = 'LMT' if limit_price else 'MKT'
            if limit_price:
                order.lmtPrice = limit_price
            order.tif = 'DAY'
            order.transmit = True

            trade = self.ib.placeOrder(opt, order)
            logger.info(f"{strategy_name}: {action} {quantity} {symbol} {strike}{right} {expiration}")

            return {
                'success': True,
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'strategy': strategy_name,
                'expiration': expiration,
                'strike': strike,
                'contracts': quantity,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"{strategy_name} failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # ========== OPTIONS CHAIN QUERIES ==========

    async def qualify_option_contract(
        self, symbol: str, expiration: str, strike: float, right: str
    ) -> Optional[Any]:
        """
        Create and qualify a single option contract.

        Returns the qualified contract object (opaque to callers outside layer 1)
        or None on failure.
        """
        if not await self._ensure_connected():
            return None
        try:
            contract = Option(symbol, expiration, float(strike), right.upper(), 'SMART')
            await self.ib.qualifyContractsAsync(contract)
            return contract
        except Exception as e:
            logger.error(f"Failed to qualify option {symbol} {expiration} {strike} {right}: {e}")
            return None

    async def get_option_chain(self, symbol: str, min_dte: int = 7, max_dte: int = 45) -> Dict[str, Any]:
        """Get option chain for a symbol with DTE filtering."""
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            stock = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(stock)
            chains = await self.ib.reqSecDefOptParamsAsync(stock.symbol, '', stock.secType, stock.conId)

            if not chains:
                return {'error': f'No option chains found for {symbol}'}

            chain = chains[0]
            today = datetime.now().date()
            valid_expirations = []

            for exp in chain.expirations:
                exp_date = datetime.strptime(exp, '%Y%m%d').date()
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    valid_expirations.append({'expiration': exp, 'dte': dte})

            return {
                'symbol': symbol,
                'exchange': chain.exchange,
                'expirations': valid_expirations,
                'strikes': sorted(chain.strikes),
                'multiplier': chain.multiplier
            }
        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol}: {e}")
            return {'error': str(e)}

    # ========== VERTICAL SPREADS ==========

    async def place_vertical_spread(
        self, symbol: str, expiration: str, long_strike: float, short_strike: float,
        right: str, quantity: int = 1, order_type: str = 'LMT', limit_price: float = None
    ) -> Dict[str, Any]:
        """
        Place a vertical spread (bull/bear call or put spread).
        Bull = long_strike < short_strike, Bear = long_strike > short_strike
        """
        spread_type = 'Bull' if long_strike < short_strike else 'Bear'
        strategy = f"{spread_type} {'Call' if right == 'C' else 'Put'} Spread"

        result = await self._place_combo_order(
            symbol, expiration,
            [(long_strike, right, 'BUY', 1), (short_strike, right, 'SELL', 1)],
            quantity, 'BUY', limit_price, strategy
        )
        if 'success' in result:
            result.update({'long_strike': long_strike, 'short_strike': short_strike, 'right': right})
        return result

    # ========== IRON CONDOR ==========

    async def place_iron_condor(
        self, symbol: str, expiration: str,
        put_long_strike: float, put_short_strike: float,
        call_short_strike: float, call_long_strike: float,
        quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """Place an Iron Condor. Profit if underlying stays between short strikes."""
        if not (put_long_strike < put_short_strike < call_short_strike < call_long_strike):
            return {'error': 'Invalid strike order: put_long < put_short < call_short < call_long'}

        result = await self._place_combo_order(
            symbol, expiration,
            [
                (put_long_strike, 'P', 'BUY', 1),
                (put_short_strike, 'P', 'SELL', 1),
                (call_short_strike, 'C', 'SELL', 1),
                (call_long_strike, 'C', 'BUY', 1),
            ],
            quantity, 'SELL', limit_price, 'Iron Condor'
        )
        if 'success' in result:
            width = min(put_short_strike - put_long_strike, call_long_strike - call_short_strike)
            result.update({
                'put_long_strike': put_long_strike, 'put_short_strike': put_short_strike,
                'call_short_strike': call_short_strike, 'call_long_strike': call_long_strike,
                'wing_width': width,
                'max_profit': limit_price * 100 * quantity if limit_price else None,
                'max_loss': (width - (limit_price or 0)) * 100 * quantity
            })
        return result

    # ========== IRON BUTTERFLY ==========

    async def place_iron_butterfly(
        self, symbol: str, expiration: str, center_strike: float, wing_width: float,
        quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """Place an Iron Butterfly (ATM). Max profit at center strike."""
        result = await self._place_combo_order(
            symbol, expiration,
            [
                (center_strike - wing_width, 'P', 'BUY', 1),
                (center_strike, 'P', 'SELL', 1),
                (center_strike, 'C', 'SELL', 1),
                (center_strike + wing_width, 'C', 'BUY', 1),
            ],
            quantity, 'SELL', limit_price, 'Iron Butterfly'
        )
        if 'success' in result:
            result.update({'center_strike': center_strike, 'wing_width': wing_width})
        return result

    # ========== STRADDLE ==========

    async def place_straddle(
        self, symbol: str, expiration: str, strike: float,
        quantity: int = 1, action: str = 'BUY', limit_price: float = None
    ) -> Dict[str, Any]:
        """Place a Straddle (ATM call + put at same strike)."""
        strategy = 'Long Straddle' if action == 'BUY' else 'Short Straddle'
        result = await self._place_combo_order(
            symbol, expiration,
            [(strike, 'C', action, 1), (strike, 'P', action, 1)],
            quantity, 'BUY', limit_price, strategy
        )
        if 'success' in result:
            result['strike'] = strike
        return result

    # ========== STRANGLE ==========

    async def place_strangle(
        self, symbol: str, expiration: str, put_strike: float, call_strike: float,
        quantity: int = 1, action: str = 'BUY', limit_price: float = None
    ) -> Dict[str, Any]:
        """Place a Strangle (OTM call + OTM put at different strikes)."""
        if put_strike >= call_strike:
            return {'error': 'Put strike must be below call strike'}

        strategy = 'Long Strangle' if action == 'BUY' else 'Short Strangle'
        result = await self._place_combo_order(
            symbol, expiration,
            [(call_strike, 'C', action, 1), (put_strike, 'P', action, 1)],
            quantity, 'BUY', limit_price, strategy
        )
        if 'success' in result:
            result.update({'put_strike': put_strike, 'call_strike': call_strike})
        return result

    # ========== BUTTERFLY ==========

    async def place_butterfly(
        self, symbol: str, expiration: str,
        lower_strike: float, middle_strike: float, upper_strike: float,
        right: str = 'C', quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """Place a Butterfly spread. Max profit at middle strike."""
        strategy = f"{'Call' if right == 'C' else 'Put'} Butterfly"
        result = await self._place_combo_order(
            symbol, expiration,
            [
                (lower_strike, right, 'BUY', 1),
                (middle_strike, right, 'SELL', 2),
                (upper_strike, right, 'BUY', 1),
            ],
            quantity, 'BUY', limit_price, strategy
        )
        if 'success' in result:
            result.update({'lower_strike': lower_strike, 'middle_strike': middle_strike, 'upper_strike': upper_strike})
        return result

    # ========== CALENDAR SPREAD ==========

    async def place_calendar_spread(
        self, symbol: str, strike: float, near_expiration: str, far_expiration: str,
        right: str = 'C', quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """Place a Calendar Spread (same strike, different expirations)."""
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            near_opt = Option(symbol, near_expiration, strike, right, 'SMART')
            far_opt = Option(symbol, far_expiration, strike, right, 'SMART')
            await self.ib.qualifyContractsAsync(near_opt, far_opt)

            # Validate both legs resolved
            bad = []
            if near_opt.conId == 0:
                bad.append(f"near leg {strike}{right} {near_expiration}")
            if far_opt.conId == 0:
                bad.append(f"far leg {strike}{right} {far_expiration}")
            if bad:
                return {'error': f"Contract not found: {', '.join(bad)}. "
                                 f"Check expiration dates exist for {symbol}.",
                        'symbol': symbol}

            combo = self._build_combo(symbol, [
                (near_opt.conId, 1, 'SELL'),
                (far_opt.conId, 1, 'BUY'),
            ])

            order = Order()
            order.action = 'BUY'
            order.totalQuantity = quantity
            order.orderType = 'LMT' if limit_price else 'MKT'
            if limit_price:
                order.lmtPrice = abs(limit_price)
            order.tif = 'DAY'
            order.transmit = True

            trade = self.ib.placeOrder(combo, order)
            strategy = f"{'Call' if right == 'C' else 'Put'} Calendar"
            logger.info(f"{strategy}: {symbol} {strike} {near_expiration}/{far_expiration} x{quantity}")

            return {
                'success': True, 'order_id': trade.order.orderId, 'symbol': symbol,
                'strategy': strategy, 'strike': strike,
                'near_expiration': near_expiration, 'far_expiration': far_expiration,
                'quantity': quantity, 'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Calendar spread failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # ========== DIAGONAL SPREAD ==========

    async def place_diagonal_spread(
        self, symbol: str, near_strike: float, far_strike: float,
        near_expiration: str, far_expiration: str,
        right: str = 'C', quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """
        Place a Diagonal Spread (different strikes AND expirations).
        
        Sells the near-term option and buys the far-term option at a different strike.
        Bullish diagonal: far_strike > near_strike (calls), Bearish: far_strike < near_strike
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            near_opt = Option(symbol, near_expiration, near_strike, right, 'SMART')
            far_opt = Option(symbol, far_expiration, far_strike, right, 'SMART')
            await self.ib.qualifyContractsAsync(near_opt, far_opt)

            # Validate both legs resolved
            bad = []
            if near_opt.conId == 0:
                bad.append(f"near leg {near_strike}{right} {near_expiration}")
            if far_opt.conId == 0:
                bad.append(f"far leg {far_strike}{right} {far_expiration}")
            if bad:
                return {'error': f"Contract not found: {', '.join(bad)}. "
                                 f"Check expiration dates exist for {symbol}. "
                                 f"Use options_chain to find valid expirations first.",
                        'symbol': symbol}

            combo = self._build_combo(symbol, [
                (near_opt.conId, 1, 'SELL'),
                (far_opt.conId, 1, 'BUY'),
            ])

            order = Order()
            order.action = 'BUY'
            order.totalQuantity = quantity
            order.orderType = 'LMT' if limit_price else 'MKT'
            if limit_price:
                order.lmtPrice = abs(limit_price)
            order.tif = 'DAY'
            order.transmit = True

            trade = self.ib.placeOrder(combo, order)
            direction = "Bullish" if (right == 'C' and far_strike > near_strike) or (right == 'P' and far_strike < near_strike) else "Bearish"
            strategy = f"{direction} {'Call' if right == 'C' else 'Put'} Diagonal"
            logger.info(f"{strategy}: {symbol} {near_strike}/{far_strike} {near_expiration}/{far_expiration} x{quantity}")

            return {
                'success': True, 'order_id': trade.order.orderId, 'symbol': symbol,
                'strategy': strategy, 'near_strike': near_strike, 'far_strike': far_strike,
                'near_expiration': near_expiration, 'far_expiration': far_expiration,
                'quantity': quantity, 'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Diagonal spread failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    # ========== BUY OPTION (Long Call/Put) ==========

    async def buy_option(
        self, symbol: str, expiration: str, strike: float, right: str,
        quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """
        Buy a call or put option.
        
        Args:
            symbol: Underlying symbol
            expiration: Expiration date 'YYYYMMDD'
            strike: Strike price
            right: 'C' for call, 'P' for put
            quantity: Number of contracts
            limit_price: Limit price (optional, uses MKT if None)
            
        Returns:
            Order result dict
        """
        right = right.upper()
        if right not in ('C', 'P'):
            return {'error': f"Invalid right '{right}', must be 'C' or 'P'"}
        
        strategy = 'Long Call' if right == 'C' else 'Long Put'
        result = await self._place_single_option(
            symbol, expiration, strike, right, 'BUY', quantity, limit_price, strategy
        )
        if 'success' in result:
            result['max_loss'] = 'Premium paid (defined risk)'
        return result

    # ========== COVERED CALL ==========

    async def place_covered_call(self, symbol: str, expiration: str, strike: float, shares: int = 100) -> Dict[str, Any]:
        """Sell covered call on existing stock position."""
        if shares % 100 != 0:
            return {'error': 'Shares must be multiple of 100'}
        return await self._place_single_option(symbol, expiration, strike, 'C', 'SELL', shares // 100, None, 'Covered Call')

    # ========== CASH-SECURED PUT ==========

    async def sell_cash_secured_put(
        self, symbol: str, expiration: str, strike: float,
        contracts: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """Sell cash-secured put. Requires cash to cover assignment."""
        result = await self._place_single_option(symbol, expiration, strike, 'P', 'SELL', contracts, limit_price, 'Cash-Secured Put')
        if 'success' in result:
            result['assignment_cost'] = strike * 100 * contracts
        return result

    # ========== PROTECTIVE PUT ==========

    async def place_protective_put(self, symbol: str, expiration: str, strike: float, shares: int = 100) -> Dict[str, Any]:
        """Buy protective put for existing long position."""
        if shares % 100 != 0:
            return {'error': 'Shares must be multiple of 100'}
        return await self._place_single_option(symbol, expiration, strike, 'P', 'BUY', shares // 100, None, 'Protective Put')

    # ========== COLLAR ==========

    async def place_collar(
        self, symbol: str, expiration: str, put_strike: float, call_strike: float, shares: int = 100
    ) -> Dict[str, Any]:
        """Place a Collar (buy put + sell call on existing stock)."""
        if shares % 100 != 0:
            return {'error': 'Shares must be multiple of 100'}
        if put_strike >= call_strike:
            return {'error': 'Put strike must be below call strike'}

        result = await self._place_combo_order(
            symbol, expiration,
            [(put_strike, 'P', 'BUY', 1), (call_strike, 'C', 'SELL', 1)],
            shares // 100, 'BUY', None, 'Collar'
        )
        if 'success' in result:
            result.update({'put_strike': put_strike, 'call_strike': call_strike, 'contracts': shares // 100})
        return result

    # ========== RATIO SPREAD ==========

    async def place_ratio_spread(
        self, symbol: str, expiration: str, long_strike: float, short_strike: float,
        right: str = 'C', ratio: tuple = (1, 2), quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """Place a Ratio Spread. WARNING: Has unlimited risk on one side."""
        result = await self._place_combo_order(
            symbol, expiration,
            [(long_strike, right, 'BUY', ratio[0]), (short_strike, right, 'SELL', ratio[1])],
            quantity, 'BUY', limit_price, f"{ratio[0]}:{ratio[1]} {'Call' if right == 'C' else 'Put'} Ratio"
        )
        if 'success' in result:
            result.update({
                'long_strike': long_strike, 'short_strike': short_strike,
                'ratio': f"{ratio[0]}:{ratio[1]}", 'warning': 'Unlimited risk on one side'
            })
        return result

    # ========== JADE LIZARD ==========

    async def place_jade_lizard(
        self, symbol: str, expiration: str,
        put_strike: float, call_short_strike: float, call_long_strike: float,
        quantity: int = 1, limit_price: float = None
    ) -> Dict[str, Any]:
        """Place a Jade Lizard (short put + bear call spread)."""
        if call_short_strike >= call_long_strike:
            return {'error': 'Call short < call long required'}

        result = await self._place_combo_order(
            symbol, expiration,
            [
                (put_strike, 'P', 'SELL', 1),
                (call_short_strike, 'C', 'SELL', 1),
                (call_long_strike, 'C', 'BUY', 1),
            ],
            quantity, 'SELL', limit_price, 'Jade Lizard'
        )
        if 'success' in result:
            result.update({
                'put_strike': put_strike,
                'call_short_strike': call_short_strike,
                'call_long_strike': call_long_strike
            })
        return result

    # ========== POSITION MANAGEMENT ==========

    async def close_option_position(
        self,
        symbol: str,
        expiration: str = None,
        strike: float = None,
        right: str = None,
        contract: Contract = None,
        quantity: int = None,
        limit_price: float = None,
        reason: str = ''
    ) -> Dict[str, Any]:
        """
        Close an existing option position.

        Args:
            symbol: Underlying symbol
            expiration: Option expiration YYYYMMDD (if no contract provided)
            strike: Strike price (if no contract provided)
            right: 'C' or 'P' (if no contract provided)
            contract: The option Contract to close (from position)
            quantity: Number of contracts to close
            limit_price: Limit price (optional, uses market if None)
            reason: Reason for closing (for logging)

        Returns:
            Dict with order result
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            # If no contract provided, try to find it from positions
            if contract is None:
                positions = self.ib.positions()
                for pos in positions:
                    c = pos.contract
                    if c.symbol == symbol and c.secType == 'OPT':
                        # If expiration/strike/right provided, match them
                        if expiration and c.lastTradeDateOrContractMonth != expiration:
                            continue
                        if strike and abs(c.strike - strike) > 0.01:
                            continue
                        if right and c.right != right:
                            continue
                        contract = c
                        if quantity is None:
                            quantity = abs(int(pos.position))
                        break

            if contract is None:
                return {'error': f'No option position found for {symbol} {right} {strike} {expiration}'}
            
            # Ensure contract has exchange set (required for orders)
            if not contract.exchange:
                contract.exchange = 'SMART'

            # Determine action based on current position
            # If we're long, we sell to close; if short, we buy to close
            positions = self.ib.positions()
            current_qty = 0
            for pos in positions:
                if pos.contract.conId == contract.conId:
                    current_qty = int(pos.position)
                    break

            action = 'SELL' if current_qty > 0 else 'BUY'
            close_qty = quantity or abs(current_qty)

            # Auto-calculate midpoint if no limit_price provided
            if limit_price is None:
                try:
                    ticker = self.ib.reqMktData(contract, '', snapshot=True, regulatorySnapshot=False)
                    await asyncio.sleep(1)  # Wait for snapshot data
                    if ticker.bid and ticker.bid > 0 and ticker.ask and ticker.ask > 0:
                        limit_price = round((ticker.bid + ticker.ask) / 2, 2)
                        logger.info(f"Auto midpoint for {symbol} {contract.strike}{contract.right}: "
                                   f"bid={ticker.bid} ask={ticker.ask} mid={limit_price}")
                    self.ib.cancelMktData(contract)
                except Exception as e:
                    logger.debug(f"Could not get midpoint for {symbol}, falling back to MKT: {e}")

            # Create order
            order = Order()
            order.action = action
            order.totalQuantity = close_qty
            order.orderType = 'LMT' if limit_price else 'MKT'
            if limit_price:
                order.lmtPrice = limit_price
            order.tif = 'DAY'
            order.transmit = True

            trade = self.ib.placeOrder(contract, order)

            logger.info(f"Close option: {action} {close_qty} {symbol} "
                       f"{contract.strike}{contract.right} {contract.lastTradeDateOrContractMonth} "
                       f"{'LMT@' + str(limit_price) if limit_price else 'MKT'} [{reason}]")

            return {
                'success': True,
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'action': action,
                'quantity': close_qty,
                'strike': contract.strike,
                'right': contract.right,
                'expiration': contract.lastTradeDateOrContractMonth,
                'reason': reason,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Close option failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    async def roll_option_position(
        self,
        symbol: str,
        current_contract: Contract,
        quantity: int,
        new_strike: float = None,
        new_dte: int = None,
        roll_type: str = 'ROLL_OUT',
        limit_price: float = None
    ) -> Dict[str, Any]:
        """
        Roll an option position to a new strike and/or expiration.

        Args:
            symbol: Underlying symbol
            current_contract: The current option Contract to close
            quantity: Number of contracts
            new_strike: New strike price (None = same strike)
            new_dte: Days to expiration for new position
            roll_type: ROLL_OUT, ROLL_UP, ROLL_DOWN, ROLL_OUT_UP, ROLL_OUT_DOWN
            limit_price: Net limit price for the roll (credit positive)

        Returns:
            Dict with order results for both legs
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            # Determine current position direction
            positions = self.ib.positions()
            current_qty = 0
            for pos in positions:
                if pos.contract.conId == current_contract.conId:
                    current_qty = int(pos.position)
                    break

            is_short = current_qty < 0
            close_action = 'BUY' if is_short else 'SELL'
            open_action = 'SELL' if is_short else 'BUY'

            # Determine new strike (agent always specifies; fallback = same strike)
            if new_strike is None:
                new_strike = current_contract.strike

            # Determine new expiration
            if new_dte:
                # Find expiration closest to target DTE
                chain = await self.get_option_chain(symbol, new_dte - 7, new_dte + 14)
                if 'error' in chain:
                    return chain

                target_dte = new_dte
                best_exp = None
                best_diff = float('inf')
                for exp_info in chain['expirations']:
                    diff = abs(exp_info['dte'] - target_dte)
                    if diff < best_diff:
                        best_diff = diff
                        best_exp = exp_info['expiration']

                new_expiration = best_exp or current_contract.lastTradeDateOrContractMonth
            else:
                # Default: roll out ~30 days
                chain = await self.get_option_chain(symbol, 25, 45)
                if 'error' not in chain and chain['expirations']:
                    new_expiration = chain['expirations'][0]['expiration']
                else:
                    return {'error': 'Could not find valid expiration for roll'}

            # Create new option contract
            new_contract = Option(
                symbol,
                new_expiration,
                new_strike,
                current_contract.right,
                'SMART'
            )
            await self.ib.qualifyContractsAsync(new_contract)
            if new_contract.conId == 0:
                return {'error': f'Roll target not found: {symbol} {new_strike}{current_contract.right} '
                                 f'{new_expiration}. Check that expiration exists.'}

            # Build roll as combo order (close current + open new)
            combo = self._build_combo(symbol, [
                (current_contract.conId, 1, close_action),
                (new_contract.conId, 1, open_action),
            ])

            order = Order()
            order.action = 'BUY'  # Combo action
            order.totalQuantity = quantity
            order.orderType = 'LMT' if limit_price else 'MKT'
            if limit_price:
                order.lmtPrice = abs(limit_price)
            order.tif = 'DAY'
            order.transmit = True

            trade = self.ib.placeOrder(combo, order)

            logger.info(f"Roll option: {symbol} "
                       f"{current_contract.strike}{current_contract.right} {current_contract.lastTradeDateOrContractMonth} -> "
                       f"{new_strike}{current_contract.right} {new_expiration} "
                       f"x{quantity} [{roll_type}]")

            return {
                'success': True,
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'roll_type': roll_type,
                'old_strike': current_contract.strike,
                'old_expiration': current_contract.lastTradeDateOrContractMonth,
                'new_strike': new_strike,
                'new_expiration': new_expiration,
                'quantity': quantity,
                'limit_price': limit_price,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Roll option failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    async def get_option_greeks(
        self,
        contract: Contract
    ) -> Dict[str, Any]:
        """
        Get Greeks for an option contract.

        Args:
            contract: The option Contract

        Returns:
            Dict with delta, gamma, theta, vega, IV
        """
        if not await self._ensure_connected():
            return {'error': 'Not connected'}

        try:
            # Request market data with Greeks
            ticker = self.ib.reqMktData(contract, '100,101,104,106', False, False)
            await self.ib.sleep(2)  # Wait for data

            greeks = {
                'symbol': contract.symbol,
                'strike': contract.strike,
                'right': contract.right,
                'expiration': contract.lastTradeDateOrContractMonth,
                'delta': getattr(ticker, 'modelGreeks', None) and ticker.modelGreeks.delta,
                'gamma': getattr(ticker, 'modelGreeks', None) and ticker.modelGreeks.gamma,
                'theta': getattr(ticker, 'modelGreeks', None) and ticker.modelGreeks.theta,
                'vega': getattr(ticker, 'modelGreeks', None) and ticker.modelGreeks.vega,
                'iv': getattr(ticker, 'modelGreeks', None) and ticker.modelGreeks.impliedVol,
                'underlying_price': ticker.last or ticker.close,
                'option_price': ticker.last or ticker.close,
                'bid': ticker.bid,
                'ask': ticker.ask,
            }

            # Cancel market data subscription
            self.ib.cancelMktData(contract)

            return greeks

        except Exception as e:
            logger.error(f"Get Greeks failed for {contract.symbol}: {e}")
            return {'error': str(e)}

    async def get_position_greeks(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get Greeks for all option positions (or filtered by symbol).

        Args:
            symbol: Optional filter by underlying symbol

        Returns:
            List of position dicts with Greeks
        """
        if not await self._ensure_connected():
            return []

        try:
            positions = self.ib.positions()
            option_positions = [
                p for p in positions
                if p.contract.secType == 'OPT' and (symbol is None or p.contract.symbol == symbol)
            ]

            results = []
            for pos in option_positions:
                greeks = await self.get_option_greeks(pos.contract)
                if 'error' not in greeks:
                    greeks['quantity'] = int(pos.position)
                    greeks['avg_cost'] = float(pos.avgCost)
                    greeks['is_short'] = pos.position < 0
                    greeks['is_long'] = pos.position > 0
                    results.append(greeks)

            return results

        except Exception as e:
            logger.error(f"Get position Greeks failed: {e}")
            return []
