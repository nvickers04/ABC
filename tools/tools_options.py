"""Options tool handlers (single leg, spreads, management, chain/greeks)."""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_expiration(exp: str) -> str:
    """Convert any expiration format to YYYYMMDD for IBKR.
    
    Handles:
    - Unix timestamp string (e.g. '1773432000') -> 'YYYYMMDD'
    - YYYY-MM-DD -> 'YYYYMMDD'
    - Already YYYYMMDD -> passthrough
    """
    if not exp:
        return exp
    exp = str(exp).strip()
    # Unix timestamp: all digits, >= 10 chars (epoch seconds)
    if exp.isdigit() and len(exp) >= 10:
        try:
            dt = datetime.utcfromtimestamp(int(exp))
            return dt.strftime('%Y%m%d')
        except (ValueError, OSError):
            return exp
    # YYYY-MM-DD
    if len(exp) == 10 and exp[4] == '-' and exp[7] == '-':
        return exp.replace('-', '')
    return exp


def _make_leg_key(symbol: str, right: str, strike: float, expiration: str) -> str:
    """Build a position key matching IBKR callback format: SYMBOL_RIGHT_STRIKE_EXPIRY."""
    return f"{symbol}_{right}_{strike}_{expiration}"


def _register_spread_if_success(result: dict, underlying: str, strategy: str,
                                 leg_keys: list):
    """Register a spread group after a successful placement (no-op without LiveState)."""
    if not result or not result.get("success"):
        return
    # Spread group tracking removed with LiveState deletion
    logger.debug(f"Spread placed: {strategy} on {underlying}, legs={leg_keys}")


# =========================================================================
# SINGLE LEG
# =========================================================================

async def handle_buy_option(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    strike = params.get("strike")
    right = params.get("right")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, strike, right]):
        return {"error": "Required: symbol, expiration ('YYYYMMDD'), strike, right ('C'/'P')"}
    pdt = executor._check_pdt('BUY')
    if pdt:
        return pdt
    if executor.gateway and executor.gateway.cash_value <= 0:
        return {"error": "INSUFFICIENT CASH: No available cash for options purchase."}
    logger.info(f"BUY OPTION: {symbol} {right.upper()}{strike} exp={expiration} qty={quantity}")
    result = await executor.gateway.buy_option(symbol, expiration, float(strike), right.upper(), int(quantity))
    logger.info(f"BUY OPTION RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    return result


async def handle_covered_call(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    strike = params.get("strike")
    shares = params.get("shares", 100)
    if not all([symbol, expiration, strike]):
        return {"error": "Required: symbol, expiration ('YYYYMMDD'), strike"}
    logger.info(f"COVERED CALL: {symbol} strike={strike} exp={expiration} shares={shares}")
    result = await executor.gateway.place_covered_call(symbol, expiration, float(strike), int(shares))
    logger.info(f"COVERED CALL RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    return result


async def handle_cash_secured_put(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    strike = params.get("strike")
    contracts = params.get("contracts", 1)
    if not all([symbol, expiration, strike]):
        return {"error": "Required: symbol, expiration ('YYYYMMDD'), strike"}
    cash = executor._check_cash(float(strike) * 100 * int(contracts))
    if cash:
        return cash
    logger.info(f"CASH SECURED PUT: {symbol} strike={strike} exp={expiration} contracts={contracts}")
    result = await executor.gateway.sell_cash_secured_put(symbol, expiration, float(strike), int(contracts))
    logger.info(f"CASH SECURED PUT RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    return result


async def handle_protective_put(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    strike = params.get("strike")
    shares = params.get("shares", 100)
    if not all([symbol, expiration, strike]):
        return {"error": "Required: symbol, expiration ('YYYYMMDD'), strike"}
    if executor.gateway and executor.gateway.cash_value <= 0:
        return {"error": "INSUFFICIENT CASH: No available cash for protective put purchase."}
    logger.info(f"PROTECTIVE PUT: {symbol} strike={strike} exp={expiration} shares={shares}")
    result = await executor.gateway.place_protective_put(symbol, expiration, float(strike), int(shares))
    logger.info(f"PROTECTIVE PUT RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    return result


# =========================================================================
# SPREADS
# =========================================================================

async def handle_vertical_spread(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    long_strike = params.get("long_strike")
    short_strike = params.get("short_strike")
    right = params.get("right")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, long_strike, short_strike, right]):
        return {"error": "Required: symbol, expiration, long_strike, short_strike, right ('C'/'P')"}
    pdt = executor._check_pdt('BUY')
    if pdt:
        return pdt
    max_debit = abs(float(long_strike) - float(short_strike)) * 100 * int(quantity)
    cash = executor._check_cash(max_debit)
    if cash:
        return cash
    logger.info(f"VERTICAL SPREAD: {symbol} {right.upper()} long={long_strike} short={short_strike} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_vertical_spread(
        symbol, expiration, float(long_strike), float(short_strike), right.upper(), int(quantity)
    )
    logger.info(f"VERTICAL SPREAD RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    r = right.upper()
    leg_keys = [
        _make_leg_key(symbol, r, float(long_strike), expiration),
        _make_leg_key(symbol, r, float(short_strike), expiration),
    ]
    strategy = f"{'Bull' if r == 'C' else 'Bear'} {'Call' if r == 'C' else 'Put'} Spread"
    _register_spread_if_success(result, symbol, strategy, leg_keys)
    return result


async def handle_iron_condor(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    put_long = params.get("put_long_strike")
    put_short = params.get("put_short_strike")
    call_short = params.get("call_short_strike")
    call_long = params.get("call_long_strike")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, put_long, put_short, call_short, call_long]):
        return {"error": "Required: symbol, expiration, put_long_strike, put_short_strike, call_short_strike, call_long_strike (in order: low to high)"}
    put_width = abs(float(put_short) - float(put_long)) * 100 * int(quantity)
    call_width = abs(float(call_long) - float(call_short)) * 100 * int(quantity)
    max_collateral = max(put_width, call_width)
    cash = executor._check_cash(max_collateral)
    if cash:
        return cash
    logger.info(f"IRON CONDOR: {symbol} puts={put_long}/{put_short} calls={call_short}/{call_long} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_iron_condor(
        symbol, expiration, float(put_long), float(put_short), float(call_short), float(call_long), int(quantity)
    )
    logger.info(f"IRON CONDOR RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    leg_keys = [
        _make_leg_key(symbol, 'P', float(put_long), expiration),
        _make_leg_key(symbol, 'P', float(put_short), expiration),
        _make_leg_key(symbol, 'C', float(call_short), expiration),
        _make_leg_key(symbol, 'C', float(call_long), expiration),
    ]
    _register_spread_if_success(result, symbol, 'Iron Condor', leg_keys)
    return result


async def handle_iron_butterfly(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    center_strike = params.get("center_strike")
    wing_width = params.get("wing_width")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, center_strike, wing_width]):
        return {"error": "Required: symbol, expiration, center_strike, wing_width"}
    max_collateral = float(wing_width) * 100 * int(quantity)
    cash = executor._check_cash(max_collateral)
    if cash:
        return cash
    cs = float(center_strike)
    ww = float(wing_width)
    logger.info(f"IRON BUTTERFLY: {symbol} center={cs} wing={ww} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_iron_butterfly(
        symbol, expiration, cs, ww, int(quantity)
    )
    logger.info(f"IRON BUTTERFLY RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    leg_keys = [
        _make_leg_key(symbol, 'P', cs - ww, expiration),
        _make_leg_key(symbol, 'P', cs, expiration),
        _make_leg_key(symbol, 'C', cs, expiration),
        _make_leg_key(symbol, 'C', cs + ww, expiration),
    ]
    _register_spread_if_success(result, symbol, 'Iron Butterfly', leg_keys)
    return result


async def handle_straddle(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    strike = params.get("strike")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, strike]):
        return {"error": "Required: symbol, expiration, strike"}
    pdt = executor._check_pdt('BUY')
    if pdt:
        return pdt
    estimated_debit = float(strike) * 0.10 * 100 * int(quantity)
    if executor.gateway and executor.gateway.cash_value < estimated_debit:
        return {"error": f"INSUFFICIENT CASH: Straddle estimated to cost ~${estimated_debit:,.0f} but only ${executor.gateway.cash_value:,.2f} available."}
    s = float(strike)
    logger.info(f"STRADDLE: {symbol} strike={s} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_straddle(symbol, expiration, s, int(quantity))
    logger.info(f"STRADDLE RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    leg_keys = [
        _make_leg_key(symbol, 'C', s, expiration),
        _make_leg_key(symbol, 'P', s, expiration),
    ]
    _register_spread_if_success(result, symbol, 'Straddle', leg_keys)
    return result


async def handle_strangle(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    put_strike = params.get("put_strike")
    call_strike = params.get("call_strike")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, put_strike, call_strike]):
        return {"error": "Required: symbol, expiration, put_strike, call_strike"}
    pdt = executor._check_pdt('BUY')
    if pdt:
        return pdt
    avg_strike = (float(put_strike) + float(call_strike)) / 2
    estimated_debit = avg_strike * 0.08 * 100 * int(quantity)
    if executor.gateway and executor.gateway.cash_value < estimated_debit:
        return {"error": f"INSUFFICIENT CASH: Strangle estimated to cost ~${estimated_debit:,.0f} but only ${executor.gateway.cash_value:,.2f} available."}
    logger.info(f"STRANGLE: {symbol} puts={put_strike} calls={call_strike} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_strangle(symbol, expiration, float(put_strike), float(call_strike), int(quantity))
    logger.info(f"STRANGLE RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    leg_keys = [
        _make_leg_key(symbol, 'P', float(put_strike), expiration),
        _make_leg_key(symbol, 'C', float(call_strike), expiration),
    ]
    _register_spread_if_success(result, symbol, 'Strangle', leg_keys)
    return result


async def handle_collar(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    put_strike = params.get("put_strike")
    call_strike = params.get("call_strike")
    shares = params.get("shares", 100)
    if not all([symbol, expiration, put_strike, call_strike]):
        return {"error": "Required: symbol, expiration, put_strike, call_strike"}
    logger.info(f"COLLAR: {symbol} put={put_strike} call={call_strike} exp={expiration} shares={shares}")
    result = await executor.gateway.place_collar(symbol, expiration, float(put_strike), float(call_strike), int(shares))
    logger.info(f"COLLAR RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    leg_keys = [
        _make_leg_key(symbol, 'P', float(put_strike), expiration),
        _make_leg_key(symbol, 'C', float(call_strike), expiration),
    ]
    _register_spread_if_success(result, symbol, 'Collar', leg_keys)
    return result


async def handle_calendar_spread(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    strike = params.get("strike")
    near_exp = params.get("near_expiration")
    far_exp = params.get("far_expiration")
    right = params.get("right", "C")
    quantity = params.get("quantity", 1)
    if not all([symbol, strike, near_exp, far_exp]):
        return {"error": "Required: symbol, strike, near_expiration, far_expiration"}
    pdt = executor._check_pdt('BUY')
    if pdt:
        return pdt
    estimated_debit = float(strike) * 0.05 * 100 * int(quantity)
    if executor.gateway and executor.gateway.cash_value < estimated_debit:
        return {"error": f"INSUFFICIENT CASH: Calendar spread estimated to cost ~${estimated_debit:,.0f} but only ${executor.gateway.cash_value:,.2f} available."}
    logger.info(f"CALENDAR SPREAD: {symbol} strike={strike} near={near_exp} far={far_exp} {right} qty={quantity}")
    result = await executor.gateway.place_calendar_spread(symbol, float(strike), near_exp, far_exp, right.upper(), int(quantity))
    logger.info(f"CALENDAR SPREAD RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    r = right.upper()
    s = float(strike)
    leg_keys = [
        _make_leg_key(symbol, r, s, near_exp),
        _make_leg_key(symbol, r, s, far_exp),
    ]
    _register_spread_if_success(result, symbol, 'Calendar Spread', leg_keys)
    return result


async def handle_diagonal_spread(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    near_strike = params.get("near_strike")
    far_strike = params.get("far_strike")
    near_exp = params.get("near_expiration")
    far_exp = params.get("far_expiration")
    right = params.get("right", "C")
    quantity = params.get("quantity", 1)
    if not all([symbol, near_strike, far_strike, near_exp, far_exp]):
        return {"error": "Required: symbol, near_strike, far_strike, near_expiration, far_expiration"}
    pdt = executor._check_pdt('BUY')
    if pdt:
        return pdt
    max_debit = abs(float(near_strike) - float(far_strike)) * 100 * int(quantity)
    if max_debit > 0:
        cash = executor._check_cash(max_debit)
        if cash:
            return cash
    logger.info(f"DIAGONAL SPREAD: {symbol} near={near_strike}/{near_exp} far={far_strike}/{far_exp} {right} qty={quantity}")
    result = await executor.gateway.place_diagonal_spread(
        symbol, float(near_strike), float(far_strike), near_exp, far_exp, right.upper(), int(quantity)
    )
    logger.info(f"DIAGONAL SPREAD RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    r = right.upper()
    leg_keys = [
        _make_leg_key(symbol, r, float(near_strike), near_exp),
        _make_leg_key(symbol, r, float(far_strike), far_exp),
    ]
    _register_spread_if_success(result, symbol, 'Diagonal Spread', leg_keys)
    return result


async def handle_butterfly(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    lower_strike = params.get("lower_strike")
    middle_strike = params.get("middle_strike")
    upper_strike = params.get("upper_strike")
    right = params.get("right", "C")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, lower_strike, middle_strike, upper_strike]):
        return {"error": "Required: symbol, expiration, lower_strike, middle_strike, upper_strike. Optional: right ('C'/'P'), quantity"}
    pdt = executor._check_pdt('BUY')
    if pdt:
        return pdt
    wing_width = float(middle_strike) - float(lower_strike)
    max_debit = wing_width * 100 * int(quantity)
    if max_debit > 0:
        cash = executor._check_cash(max_debit)
        if cash:
            return cash
    logger.info(f"BUTTERFLY: {symbol} {lower_strike}/{middle_strike}/{upper_strike} {right} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_butterfly(
        symbol, expiration, float(lower_strike), float(middle_strike), float(upper_strike),
        right.upper(), int(quantity)
    )
    logger.info(f"BUTTERFLY RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    r = right.upper()
    leg_keys = [
        _make_leg_key(symbol, r, float(lower_strike), expiration),
        _make_leg_key(symbol, r, float(middle_strike), expiration),
        _make_leg_key(symbol, r, float(upper_strike), expiration),
    ]
    _register_spread_if_success(result, symbol, 'Butterfly', leg_keys)
    return result


async def handle_ratio_spread(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    long_strike = params.get("long_strike")
    short_strike = params.get("short_strike")
    right = params.get("right", "C")
    ratio = params.get("ratio", [1, 2])
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, long_strike, short_strike]):
        return {"error": "Required: symbol, expiration, long_strike, short_strike. Optional: right, ratio [long, short], quantity"}
    logger.info(f"RATIO SPREAD: {symbol} long={long_strike} short={short_strike} {right} ratio={ratio} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_ratio_spread(
        symbol, expiration, float(long_strike), float(short_strike),
        right.upper(), tuple(ratio), int(quantity)
    )
    logger.info(f"RATIO SPREAD RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    r = right.upper()
    leg_keys = [
        _make_leg_key(symbol, r, float(long_strike), expiration),
        _make_leg_key(symbol, r, float(short_strike), expiration),
    ]
    _register_spread_if_success(result, symbol, 'Ratio Spread', leg_keys)
    return result


async def handle_jade_lizard(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    put_strike = params.get("put_strike")
    call_short_strike = params.get("call_short_strike")
    call_long_strike = params.get("call_long_strike")
    quantity = params.get("quantity", 1)
    if not all([symbol, expiration, put_strike, call_short_strike, call_long_strike]):
        return {"error": "Required: symbol, expiration, put_strike, call_short_strike, call_long_strike"}
    max_collateral = float(put_strike) * 100 * int(quantity)
    cash = executor._check_cash(max_collateral)
    if cash:
        return cash
    logger.info(f"JADE LIZARD: {symbol} put={put_strike} calls={call_short_strike}/{call_long_strike} exp={expiration} qty={quantity}")
    result = await executor.gateway.place_jade_lizard(
        symbol, expiration, float(put_strike), float(call_short_strike), float(call_long_strike),
        int(quantity)
    )
    logger.info(f"JADE LIZARD RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    leg_keys = [
        _make_leg_key(symbol, 'P', float(put_strike), expiration),
        _make_leg_key(symbol, 'C', float(call_short_strike), expiration),
        _make_leg_key(symbol, 'C', float(call_long_strike), expiration),
    ]
    _register_spread_if_success(result, symbol, 'Jade Lizard', leg_keys)
    return result


# =========================================================================
# MANAGEMENT
# =========================================================================

async def handle_close_option(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    strike = params.get("strike")
    right = params.get("right")
    limit_price = params.get("limit_price")
    force = params.get("force", False)
    if not all([symbol, expiration, strike, right]):
        return {"error": "Required: symbol, expiration, strike, right"}
    
    # Warn if this leg belongs to a spread (spread tracking removed with LiveState)
    # With no spread registry, we skip the orphan-leg warning.
    # The agent should still manage spreads carefully.
    
    logger.info(f"CLOSE OPTION: {symbol} {right}{strike} exp={expiration} limit={limit_price}")
    result = await executor.gateway.close_option_position(
        symbol, expiration, float(strike), right.upper(),
        limit_price=float(limit_price) if limit_price else None
    )
    logger.info(f"CLOSE OPTION RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    return result


async def handle_roll_option(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    old_exp = params.get("old_expiration")
    old_strike = params.get("old_strike")
    new_exp = params.get("new_expiration")
    new_strike = params.get("new_strike")
    right = params.get("right")
    quantity = params.get("quantity", 1)
    if not all([symbol, old_exp, old_strike, new_exp, new_strike, right]):
        return {"error": "Required: symbol, old_expiration, old_strike, new_expiration, new_strike, right"}
    current_contract = await executor.gateway.qualify_option_contract(
        symbol, old_exp, float(old_strike), right
    )
    if current_contract is None:
        return {"error": f"Could not qualify old contract: {symbol} {old_exp} {old_strike} {right}"}
    from datetime import datetime as dt
    try:
        exp_date = dt.strptime(new_exp, '%Y%m%d')
        new_dte = (exp_date - dt.now()).days
    except ValueError:
        new_dte = 30
    logger.info(f"ROLL OPTION: {symbol} {right}{old_strike} {old_exp} -> {new_strike} {new_exp} qty={quantity}")
    result = await executor.gateway.roll_option_position(
        symbol, current_contract, int(quantity),
        new_strike=float(new_strike), new_dte=new_dte
    )
    logger.info(f"ROLL OPTION RESULT: {symbol} -> {result}")
    await executor._refresh_state()
    return result


async def handle_option_chain(executor, params: dict) -> Any:
    """
    Get option chain with input normalization.
    
    Accepts:
      symbol: required
      expiration: 'YYYYMMDD' or 'YYYY-MM-DD' (normalized to YYYY-MM-DD)
      side: 'call'/'put'/'C'/'P' (normalized to 'call'/'put')
      dte_min, dte_max: DTE range (omit for all expirations)
      strike_min, strike_max: strike price range (omit for all strikes)
      limit: max contracts to return (default 20)
    """
    import re
    
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    symbol = symbol.upper()
    
    # --- Normalize expiration: YYYYMMDD -> YYYY-MM-DD ---
    expiration = params.get("expiration")
    if expiration:
        expiration = str(expiration).strip()
        if re.match(r'^\d{8}$', expiration):
            expiration = f"{expiration[:4]}-{expiration[4:6]}-{expiration[6:8]}"
        elif not re.match(r'^\d{4}-\d{2}-\d{2}$', expiration):
            return {"error": f"Invalid expiration format '{expiration}'. Use YYYYMMDD or YYYY-MM-DD."}
    
    # --- Normalize side: C/P -> call/put ---
    side = params.get("side")
    if side:
        side = str(side).strip().lower()
        if side in ('c', 'call'):
            side = 'call'
        elif side in ('p', 'put'):
            side = 'put'
        else:
            return {"error": f"Invalid side '{side}'. Use 'call' or 'put'."}
    
    # --- Build DTE range (None if not specified) ---
    dte_min = params.get("dte_min")
    dte_max = params.get("dte_max")
    dte_range = (dte_min, dte_max) if dte_min is not None or dte_max is not None else None
    
    # --- Build strike range (None if not specified) ---
    strike_min = params.get("strike_min")
    strike_max = params.get("strike_max")
    strike_range = None
    if strike_min is not None or strike_max is not None:
        strike_range = (
            float(strike_min) if strike_min is not None else 0,
            float(strike_max) if strike_max is not None else 999999
        )
    
    limit = params.get("limit", 20)
    
    # --- Fetch from MarketData ---
    chain = executor.data_provider.get_option_chain(
        symbol,
        expiration=expiration,
        side=side,
        strike_range=strike_range,
        dte_range=dte_range
    )
    
    # --- No data: return what was requested + what to try ---
    if not chain or not chain.contracts:
        return {
            "error": f"No option chain found for {symbol}",
            "requested": {
                "symbol": symbol,
                "expiration": expiration,
                "side": side,
                "dte_min": dte_min,
                "dte_max": dte_max,
                "strike_min": strike_min,
                "strike_max": strike_max
            },
            "retry_suggestions": [
                "Omit 'expiration' to search all available dates",
                "Omit 'dte_min'/'dte_max' to get all expirations",
                "Try dte_min=7, dte_max=90 for a wider range",
                "Omit 'strike_min'/'strike_max' to get all strikes",
                "Check if symbol has listed options (some ETNs/small caps don't)"
            ],
            "available_params": {
                "symbol": "required - underlying ticker",
                "side": "optional - 'call' or 'put' (omit for both)",
                "expiration": "optional - 'YYYY-MM-DD' or 'YYYYMMDD'",
                "dte_min": "optional - minimum days to expiration",
                "dte_max": "optional - maximum days to expiration",
                "strike_min": "optional - minimum strike price",
                "strike_max": "optional - maximum strike price",
                "limit": "optional - max contracts to return (default 20)"
            }
        }
    
    # --- Format output ---
    contracts_out = []
    for c in chain.contracts[:limit]:
        contracts_out.append({
            "symbol": c.option_symbol,
            "strike": c.strike,
            "side": c.side,
            "expiration": c.expiration,
            "dte": c.dte,
            "bid": c.bid,
            "ask": c.ask,
            "last": c.last,
            "volume": c.volume,
            "open_interest": c.open_interest,
            "delta": c.delta,
            "gamma": c.gamma,
            "theta": c.theta,
            "vega": c.vega,
            "iv": c.iv
        })
    
    return {
        "symbol": symbol,
        "count": len(contracts_out),
        "contracts": contracts_out,
        "source": chain.source
    }


async def handle_option_greeks(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    expiration = params.get("expiration")
    strike = params.get("strike")
    right = params.get("right")
    if not all([symbol, expiration, strike, right]):
        return {"error": "Required: symbol, expiration ('YYYYMMDD'), strike, right ('C'/'P')"}

    target_strike = float(strike)

    # Try data_provider first (MarketData.app — works without broker)
    try:
        chain = executor.data_provider.get_option_chain(
            symbol, expiration=expiration, side=right.upper()
        )
        if chain and chain.contracts:
            for c in chain.contracts:
                if abs(c.strike - target_strike) < 0.01:
                    return {
                        "symbol": c.option_symbol or f"{symbol} {expiration} {strike} {right}",
                        "strike": c.strike,
                        "expiration": c.expiration,
                        "right": right.upper(),
                        "delta": c.delta,
                        "gamma": c.gamma,
                        "theta": c.theta,
                        "vega": c.vega,
                        "iv": c.iv,
                        "bid": c.bid,
                        "ask": c.ask,
                        "last": c.last,
                        "volume": c.volume,
                        "open_interest": c.open_interest,
                        "source": chain.source,
                    }
    except Exception as e:
        logger.debug(f"Data provider option_greeks failed for {symbol}: {e}")

    # Fallback to broker if data_provider didn't have the contract
    if not executor.gateway:
        return {"error": f"Could not find {symbol} {expiration} {strike} {right} in data provider and broker not connected"}
    contract = await executor.gateway.qualify_option_contract(
        symbol, expiration, float(strike), right
    )
    if contract is None:
        return {"error": f"Could not qualify contract: {symbol} {expiration} {strike} {right}"}
    return await executor.gateway.get_option_greeks(contract)


async def handle_position_greeks(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    results = await executor.gateway.get_position_greeks(symbol)
    return {"positions": results, "count": len(results)}


HANDLERS = {
    # Single leg
    "buy_option": handle_buy_option,
    "covered_call": handle_covered_call,
    "cash_secured_put": handle_cash_secured_put,
    "protective_put": handle_protective_put,
    # Spreads
    "vertical_spread": handle_vertical_spread,
    "iron_condor": handle_iron_condor,
    "iron_butterfly": handle_iron_butterfly,
    "straddle": handle_straddle,
    "strangle": handle_strangle,
    "collar": handle_collar,
    "calendar_spread": handle_calendar_spread,
    "diagonal_spread": handle_diagonal_spread,
    "butterfly": handle_butterfly,
    "ratio_spread": handle_ratio_spread,
    "jade_lizard": handle_jade_lizard,
    # Management
    "close_option": handle_close_option,
    "roll_option": handle_roll_option,
    "option_chain": handle_option_chain,
    "option_greeks": handle_option_greeks,
    "position_greeks": handle_position_greeks,
}
