"""Position sizing tool handlers — deterministic sizing the agent calls before orders."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _build_portfolio_context(state, net_liq: float) -> dict:
    """Build objective portfolio-level facts for the agent (no recommendations)."""
    positions = list(state._positions.values())
    if not positions:
        return {"total_positions": 0}

    import statistics
    hold_hours = [p.hold_time_hours for p in positions if p.hold_time_hours is not None]
    pnl_hrs = [p.pnl_per_hour for p in positions if p.pnl_per_hour is not None]
    eff_list = [p.efficiency for p in positions if p.efficiency is not None]

    context = {
        "total_positions": len(positions),
    }

    if hold_hours:
        median_hrs = statistics.median(hold_hours)
        context["median_hold_hours"] = round(median_hrs, 1)

        # Capital in longest 25%
        positions_sorted = sorted(positions, key=lambda p: p.hold_time_hours or 0, reverse=True)
        num_longest = max(1, int(len(positions) * 0.25))
        longest_25pct = positions_sorted[:num_longest]
        long_cap = sum(abs(p.market_value) for p in longest_25pct)
        long_pct = round(long_cap / net_liq * 100, 1) if net_liq > 0 else 0
        context["capital_longest_25pct"] = round(long_cap, 2)
        context["capital_longest_25pct_pct_nl"] = long_pct

    if pnl_hrs:
        median_pnl_hr = statistics.median(pnl_hrs)
        context["median_pnl_per_hour"] = round(median_pnl_hr, 2)

    if eff_list:
        context["median_efficiency_pct_hr"] = round(statistics.median(eff_list), 2)

    return context


async def handle_calculate_size(executor, params: dict) -> Any:
    """
    Calculate optimal position size based on account risk, concentration limits,
    and available cash. The agent should call this BEFORE placing any order.

    Required: symbol, side ('BUY'/'SELL')
    Optional: stop_distance_pct (default: auto from ATR),
              risk_per_trade_pct (default: 1.5% of net liq),
              max_position_pct (default: 20% of net liq)

    Returns: recommended quantity + reasoning breakdown.
    """
    symbol = params.get("symbol")
    side = params.get("side")
    if not symbol or not side:
        return {"error": "Required: symbol, side ('BUY'/'SELL')"}

    symbol = symbol.upper()
    side = side.upper()

    # STRICT cash-only guardrail: block sizing for short entries
    if executor.cash_only and side == "SELL":
        portfolio = executor.gateway.get_cached_portfolio() if executor.gateway else []
        is_closing_long = any(
            item.contract.symbol.upper() == symbol and item.position > 0
            for item in portfolio
        )
        if not is_closing_long:
            return {
                "error": f"CASH-ONLY BLOCKED: Cannot size short entry for {symbol}. "
                         f"This account is restricted to long-only (BUY side). "
                         f"For bearish views, use long puts or bear put spreads."
            }

    risk_per_trade_pct = float(params.get("risk_per_trade_pct", 1.5))
    max_position_pct = float(params.get("max_position_pct", 20.0))
    stop_distance_pct = params.get("stop_distance_pct")

    # --- Gather data ---

    # Account values — query gateway directly
    net_liq = executor.gateway.net_liquidation if executor.gateway else 0
    cash = executor.gateway.cash_value if executor.gateway else 0

    if net_liq <= 0:
        return {"error": "Cannot size: net liquidation is $0 or unknown"}

    # Quote
    quote = executor.data_provider.get_quote(symbol)
    if not quote or not quote.last or quote.last <= 0:
        return {"error": f"Cannot size: no quote for {symbol}"}
    price = quote.last
    ask = quote.ask or price
    bid = quote.bid or price

    # ATR for auto stop distance
    atr_result = executor.data_provider.get_atr(symbol)
    atr_value = atr_result.value if atr_result else None
    atr_pct = (atr_value / price * 100) if atr_value and price > 0 else None

    # Auto-calculate stop distance from ATR if not provided
    if stop_distance_pct is not None:
        stop_distance_pct = float(stop_distance_pct)
    elif atr_pct:
        # 1.5x ATR is standard stop distance
        stop_distance_pct = round(min(atr_pct * 1.5, 15.0), 2)
    else:
        # Fallback: 5% stop distance
        stop_distance_pct = 5.0

    reasoning = []

    # --- 1. Risk-based sizing (Kelly-lite) ---
    # How many shares can we buy such that if the stop hits, we lose risk_per_trade_pct of NL?
    risk_dollars = net_liq * (risk_per_trade_pct / 100)
    stop_loss_per_share = price * (stop_distance_pct / 100)
    if stop_loss_per_share > 0:
        risk_shares = int(risk_dollars / stop_loss_per_share)
    else:
        risk_shares = 0
    reasoning.append(
        f"Risk-based: {risk_shares} shares "
        f"(risking ${risk_dollars:,.0f} = {risk_per_trade_pct}% of NL, "
        f"stop distance {stop_distance_pct:.1f}%)"
    )

    # --- 2. Concentration limit ---
    max_position_dollars = net_liq * (max_position_pct / 100)
    entry_price = ask if side == 'BUY' else bid
    concentration_shares = int(max_position_dollars / entry_price) if entry_price > 0 else 0

    # Check existing position in same symbol
    existing_value = 0.0
    with state._lock:
        for key, pos in state._positions.items():
            underlying = state._underlying_from_key(key)
            if underlying == symbol:
                existing_value += abs(pos.market_value)
    remaining_capacity = max(0, max_position_dollars - existing_value)
    concentration_shares_adj = int(remaining_capacity / entry_price) if entry_price > 0 else 0

    if existing_value > 0:
        reasoning.append(
            f"Concentration: {concentration_shares_adj} shares "
            f"(max {max_position_pct}% of NL = ${max_position_dollars:,.0f}, "
            f"already have ${existing_value:,.0f} in {symbol})"
        )
    else:
        reasoning.append(
            f"Concentration: {concentration_shares} shares "
            f"(max {max_position_pct}% of NL = ${max_position_dollars:,.0f})"
        )

    # --- 3. Cash constraint (CASH-ONLY account — applies to BUY) ---
    # Use available_funds (AvailableFunds) — NOT buying_power (includes margin)
    available_funds = state._available_funds
    if state._broker:
        try:
            if hasattr(state._broker, 'available_funds') and state._broker.available_funds > 0:
                available_funds = state._broker.available_funds
        except Exception:
            pass
    # Fallback to cash if available_funds not yet populated
    if available_funds <= 0:
        available_funds = cash

    if side == 'BUY':
        # Reserve 5% cash buffer — CASH ONLY, no margin
        available_cash = max(0, available_funds * 0.95)
        cash_shares = int(available_cash / entry_price) if entry_price > 0 else 0
        reasoning.append(
            f"Cash: {cash_shares} shares "
            f"(${available_cash:,.0f} available funds after 5% buffer, CASH-ONLY)"
        )
    else:
        # Check if this is a short ENTRY (no existing position) vs closing a long
        portfolio = executor.gateway.get_cached_portfolio() if executor.gateway else []
        existing_pos = None
        for item in portfolio:
            if item.contract.symbol.upper() == symbol and item.position != 0:
                existing_pos = item
                break
        is_short_entry = existing_pos is None or existing_pos.position > 0
        if is_short_entry:
            # Short entry requires cash collateral — treat like BUY for sizing
            available_cash = max(0, available_funds * 0.95)
            cash_shares = int(available_cash / entry_price) if entry_price > 0 else 0
            reasoning.append(
                f"Cash: {cash_shares} shares "
                f"(${available_cash:,.0f} available for short collateral, CASH-ONLY)"
            )
        else:
            cash_shares = 999_999  # No cash constraint for closing existing long

    # --- Final recommendation ---
    max_by_risk = risk_shares
    max_by_concentration = concentration_shares_adj if existing_value > 0 else concentration_shares
    max_by_cash = cash_shares

    recommended = min(max_by_risk, max_by_concentration, max_by_cash)
    recommended = max(recommended, 0)

    # Determine binding constraint
    if recommended <= 0:
        binding = "NO CAPACITY"
        reasoning.append("⚠️ Cannot size: all constraints produce 0 shares")
    elif recommended == max_by_risk:
        binding = "RISK"
    elif recommended == max_by_concentration:
        binding = "CONCENTRATION"
    else:
        binding = "CASH"

    # Position cost
    position_cost = recommended * entry_price
    position_pct = (position_cost / net_liq * 100) if net_liq > 0 else 0
    max_loss = recommended * stop_loss_per_share
    max_loss_pct = (max_loss / net_liq * 100) if net_liq > 0 else 0

    return {
        "symbol": symbol,
        "side": side,
        "recommended_quantity": recommended,
        "binding_constraint": binding,
        "position_cost": round(position_cost, 2),
        "position_pct_of_nl": round(position_pct, 1),
        "max_loss_at_stop": round(max_loss, 2),
        "max_loss_pct_of_nl": round(max_loss_pct, 2),
        "entry_price": round(entry_price, 2),
        "stop_distance_pct": stop_distance_pct,
        "atr_pct": round(atr_pct, 2) if atr_pct else None,
        # --- Pass these directly to plan_order for consistent stops ---
        "plan_order_params": {
            "symbol": symbol,
            "side": side,
            "quantity": recommended,
            "stop_distance_pct": stop_distance_pct,
            "execute": True,
        },
        "breakdown": {
            "risk_shares": max_by_risk,
            "concentration_shares": max_by_concentration,
            "cash_shares": max_by_cash if side == 'BUY' else "N/A (sell)",
        },
        "reasoning": reasoning,
        "account": {
            "net_liq": round(net_liq, 2),
            "available_funds": round(available_funds, 2),
            "cash": round(cash, 2),
            "existing_exposure_in_symbol": round(existing_value, 2),
        },
        "portfolio_context": _build_portfolio_context(state, net_liq),
    }


HANDLERS = {
    "calculate_size": handle_calculate_size,
}
