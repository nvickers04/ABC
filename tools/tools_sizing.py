"""Position sizing tool handlers — deterministic sizing the agent calls before orders."""

from typing import Any

import numpy as np

from core.log_context import get_logger
from core.runtime.operating_context import get_operating_context
from research.config import CV_BOOTSTRAP_SAMPLES

logger = get_logger(__name__)


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

    risk_per_trade_pct = float(params.get("risk_per_trade_pct",
                                             _cv_adjusted_risk(1.5)))
    max_position_pct = float(params.get("max_position_pct", 20.0))
    stop_distance_pct = params.get("stop_distance_pct")

    # --- Gather data ---

    # Account values — query gateway directly
    net_liq = executor.gateway.net_liquidation if executor.gateway else 0
    cash = executor.gateway.cash_value if executor.gateway else 0

    if net_liq <= 0:
        return {"error": "Cannot size: net liquidation is $0 or unknown"}

    # Quote (replay sim returns dict; live MDA returns Quote)
    quote = executor.data_provider.get_quote(symbol)
    if isinstance(quote, dict):
        price = float(quote.get("last") or quote.get("close") or 0)
        ask = float(quote.get("ask") or price)
        bid = float(quote.get("bid") or price)
    elif quote and getattr(quote, "last", None):
        price = float(quote.last)
        ask = float(quote.ask or price)
        bid = float(quote.bid or price)
    else:
        price = 0.0
        ask = bid = 0.0
    if price <= 0:
        return {"error": f"Cannot size: no quote for {symbol}"}

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

    # Host applies risk_multiplier via QualityMatrix.get_scaled_quantity() on orders.
    # Do not scale here — avoids double-reduction when calculate_size → plan_order.
    try:
        ctx = get_operating_context()
        rm = ctx.risk_multiplier
        if rm < 1.0:
            reasoning.append(
                f"Host policy: order quantities will be scaled to {rm*100:.0f}% "
                f"at execution (QualityMatrix enforcement)."
            )
    except Exception:
        pass

    reasoning.append(
        f"Risk-based: {risk_shares} shares "
        f"(risking ${risk_dollars:,.0f} = {risk_per_trade_pct}% of NL, "
        f"stop distance {stop_distance_pct:.1f}%)"
    )

    # --- 2. Concentration limit ---
    max_position_dollars = net_liq * (max_position_pct / 100)
    entry_price = ask if side == 'BUY' else bid
    concentration_shares = int(max_position_dollars / entry_price) if entry_price > 0 else 0

    # Check existing position in same symbol via gateway
    existing_value = 0.0
    portfolio = executor.gateway.get_cached_portfolio() if executor.gateway else []
    for item in portfolio:
        if item.contract.symbol.upper() == symbol and item.position != 0:
            existing_value += abs(item.marketValue)
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
    # Use TotalCashValue from gateway (NEVER AvailableFunds — that includes margin)
    available_cash_total = cash

    if side == 'BUY':
        # Reserve 5% cash buffer — CASH ONLY, no margin
        available_cash = max(0, available_cash_total * 0.95)
        cash_shares = int(available_cash / entry_price) if entry_price > 0 else 0
        reasoning.append(
            f"Cash: {cash_shares} shares "
            f"(${available_cash:,.0f} cash after 5% buffer, CASH-ONLY)"
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
            available_cash = max(0, available_cash_total * 0.95)
            cash_shares = int(available_cash / entry_price) if entry_price > 0 else 0
            reasoning.append(
                f"Cash: {cash_shares} shares "
                f"(${available_cash:,.0f} cash for short collateral, CASH-ONLY)"
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

    # Execution cost model — historical gap for this symbol
    exec_cost = None
    try:
        from memory import get_execution_cost
        ec = get_execution_cost(symbol=symbol)
        if ec and ec.get("trades", 0) >= 3:
            exec_cost = {
                "avg_gap_pct": ec["avg_gap_pct"],
                "trades_observed": ec["trades"],
            }
            if ec["avg_gap_pct"] > 0.005:
                reasoning.append(
                    f"⚠ Execution cost: {symbol} has {ec['avg_gap_pct']:.2%} avg gap "
                    f"over {ec['trades']} trades — factor into stops"
                )
    except Exception as e:
        logger.debug(f"Execution cost lookup failed for {symbol}: {e}")

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
        "execution_cost": exec_cost,
        "account": {
            "net_liq": round(net_liq, 2),
            "cash_available": round(available_cash_total, 2),
            "cash": round(cash, 2),
            "existing_exposure_in_symbol": round(existing_value, 2),
        },
    }


def _estimate_cv_edge() -> float:
    """
    Bootstrap-resample matched trade returns to estimate coefficient of variation
    of the edge. Higher CV = less certain about edge = smaller positions.

    Returns CV (0.0 if < 20 matched fills — inactive).
    """
    try:
        from memory import get_db
        db = get_db()
        rows = db.execute(
            "SELECT actual_pnl FROM trade_feedback WHERE actual_pnl IS NOT NULL ORDER BY ts DESC LIMIT 200"
        ).fetchall()
        if not rows or len(rows) < 20:
            return 0.0

        returns = np.array([r["actual_pnl"] for r in rows], dtype=float)
        n = len(returns)
        edges = np.empty(CV_BOOTSTRAP_SAMPLES)
        for i in range(CV_BOOTSTRAP_SAMPLES):
            sample = np.random.choice(returns, size=n, replace=True)
            edges[i] = sample.mean()

        mu = edges.mean()
        sigma = edges.std()
        if abs(mu) < 1e-9:
            return 1.0  # Edge indistinguishable from zero — maximum caution
        return float(min(abs(sigma / mu), 1.0))
    except Exception:
        return 0.0


def _cv_adjusted_risk(base_risk: float) -> float:
    """Adjust base risk % downward based on CV of edge estimate."""
    cv = _estimate_cv_edge()
    return round(base_risk * (1.0 - cv), 4)


HANDLERS = {
    "calculate_size": handle_calculate_size,
}


def register_handlers(registry) -> None:
    """Register this module's handlers on the central :class:`core.tool_registry.ToolRegistry`."""
    registry.bind_handlers(HANDLERS)
