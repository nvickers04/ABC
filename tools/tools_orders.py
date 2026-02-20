"""Stock order tool handlers — factory-based implementation.

All stock orders share common validation (broker, params, PDT, cash).
The _run_order helper centralizes this; each handler only specifies
its unique gateway call and cash estimation strategy.

Reduction: 594 → ~270 lines with identical runtime behavior.
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# PARAM SANITIZATION — LLM sometimes sends dicts where scalars are expected
# =============================================================================

def _safe_float(val) -> float:
    """Extract float from value, handling dict wrapping from LLM."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val)
    if isinstance(val, dict):
        # LLM wraps values: {"price": 148.52} or {"value": 10.5}
        for key in ("price", "value", "amount", "stop", "target", "limit"):
            if key in val:
                return float(val[key])
        # Try first numeric value in the dict
        for v in val.values():
            if isinstance(v, (int, float)):
                return float(v)
        raise ValueError(f"Cannot extract float from dict: {val}")
    raise TypeError(f"Cannot convert {type(val).__name__} to float: {val}")


def _safe_int(val) -> int:
    """Extract int from value, handling dict wrapping from LLM."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        return int(float(val))
    if isinstance(val, dict):
        for key in ("quantity", "size", "shares", "count", "value"):
            if key in val:
                return int(float(val[key]))
        for v in val.values():
            if isinstance(v, (int, float)):
                return int(v)
        raise ValueError(f"Cannot extract int from dict: {val}")
    raise TypeError(f"Cannot convert {type(val).__name__} to int: {val}")


# =============================================================================
# COMMON EXECUTION HELPER
# =============================================================================

async def _run_order(
    executor, params: dict, *,
    required: list[str],
    invoke: Callable,
    check_side: bool = True,
    estimate_cash: Callable = None,
    extra_validate: Callable = None,
    refresh: bool = True,
) -> Any:
    """
    Execute a stock order with standard checks.

    Args:
        required:       Required param names (error if missing)
        invoke:         Callable returning awaitable gateway call
        check_side:     Check PDT/cash on BUY side (False for direction-based)
        estimate_cash:  Callable returning estimated cost (float) or None
        extra_validate: Callable returning error dict or None
        refresh:        Refresh state after order
    """
    if not executor.gateway:
        return {"error": "broker not connected"}

    missing = [r for r in required if params.get(r) is None]
    if missing:
        return {"error": f"Required: {', '.join(required)}"}

    if extra_validate:
        err = extra_validate()
        if err:
            return err

    if check_side and params.get("side", "").upper() == "BUY":
        pdt = executor._check_pdt("BUY")
        if pdt:
            return pdt
        if estimate_cash:
            cost = estimate_cash()
            if cost is not None:
                cash_err = executor._check_cash(cost)
                if cash_err:
                    return cash_err

    result = await invoke()
    if refresh:
        await executor._refresh_state()
    return result


# =============================================================================
# CASH ESTIMATION HELPERS
# =============================================================================

def _cost_price(params, key="limit_price", qty_key="quantity"):
    """Estimate cost from a price param × quantity."""
    def _est():
        p, q = params.get(key), params.get(qty_key)
        return _safe_float(p) * _safe_int(q) if p is not None and q is not None else None
    return _est


def _cost_quote(executor, params, qty_key="quantity"):
    """Estimate cost from live ask × quantity."""
    def _est():
        try:
            q = executor.data_provider.get_quote(params.get("symbol"))
            qty = params.get(qty_key)
            if q and q.ask and qty:
                return float(q.ask) * int(qty)
        except Exception:
            pass
        return None
    return _est


def _cost_smart(executor, params, price_key="limit_price"):
    """Use price param if available, else fall back to quote."""
    def _est():
        if params.get(price_key):
            return _safe_float(params[price_key]) * _safe_int(params.get("quantity", 1))
        return _cost_quote(executor, params)()
    return _est


# =============================================================================
# BASIC ORDERS
# =============================================================================

async def handle_market_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_market_order(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"])),
        estimate_cash=_cost_quote(executor, params))


async def handle_limit_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "limit_price"],
        invoke=lambda: executor.gateway.place_limit_order(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["limit_price"])),
        estimate_cash=_cost_price(params, "limit_price"))


async def handle_stop_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "stop_price"],
        invoke=lambda: executor.gateway.place_stop_order(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["stop_price"])),
        estimate_cash=_cost_price(params, "stop_price"))


async def handle_stop_limit(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "stop_price", "limit_price"],
        invoke=lambda: executor.gateway.place_stop_limit(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["stop_price"]), _safe_float(params["limit_price"])),
        estimate_cash=_cost_price(params, "limit_price"))


async def handle_trailing_stop(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "quantity", "direction", "trail_percent"],
        invoke=lambda: executor.gateway.place_trailing_stop(
            params["symbol"], _safe_int(params["quantity"]),
            params["direction"].upper(),
            trail_percent=_safe_float(params["trail_percent"])),
        check_side=False)


async def handle_bracket_order(executor, params: dict) -> Any:
    direction = "LONG" if params.get("side", "").upper() == "BUY" else "SHORT"
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "limit_price",
                  "stop_loss", "take_profit"],
        invoke=lambda: executor.gateway.place_bracket_order(
            params["symbol"], _safe_int(params["quantity"]), direction,
            _safe_float(params["limit_price"]), _safe_float(params["stop_loss"]),
            _safe_float(params["take_profit"])),
        estimate_cash=_cost_price(params, "limit_price"))


# =============================================================================
# SPECIAL ORDERS (unique patterns)
# =============================================================================

async def handle_modify_stop(executor, params: dict) -> Any:
    """Modify existing stop — no PDT, no cash, no refresh."""
    if not executor.gateway:
        return {"error": "broker not connected"}
    if not params.get("order_id") or not params.get("new_stop_price"):
        return {"error": "Required: order_id, new_stop_price"}
    return await executor.gateway.modify_stop_price(
        _safe_int(params["order_id"]), _safe_float(params["new_stop_price"]))


async def handle_oca_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "quantity", "direction",
                  "stop_price", "target_price"],
        invoke=lambda: executor.gateway.place_oca(
            params["symbol"], _safe_int(params["quantity"]),
            params["direction"].upper(),
            _safe_float(params["stop_price"]), _safe_float(params["target_price"])),
        check_side=False)


async def handle_flatten_limits(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    result = await executor.gateway.flatten_limits()
    await executor._refresh_state()
    return result


# =============================================================================
# AUCTION ORDERS
# =============================================================================

async def handle_moc_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_market_on_close(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"])),
        estimate_cash=_cost_quote(executor, params))


async def handle_loc_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "limit_price"],
        invoke=lambda: executor.gateway.place_limit_on_close(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["limit_price"])),
        estimate_cash=_cost_price(params, "limit_price"))


async def handle_moo_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_market_on_open(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"])),
        estimate_cash=_cost_quote(executor, params))


async def handle_loo_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "limit_price"],
        invoke=lambda: executor.gateway.place_limit_on_open(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["limit_price"])),
        estimate_cash=_cost_price(params, "limit_price"))


# =============================================================================
# ADVANCED ORDERS
# =============================================================================

async def handle_trailing_stop_limit(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "quantity", "direction"],
        extra_validate=lambda: (
            {"error": "Must provide trail_amount or trail_percent"}
            if params.get("trail_amount") is None
            and params.get("trail_percent") is None
            else None),
        invoke=lambda: executor.gateway.place_trailing_stop_limit(
            params["symbol"], _safe_int(params["quantity"]),
            params["direction"].upper(),
            trail_amount=(_safe_float(params["trail_amount"])
                          if params.get("trail_amount") else None),
            trail_percent=(_safe_float(params["trail_percent"])
                           if params.get("trail_percent") else None),
            limit_offset=_safe_float(params.get("limit_offset", 0.10))),
        check_side=False)


async def handle_adaptive_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_adaptive(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            params.get("order_type", "MKT"),
            limit_price=(_safe_float(params["limit_price"])
                         if params.get("limit_price") else None),
            priority=params.get("priority", "Normal")),
        estimate_cash=_cost_smart(executor, params, "limit_price"))


async def handle_midprice_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_midprice(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            price_cap=(_safe_float(params["price_cap"])
                       if params.get("price_cap") else None)),
        estimate_cash=_cost_smart(executor, params, "price_cap"))


async def handle_relative_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_relative(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            offset=_safe_float(params.get("offset", 0.01)),
            limit_price=(_safe_float(params["limit_price"])
                         if params.get("limit_price") else None)),
        estimate_cash=_cost_smart(executor, params, "limit_price"))


async def handle_gtd_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "limit_price",
                  "good_till_date"],
        invoke=lambda: executor.gateway.place_limit_order_gtd(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["limit_price"]), params["good_till_date"]),
        estimate_cash=_cost_price(params, "limit_price"))


async def handle_fok_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "limit_price"],
        invoke=lambda: executor.gateway.place_fill_or_kill(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["limit_price"])),
        estimate_cash=_cost_price(params, "limit_price"))


async def handle_ioc_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity", "limit_price"],
        invoke=lambda: executor.gateway.place_immediate_or_cancel(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            _safe_float(params["limit_price"])),
        estimate_cash=_cost_price(params, "limit_price"))


# =============================================================================
# ALGO ORDERS
# =============================================================================

async def handle_vwap_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_vwap(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            max_pct_volume=_safe_float(params.get("max_pct_volume", 25.0))),
        estimate_cash=_cost_quote(executor, params))


async def handle_twap_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_twap(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"]),
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            randomize_pct=_safe_float(params.get("randomize_pct", 55.0))),
        estimate_cash=_cost_quote(executor, params))


async def handle_iceberg_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "total_quantity",
                  "display_size", "limit_price"],
        invoke=lambda: executor.gateway.place_iceberg_order(
            params["symbol"], params["side"].upper(),
            _safe_int(params["total_quantity"]),
            _safe_int(params["display_size"]), _safe_float(params["limit_price"])),
        estimate_cash=_cost_price(params, "limit_price", "total_quantity"))


async def handle_snap_mid_order(executor, params: dict) -> Any:
    return await _run_order(executor, params,
        required=["symbol", "side", "quantity"],
        invoke=lambda: executor.gateway.place_snap_to_midpoint(
            params["symbol"], params["side"].upper(), _safe_int(params["quantity"])),
        estimate_cash=_cost_quote(executor, params))


# =============================================================================
# HANDLER REGISTRY
# =============================================================================

HANDLERS = {
    # Basic
    "market_order": handle_market_order,
    "limit_order": handle_limit_order,
    "stop_order": handle_stop_order,
    "stop_limit": handle_stop_limit,
    "trailing_stop": handle_trailing_stop,
    "bracket_order": handle_bracket_order,
    # Special
    "modify_stop": handle_modify_stop,
    "oca_order": handle_oca_order,
    "flatten_limits": handle_flatten_limits,
    # Auction
    "moc_order": handle_moc_order,
    "loc_order": handle_loc_order,
    "moo_order": handle_moo_order,
    "loo_order": handle_loo_order,
    # Advanced
    "trailing_stop_limit": handle_trailing_stop_limit,
    "adaptive_order": handle_adaptive_order,
    "midprice_order": handle_midprice_order,
    "relative_order": handle_relative_order,
    "gtd_order": handle_gtd_order,
    "fok_order": handle_fok_order,
    "ioc_order": handle_ioc_order,
    # Algo
    "vwap_order": handle_vwap_order,
    "twap_order": handle_twap_order,
    "iceberg_order": handle_iceberg_order,
    "snap_mid_order": handle_snap_mid_order,
}
