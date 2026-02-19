"""Account state and order management tool handlers."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def handle_cancel_order(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    order_id = params.get("order_id")
    if not order_id:
        return {"error": "order_id required"}
    return await executor.gateway.cancel_order(order_id)


async def handle_cancel_stops(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    return await executor.gateway.cancel_stops(symbol)


async def handle_cancel_all_orphans(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    # Protect pending entry orders: entry-type orders (LMT/MIDPRICE/MKT etc.)
    # for symbols with no position are intentional — don't cancel them.
    exclude = set()
    try:
        from data.live_state import get_live_state
        ls = get_live_state()
        orphans = ls.get_orphan_orders()
        entry_types = {'LMT', 'MKT', 'MIDPRICE', 'REL', 'MOC', 'MOO', 'LOC', 'LOO'}
        for o in orphans:
            if o.order_type in entry_types or o.sec_type in ('OPT', 'BAG'):
                exclude.add(o.symbol)
    except Exception:
        pass
    result = await executor.gateway.cancel_all_orphans(exclude_symbols=exclude)
    await executor._refresh_state()
    return result


async def handle_positions(executor, params: dict) -> Any:
    from data.live_state import get_live_state
    ls = get_live_state()
    with ls._lock:
        result = []
        for key, pos in ls._positions.items():
            pph = pos.pnl_per_hour
            eff = pos.efficiency
            result.append({
                "symbol": key,
                "quantity": pos.quantity,
                "avg_cost": round(pos.avg_cost, 4),
                "market_price": round(pos.market_price, 4),
                "market_value": round(pos.market_value, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "pnl_pct": round(pos.pnl_percent, 2),
                "pnl_per_hour": round(pph, 2) if pph is not None else None,
                "efficiency_pct_hr": round(eff, 2) if eff is not None else None,
                "direction": pos.direction,
                "hold_time": pos.format_hold_time(),
                "open_orders": len(ls.get_orders_for(key))
            })
    return {"positions": result, "count": len(result)}


async def handle_account(executor, params: dict) -> Any:
    from data.live_state import get_live_state
    ls = get_live_state()
    with ls._lock:
        available = ls._available_funds or ls._cash
    # Try to get freshest available_funds from broker
    if executor.gateway:
        try:
            if hasattr(executor.gateway, 'available_funds') and executor.gateway.available_funds > 0:
                available = executor.gateway.available_funds
        except Exception:
            pass
    with ls._lock:
        return {
            "net_liq": round(ls._net_liq, 2),
            "available_funds": round(available, 2),
            "cash": round(ls._cash, 2),
            "daily_pnl": round(ls._daily_pnl, 2),
            "realized_pnl": round(ls._realized_pnl, 2),
            "unrealized_pnl": round(ls._unrealized_pnl, 2),
            "position_count": len(ls._positions),
            "open_order_count": len(ls._orders),
            "note": "CASH-ONLY account. Use available_funds for sizing."
        }


async def handle_open_orders(executor, params: dict) -> Any:
    from data.live_state import get_live_state
    ls = get_live_state()
    with ls._lock:
        result = []
        for oid, order in ls._orders.items():
            result.append({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "action": order.action,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "status": order.status,
                "limit_price": order.limit_price,
                "aux_price": order.aux_price,
                "sec_type": order.sec_type,
                "filled_qty": order.filled_qty,
                "remaining_qty": order.remaining_qty
            })
    return {"orders": result, "count": len(result)}


async def handle_get_position(executor, params: dict) -> Any:
    from data.live_state import get_live_state
    ls = get_live_state()
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    with ls._lock:
        pos = ls._positions.get(symbol)
        if not pos:
            matches = {k: v for k, v in ls._positions.items() 
                      if k == symbol or k.startswith(symbol + '_')}
            if not matches:
                return {"error": f"No position found for {symbol}"}
            result = []
            for key, p in matches.items():
                pph = p.pnl_per_hour
                eff = p.efficiency
                result.append({
                    "symbol": key,
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 4),
                    "market_price": round(p.market_price, 4),
                    "market_value": round(p.market_value, 2),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                    "pnl_pct": round(p.pnl_percent, 2),
                    "pnl_per_hour": round(pph, 2) if pph is not None else None,
                    "efficiency_pct_hr": round(eff, 2) if eff is not None else None,
                    "direction": p.direction,
                    "hold_time": p.format_hold_time(),
                    "orders": [{"order_id": o.order_id, "action": o.action, 
                                "status": o.status, "type": o.order_type}
                               for o in ls.get_orders_for(key)]
                })
            return {"positions": result, "count": len(result)}
        else:
            orders = ls.get_orders_for(symbol)
            pph = pos.pnl_per_hour
            eff = pos.efficiency
            return {
                "symbol": symbol,
                "quantity": pos.quantity,
                "avg_cost": round(pos.avg_cost, 4),
                "market_price": round(pos.market_price, 4),
                "market_value": round(pos.market_value, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "pnl_pct": round(pos.pnl_percent, 2),
                "pnl_per_hour": round(pph, 2) if pph is not None else None,
                "efficiency_pct_hr": round(eff, 2) if eff is not None else None,
                "direction": pos.direction,
                "hold_time": pos.format_hold_time(),
                "orders": [{"order_id": o.order_id, "action": o.action, 
                            "status": o.status, "type": o.order_type}
                           for o in orders]
            }


HANDLERS = {
    "cancel_order": handle_cancel_order,
    "cancel_stops": handle_cancel_stops,
    "cancel_all_orphans": handle_cancel_all_orphans,
    "positions": handle_positions,
    "account": handle_account,
    "open_orders": handle_open_orders,
    "get_position": handle_get_position,
}
