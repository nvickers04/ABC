"""Account state and order management tool handlers — queries broker directly (no LiveState)."""

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
    result = await executor.gateway.cancel_all_orphans()
    return result


async def handle_positions(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    positions = await executor.gateway.get_positions()
    result = []
    for p in positions:
        qty = p.get("quantity", 0)
        avg = p.get("avg_cost", 0)
        mkt = p.get("market_price", 0)
        mkt_val = p.get("market_value", 0)
        upnl = p.get("unrealized_pnl", 0)
        pnl_pct = ((mkt - avg) / avg * 100) if avg > 0 and qty > 0 else (
            (avg - mkt) / avg * 100 if avg > 0 and qty < 0 else 0
        )
        result.append({
            "symbol": p.get("symbol"),
            "quantity": qty,
            "avg_cost": round(avg, 4),
            "market_price": round(mkt, 4),
            "market_value": round(mkt_val, 2),
            "unrealized_pnl": round(upnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "direction": "LONG" if qty > 0 else "SHORT",
            "sec_type": p.get("sec_type", "STK"),
        })
    return {"positions": result, "count": len(result)}


async def handle_account(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    summary = await executor.gateway.get_account_summary()
    if "error" in summary:
        return summary
    cash = summary.get("totalcashvalue", 0)
    available = summary.get("availablefunds", 0) or cash
    return {
        "cash": round(cash, 2),
        "available_funds": round(available, 2),
        "daily_pnl": round(summary.get("dailypnl", 0), 2),
        "realized_pnl": round(summary.get("realizedpnl", 0), 2),
        "unrealized_pnl": round(summary.get("unrealizedpnl", 0), 2),
        "note": "CASH-ONLY account. Use available_funds or cash for sizing — ignore margin."
    }


async def handle_open_orders(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    orders = await executor.gateway.get_open_orders()
    return {"orders": orders, "count": len(orders)}


async def handle_get_position(executor, params: dict) -> Any:
    if not executor.gateway:
        return {"error": "broker not connected"}
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    positions = await executor.gateway.get_positions()
    matches = [p for p in positions if p.get("symbol", "").upper() == symbol.upper()]
    if not matches:
        return {"error": f"No position found for {symbol}"}
    if len(matches) == 1:
        p = matches[0]
        qty = p.get("quantity", 0)
        avg = p.get("avg_cost", 0)
        mkt = p.get("market_price", 0)
        return {
            "symbol": p.get("symbol"),
            "quantity": qty,
            "avg_cost": round(avg, 4),
            "market_price": round(mkt, 4),
            "market_value": round(p.get("market_value", 0), 2),
            "unrealized_pnl": round(p.get("unrealized_pnl", 0), 2),
            "direction": "LONG" if qty > 0 else "SHORT",
            "sec_type": p.get("sec_type", "STK"),
        }
    return {"positions": matches, "count": len(matches)}


HANDLERS = {
    "cancel_order": handle_cancel_order,
    "cancel_stops": handle_cancel_stops,
    "cancel_all_orphans": handle_cancel_all_orphans,
    "positions": handle_positions,
    "account": handle_account,
    "open_orders": handle_open_orders,
    "get_position": handle_get_position,
}
