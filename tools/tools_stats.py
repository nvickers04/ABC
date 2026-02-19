"""Observability and performance stats — broker P&L + LLM costs only."""

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# handle_stats — the only handler that matters
# ---------------------------------------------------------------------------

async def handle_stats(executor, params: dict) -> Any:
    """Broker P&L, position health, and LLM cost summary."""
    result: dict[str, Any] = {}

    # — P&L from broker —
    try:
        gw = executor.gateway
        summary = await gw.get_account_summary()
        positions_list = await gw.get_positions()
        orders_list = await gw.get_open_orders()

        cash = summary.get("totalcashvalue", 0)
        net_liq = summary.get("netliquidation", 0)

        winners = sum(1 for p in positions_list if p.get("unrealized_pnl", 0) > 0)
        losers = sum(1 for p in positions_list if p.get("unrealized_pnl", 0) < 0)
        best = max(positions_list, key=lambda p: p.get("unrealized_pnl", 0)) if positions_list else None
        worst = min(positions_list, key=lambda p: p.get("unrealized_pnl", 0)) if positions_list else None

        result["pnl"] = {
            "daily_pnl": round(summary.get("dailypnl", 0), 2),
            "realized_pnl": round(summary.get("realizedpnl", 0), 2),
            "unrealized_pnl": round(summary.get("unrealizedpnl", 0), 2),
        }
        result["positions"] = {
            "total": len(positions_list),
            "winners": winners,
            "losers": losers,
            "win_rate_pct": round(winners / (winners + losers) * 100, 1) if (winners + losers) else 0,
            "open_orders": len(orders_list),
            "best": {"symbol": best.get("symbol", "?"), "pnl": round(best.get("unrealized_pnl", 0), 2)} if best else None,
            "worst": {"symbol": worst.get("symbol", "?"), "pnl": round(worst.get("unrealized_pnl", 0), 2)} if worst else None,
        }
        result["account"] = {
            "cash": round(cash, 2),
            "net_liq": round(net_liq, 2),
            "cash_pct": round(cash / net_liq * 100, 1) if net_liq else 0,
        }
    except Exception as e:
        result["pnl"] = {"error": str(e)}
        result["positions"] = {"error": str(e)}

    # — LLM cost summary —
    try:
        from data.cost_tracker import get_cost_tracker
        tracker = get_cost_tracker()
        budget = tracker.get_budget_summary()
        result["llm_costs"] = {
            "today_cost": round(budget.today_llm_cost, 4),
            "today_calls": budget.today_llm_calls,
            "total_cost": round(budget.total_llm_cost, 4),
            "total_calls": budget.total_llm_calls,
            "budget_remaining": round(budget.budget_remaining, 2),
            "budget_status": budget.budget_status,
            "roi_pct": round(tracker.get_roi(), 1),
        }
    except Exception as e:
        result["llm_costs"] = {"error": str(e)}

    return result


# ---------------------------------------------------------------------------
# handle_review_trades — thin wrapper over broker executions
# ---------------------------------------------------------------------------

async def handle_review_trades(executor, params: dict) -> Any:
    """Return today's broker executions.  Optional: symbol filter."""
    symbol_filter = params.get("symbol")
    try:
        fills = await executor.gateway.get_recent_executions()
    except Exception as e:
        return {"error": str(e)}

    trades = [
        {
            "symbol": f.get("symbol", "?"),
            "side": f.get("side", "?"),
            "shares": f.get("shares", 0),
            "price": f.get("price", 0),
            "time": f.get("time", ""),
        }
        for f in fills
    ]
    if symbol_filter:
        trades = [t for t in trades if t["symbol"].upper() == symbol_filter.upper()]

    total_pnl = sum(t.get("pnl", 0) for t in trades)
    return {
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# daily_summary is just stats with a date stamp
# ---------------------------------------------------------------------------

async def handle_daily_summary(executor, params: dict) -> Any:
    """Alias for stats with today's date attached."""
    stats = await handle_stats(executor, params)
    stats["date"] = date.today().isoformat()
    return stats


HANDLERS = {
    "stats": handle_stats,
    "daily_summary": handle_daily_summary,
    "review_trades": handle_review_trades,
}
