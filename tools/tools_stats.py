"""Observability and performance stats tool handlers."""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# No daily_store persistence — stats are derived from in-memory live_state
def _no_op_read(*args, **kwargs):
    return []

read_events = _no_op_read
read_trades = _no_op_read


def _load_decisions_today() -> list[dict]:
    """No persistence — returns empty list."""
    return []


def _compute_trade_stats(decisions: list[dict]) -> dict:
    """Compute trade-level stats from decision records."""
    order_actions = {
        "market_order", "limit_order", "stop_order", "stop_limit",
        "trailing_stop", "bracket_order", "plan_order", "enter_option",
        "buy_option", "covered_call", "cash_secured_put", "protective_put",
        "vertical_spread", "iron_condor", "close_option", "roll_option",
        "moc_order", "loc_order", "adaptive_order", "midprice_order",
        "vwap_order", "twap_order",
    }
    plan_with_execute = set()
    attempted_plan = set()
    direct_orders = []
    attempted_direct = []

    for d in decisions:
        action = d.get("action", "")
        params = d.get("params", {})
        success = d.get("success", False)

        # plan_order / enter_option with execute=true count attempts and successes
        if action in ("plan_order", "enter_option") and params.get("execute"):
            attempted_plan.add(d.get("turn", 0))
            if success:
                plan_with_execute.add(d.get("turn", 0))
        elif action in order_actions:
            attempted_direct.append(d)
            if success:
                direct_orders.append(d)

    total_orders = len(direct_orders) + len(plan_with_execute)
    attempted_orders = len(attempted_direct) + len(attempted_plan)
    successful = total_orders

    # Unique symbols traded
    symbols = set()
    for d in direct_orders:
        sym = d.get("params", {}).get("symbol")
        if sym:
            symbols.add(sym.upper())

    return {
        "orders_today": total_orders,
        "attempted_orders": attempted_orders,
        "successful_orders": successful,
        "symbols_traded": sorted(symbols),
        "unique_symbols": len(symbols),
    }


def _compute_action_breakdown(decisions: list[dict]) -> dict:
    """Count actions by category."""
    research = 0
    orders = 0
    planning = 0
    account = 0
    other = 0

    _research = {
        "quote", "candles", "atr", "fundamentals", "earnings",
        "iv_info", "news", "analysts", "extended_fundamentals",
        "institutional_data", "insider_data", "peer_comparison", "screen",
    }
    _orders = {
        "market_order", "limit_order", "stop_order", "stop_limit",
        "trailing_stop", "bracket_order", "plan_order", "enter_option",
        "buy_option", "covered_call", "cash_secured_put", "close_option",
        "roll_option", "adaptive_order", "midprice_order",
    }
    _planning = {"prepare_session", "calculate_size", "instrument_selector"}
    _account = {"positions", "account", "open_orders", "get_position", "cancel_order", "cancel_stops"}
    research_actions = _research
    order_actions = _orders
    planning_actions = _planning
    account_actions = _account

    for d in decisions:
        action = d.get("action", "")
        if action in research_actions:
            research += 1
        elif action in order_actions:
            orders += 1
        elif action in planning_actions:
            planning += 1
        elif action in account_actions:
            account += 1
        else:
            other += 1

    return {
        "research": research,
        "orders": orders,
        "planning": planning,
        "account": account,
        "other": other,
        "total": len(decisions),
    }


def _compute_confidence_calibration(decisions: list[dict]) -> dict:
    """Compute confidence calibration metrics from decision records.

    Uses additive confidence fields written by the agent loop:
      decision["confidence"] = {
        "band": "low|medium|high",
        "why": str,
        "evidence": list[str],
        "unknowns": list[str],
      }
    """
    order_actions = {
        "market_order", "limit_order", "stop_order", "stop_limit",
        "trailing_stop", "bracket_order", "plan_order", "enter_option",
        "buy_option", "covered_call", "cash_secured_put", "close_option",
    }

    bands = ("low", "medium", "high")
    by_band = {
        band: {"total": 0, "success": 0, "failure": 0}
        for band in bands
    }
    action_breakdown = {
        "tool_actions_with_confidence": 0,
        "meta_actions_with_confidence": 0,
        "missing_confidence": 0,
    }
    high_conf_failures = 0
    low_conf_successes = 0
    unknown_count = 0
    evidence_count = 0
    unknown_records = 0
    evidence_records = 0

    for d in decisions:
        action = d.get("action", "")
        success = bool(d.get("success", False))
        conf = d.get("confidence")

        is_meta = action in ("think", "feedback", "done", "wait_rejected")
        if not isinstance(conf, dict):
            action_breakdown["missing_confidence"] += 1
            continue

        band = str(conf.get("band", "")).strip().lower()
        if band not in by_band:
            action_breakdown["missing_confidence"] += 1
            continue

        if is_meta:
            action_breakdown["meta_actions_with_confidence"] += 1
        else:
            action_breakdown["tool_actions_with_confidence"] += 1

        by_band[band]["total"] += 1
        if success:
            by_band[band]["success"] += 1
        else:
            by_band[band]["failure"] += 1

        if band == "high" and not success:
            high_conf_failures += 1
        if band == "low" and success:
            low_conf_successes += 1

        evidence = conf.get("evidence", [])
        if isinstance(evidence, list):
            evidence_count += len([e for e in evidence if str(e).strip()])
            evidence_records += 1

        unknowns = conf.get("unknowns", [])
        if isinstance(unknowns, list):
            unknown_count += len([u for u in unknowns if str(u).strip()])
            unknown_records += 1

    # Success rates by confidence band
    rates = {}
    for band, row in by_band.items():
        total = row["total"]
        rates[band] = round(row["success"] / total * 100, 1) if total > 0 else None

    # Focused view for execution-heavy actions
    execution_rows = [
        d for d in decisions
        if d.get("action") in order_actions or d.get("action") in ("plan_order", "enter_option")
    ]
    execution_conf = 0
    execution_high_fail = 0
    execution_low_success = 0
    for d in execution_rows:
        conf = d.get("confidence")
        if not isinstance(conf, dict):
            continue
        band = str(conf.get("band", "")).strip().lower()
        if band not in bands:
            continue
        execution_conf += 1
        ok = bool(d.get("success", False))
        if band == "high" and not ok:
            execution_high_fail += 1
        if band == "low" and ok:
            execution_low_success += 1

    return {
        "coverage": {
            "records_with_confidence": sum(v["total"] for v in by_band.values()),
            "records_missing_confidence": action_breakdown["missing_confidence"],
            "tool_actions_with_confidence": action_breakdown["tool_actions_with_confidence"],
            "meta_actions_with_confidence": action_breakdown["meta_actions_with_confidence"],
        },
        "by_band": by_band,
        "success_rate_by_band_pct": rates,
        "calibration_flags": {
            "high_confidence_failures": high_conf_failures,
            "low_confidence_successes": low_conf_successes,
            "execution_high_confidence_failures": execution_high_fail,
            "execution_low_confidence_successes": execution_low_success,
        },
        "signal_quality": {
            "avg_evidence_items": round(evidence_count / evidence_records, 2) if evidence_records else 0,
            "avg_unknown_items": round(unknown_count / unknown_records, 2) if unknown_records else 0,
        },
    }


async def handle_stats(executor, params: dict) -> Any:
    """
    Comprehensive performance stats for the current session.

    Returns: trade stats, P&L, LLM costs, action breakdown, position health.
    """
    result = {}

    # 1. Today's decisions
    decisions = _load_decisions_today()
    result["action_breakdown"] = _compute_action_breakdown(decisions)
    result["trade_stats"] = _compute_trade_stats(decisions)
    result["confidence"] = _compute_confidence_calibration(decisions)

    # 2. P&L from LiveState
    try:
        from data.live_state import get_live_state
        state = get_live_state()
        with state._lock:
            positions = list(state._positions.values())
            orders = list(state._orders.values())
            daily_pnl = state._daily_pnl
            realized_pnl = state._realized_pnl
            unrealized_pnl = state._unrealized_pnl
            cash = state._cash
            net_liq = state._net_liq

        # Position health
        winners = sum(1 for p in positions if p.unrealized_pnl > 0)
        losers = sum(1 for p in positions if p.unrealized_pnl < 0)
        flat = sum(1 for p in positions if p.unrealized_pnl == 0)

        # Unprotected positions
        unprotected = 0
        for p in positions:
            pos_orders = state.get_orders_for(p.symbol)
            if p.is_long and not any(o.action == "SELL" for o in pos_orders):
                unprotected += 1
            elif p.is_short and not any(o.action == "BUY" for o in pos_orders):
                unprotected += 1

        best_pos = max(positions, key=lambda p: p.unrealized_pnl) if positions else None
        worst_pos = min(positions, key=lambda p: p.unrealized_pnl) if positions else None

        result["pnl"] = {
            "daily_pnl": round(daily_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
        }
        result["positions"] = {
            "total": len(positions),
            "winners": winners,
            "losers": losers,
            "flat": flat,
            "win_rate_pct": round(winners / (winners + losers) * 100, 1) if (winners + losers) > 0 else 0,
            "unprotected": unprotected,
            "open_orders": len(orders),
            "best": {
                "symbol": best_pos.symbol,
                "pnl": round(best_pos.unrealized_pnl, 2),
            } if best_pos else None,
            "worst": {
                "symbol": worst_pos.symbol,
                "pnl": round(worst_pos.unrealized_pnl, 2),
            } if worst_pos else None,
        }
        result["account"] = {
            "cash": round(cash, 2),
            "net_liq": round(net_liq, 2),
            "cash_pct": round(cash / net_liq * 100, 1) if net_liq > 0 else 0,
        }

        # 2b. Profit protection metrics
        try:
            from data.profit_metrics import compute_session_metrics
            closed_trades = list(state._closed_trades_today)
            result["profit_protection"] = compute_session_metrics(
                closed_trades, state._hwm_tracker, state._equity_tracker,
            )
        except Exception as e:
            result["profit_protection"] = {"error": str(e)}
    except Exception as e:
        result["pnl"] = {"error": str(e)}
        result["positions"] = {"error": str(e)}

    # 3. LLM cost summary
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

    # 4. Cycles today
    cycle_ids = set(d.get("cycle_id", 0) for d in decisions)
    result["session"] = {
        "cycles_today": len(cycle_ids),
        "total_turns": len(decisions),
    }

    return result


async def handle_daily_summary(executor, params: dict) -> Any:
    """
    Generate and persist a comprehensive daily summary.

    Writes to logs/daily_summary.json with append-by-date structure.
    Returns the same summary to the agent.
    """
    today_str = date.today().isoformat()

    # Gather all stats (reuse stats handler logic)
    stats = await handle_stats(executor, params)

    # Add timestamp
    summary = {
        "date": today_str,
        "generated_at": datetime.now().isoformat(),
        **stats,
    }

    # No persistence — in-memory only
    summary["persisted"] = False

    return summary


async def handle_review_trades(executor, params: dict) -> Any:
    """
    Review closed trades from recent days.
    The agent calls this to learn from past wins/losses.

    Optional: days (default 3) — how many days back to look.
    Optional: sort (default "efficiency") — "efficiency", "pnl", or "recency".
    Optional: symbol — filter to a specific symbol.
    Returns: trade-by-trade breakdown + summary stats, sorted by efficiency.
    """
    import statistics as stats_mod

    sort_by = params.get("sort", "efficiency")
    symbol_filter = params.get("symbol")
    
    # Get closed trades from in-memory live_state (no disk persistence)
    all_trades = []
    try:
        from data.live_state import get_live_state
        ls = get_live_state()
        for record in getattr(ls, '_closed_trades_today', []):
            record = dict(record)  # copy
            record["date"] = date.today().isoformat()
            if "efficiency" not in record:
                hh = record.get("hold_hours")
                pp = record.get("pnl_pct")
                if hh and hh > 0.017 and pp is not None:
                    record["efficiency"] = round(pp / hh, 2)
                else:
                    record["efficiency"] = None
            if "pnl_per_hour" not in record:
                hh = record.get("hold_hours")
                pnl = record.get("pnl")
                if hh and hh > 0.017 and pnl is not None:
                    record["pnl_per_hour"] = round(pnl / hh, 2)
                else:
                    record["pnl_per_hour"] = None
            all_trades.append(record)
    except Exception as e:
        logger.debug(f"Failed to load closed trades: {e}")
    
    if not all_trades:
        return {"message": f"No closed trades found in the last {days_back} days", "trades": []}
    
    # Apply symbol filter
    if symbol_filter:
        all_trades = [t for t in all_trades if t.get("symbol", "").upper() == symbol_filter.upper()]
        if not all_trades:
            return {"message": f"No closed trades for {symbol_filter} in the last {days_back} days", "trades": []}

    # Sort
    if sort_by == "efficiency":
        all_trades.sort(key=lambda t: t.get("efficiency") if t.get("efficiency") is not None else 0)
    elif sort_by == "pnl":
        all_trades.sort(key=lambda t: t.get("pnl", 0))
    else:  # recency
        all_trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)

    # Summary stats
    wins = [t for t in all_trades if t.get('pnl', 0) > 0]
    losses = [t for t in all_trades if t.get('pnl', 0) <= 0]
    total_pnl = sum(t.get('pnl', 0) for t in all_trades)
    avg_win = (sum(t['pnl'] for t in wins) / len(wins)) if wins else 0
    avg_loss = (sum(t['pnl'] for t in losses) / len(losses)) if losses else 0
    win_rate = (len(wins) / len(all_trades) * 100) if all_trades else 0
    
    # Profit factor
    gross_wins = sum(t['pnl'] for t in wins) if wins else 0
    gross_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf') if gross_wins > 0 else 0
    
    # Efficiency stats
    eff_values = [t["efficiency"] for t in all_trades if t.get("efficiency") is not None]
    
    # Best / worst by efficiency
    trades_with_eff = [t for t in all_trades if t.get("efficiency") is not None]
    best_eff = max(trades_with_eff, key=lambda t: t["efficiency"]) if trades_with_eff else None
    worst_eff = min(trades_with_eff, key=lambda t: t["efficiency"]) if trades_with_eff else None
    
    # Best / worst by P&L
    best = max(all_trades, key=lambda t: t.get('pnl', 0))
    worst = min(all_trades, key=lambda t: t.get('pnl', 0))
    
    # Expectancy: the expected $ per trade
    # = (win_rate × avg_win) + (loss_rate × avg_loss)  [avg_loss is negative]
    if all_trades:
        expectancy = (len(wins) / len(all_trades)) * avg_win + (len(losses) / len(all_trades)) * avg_loss
    else:
        expectancy = 0
    # Win/loss size ratio (how much bigger winners are vs losers)
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0

    summary = {
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "win_loss_ratio": round(wl_ratio, 2) if wl_ratio != float('inf') else "inf",
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
        "gross_wins": round(gross_wins, 2),
        "gross_losses": round(gross_losses, 2),
    }
    
    if eff_values:
        summary["avg_efficiency_pct_hr"] = round(stats_mod.mean(eff_values), 2)
        summary["median_efficiency_pct_hr"] = round(stats_mod.median(eff_values), 2)
    
    # --- Max drawdown (peak-to-trough cumulative P&L) ---
    pnl_values = [t.get('pnl', 0) for t in all_trades]
    if pnl_values:
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_values:
            cum_pnl += pnl
            if cum_pnl > peak:
                peak = cum_pnl
            dd = peak - cum_pnl
            if dd > max_dd:
                max_dd = dd
        summary["max_drawdown"] = round(max_dd, 2)
    
    # --- Pseudo-Sharpe (mean return / std dev of returns) ---
    if len(pnl_values) >= 2:
        mean_pnl = stats_mod.mean(pnl_values)
        std_pnl = stats_mod.stdev(pnl_values)
        summary["sharpe_ratio"] = round(mean_pnl / std_pnl, 2) if std_pnl > 0 else None
    
    # --- Win/loss streak analysis ---
    if pnl_values:
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        streak_type = None  # 'win' or 'loss'
        for pnl in pnl_values:
            is_win = pnl > 0
            if streak_type is None:
                streak_type = 'win' if is_win else 'loss'
                current_streak = 1
            elif (is_win and streak_type == 'win') or (not is_win and streak_type == 'loss'):
                current_streak += 1
            else:
                streak_type = 'win' if is_win else 'loss'
                current_streak = 1
            if streak_type == 'win':
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)
        summary["max_win_streak"] = max_win_streak
        summary["max_loss_streak"] = max_loss_streak
        summary["current_streak"] = f"{current_streak} {'win' if streak_type == 'win' else 'loss'}{'s' if current_streak > 1 else ''}"
    
    result = {
        "days_reviewed": days_back,
        "total_trades": len(all_trades),
        "sorted_by": sort_by,
        "summary": summary,
        "best_trade": {
            "symbol": best.get("symbol"),
            "pnl": best.get("pnl"),
            "efficiency": best.get("efficiency"),
            "direction": best.get("direction"),
            "date": best.get("date"),
        },
        "worst_trade": {
            "symbol": worst.get("symbol"),
            "pnl": worst.get("pnl"),
            "efficiency": worst.get("efficiency"),
            "direction": worst.get("direction"),
            "date": worst.get("date"),
        },
        "trades": all_trades,
    }
    
    if best_eff:
        result["most_efficient"] = {
            "symbol": best_eff.get("symbol"),
            "efficiency": best_eff.get("efficiency"),
            "pnl": best_eff.get("pnl"),
            "direction": best_eff.get("direction"),
            "date": best_eff.get("date"),
        }
    if worst_eff:
        result["least_efficient"] = {
            "symbol": worst_eff.get("symbol"),
            "efficiency": worst_eff.get("efficiency"),
            "pnl": worst_eff.get("pnl"),
            "direction": worst_eff.get("direction"),
            "date": worst_eff.get("date"),
        }
    
    return result



def compute_reliability_stats() -> dict:
    """Compute execution reliability KPIs from today's decisions and live state."""
    decisions = _load_decisions_today()

    # --- Stop placement stats ---
    stop_attempts = 0
    stop_successes = 0
    stop_failures = 0
    partial_fills = 0
    auto_flattens = 0
    duplicate_suppressions = 0

    for d in decisions:
        action = d.get("action", "")
        result_str = str(d.get("result", ""))
        params = d.get("params", {})

        # plan_order with execute=true: check stop outcome
        if action == "plan_order" and params.get("execute"):
            if "stop" in result_str.lower():
                stop_attempts += 1
                if '"partial": true' in result_str.lower() or '"partial":true' in result_str.lower():
                    stop_failures += 1
                    partial_fills += 1
                elif '"stop"' in result_str and '"success": true' in result_str:
                    stop_successes += 1
                elif '"stop"' in result_str and '"success": false' in result_str:
                    stop_failures += 1

        # Check for auto-flatten events
        if "auto_flatten" in result_str.lower():
            auto_flattens += 1

        # Check for duplicate suppression
        if "duplicate_suppressed" in result_str:
            duplicate_suppressions += 1

    stop_success_rate = round(stop_successes / stop_attempts * 100, 1) if stop_attempts > 0 else None

    # --- Deferred stop queue depth ---
    deferred_depth = 0

    # --- Farm health ---
    farm_health = {}
    try:
        from data.live_state import get_live_state
        farm_health = get_live_state().farm_health
    except Exception:
        pass

    return {
        "stop_attempts": stop_attempts,
        "stop_successes": stop_successes,
        "stop_failures": stop_failures,
        "stop_success_rate_pct": stop_success_rate,
        "partial_fills": partial_fills,
        "auto_flattens": auto_flattens,
        "duplicate_suppressions": duplicate_suppressions,
        "deferred_stop_queue_depth": deferred_depth,
        "farm_health": farm_health,
    }


HANDLERS = {
    "stats": handle_stats,
    "daily_summary": handle_daily_summary,
    "review_trades": handle_review_trades,


}
