"""Shared trader tool smoke helpers (used by ``scripts/smoke_*.py``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tools.smoke_manifest import NEVER_AUTOTEST, broker_mutating_names, safe_names
from tools.tools_executor import ToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SmokeContext:
    symbol: str
    sample_call: dict | None = None
    sample_put: dict | None = None
    wm_clear: list[tuple[str, int]] = field(default_factory=list)
    last: float | None = None


def _norm_exp(exp: str | None) -> str:
    if not exp:
        return ""
    s = str(exp).strip()
    if "-" in s:
        return s.replace("-", "")
    return s


async def warm_option_samples(executor: ToolExecutor, ctx: SmokeContext) -> None:
    tr = await executor.execute(
        "option_chain",
        {
            "symbol": ctx.symbol,
            "dte_min": 7,
            "dte_max": 200,
            "limit": 32,
        },
    )
    payload = tr.data if isinstance(tr.data, dict) else {}
    for c in payload.get("contracts") or []:
        side = (c.get("side") or "").lower()
        if side == "call" and ctx.sample_call is None:
            ctx.sample_call = c
        elif side == "put" and ctx.sample_put is None:
            ctx.sample_put = c
        if ctx.sample_call and ctx.sample_put:
            break


def params_for_safe_tool(name: str, ctx: SmokeContext) -> dict | None:
    sym = ctx.symbol
    if name in (
        "quote",
        "fundamentals",
        "earnings",
        "atr",
        "iv_info",
        "news",
        "analysts",
        "extended_fundamentals",
        "institutional_data",
        "insider_data",
        "peer_comparison",
        "chart_intraday",
        "chart_swing",
        "chart_full",
        "chart_quick",
        "signal_breakdown",
    ):
        return {"symbol": sym}
    if name == "candles":
        return {"symbol": sym, "days": 5, "resolution": "D"}
    if name in ("market_hours", "budget", "economic_calendar"):
        return {}
    if name == "briefing":
        return {"detail": "summary"}
    if name == "prior_research":
        return {}
    if name in ("research_engine", "trader_rules"):
        return {"action": "status"}
    if name in ("execution_status", "open_hypotheses"):
        return {}
    if name == "research":
        return {
            "query": f"{sym} one-sentence business summary (tool smoke)",
            "deep": False,
        }
    if name in ("stats", "daily_summary", "review_trades"):
        return {}
    if name in ("account", "positions", "open_orders", "refresh_state"):
        return {}
    if name == "get_position":
        return {"symbol": sym}
    if name == "calculate_size":
        return {"symbol": sym, "side": "BUY", "risk_per_trade_pct": 0.1, "max_position_pct": 1.0}
    if name == "plan_order":
        return {
            "symbol": sym,
            "side": "BUY",
            "quantity": 1,
            "intent": "entry",
            "urgency": "low",
        }
    if name == "enter_option":
        return {
            "symbol": sym,
            "strategy": "long_call",
            "quantity": 1,
            "dte_target": 30,
            "max_spread_pct": 25.0,
        }
    if name == "instrument_selector":
        return {"symbol": sym, "outlook": "bullish"}
    if name == "option_chain":
        return {"symbol": sym, "dte_min": 14, "dte_max": 120, "limit": 10}
    if name == "position_greeks":
        return {"symbol": sym}
    if name == "option_quote":
        src = ctx.sample_call or ctx.sample_put
        if not src or not src.get("symbol"):
            return None
        return {"option_symbol": src["symbol"]}
    if name == "option_greeks":
        src = ctx.sample_call or ctx.sample_put
        if not src:
            return None
        exp = _norm_exp(src.get("expiration"))
        if not exp:
            return None
        right = "C" if (src.get("side") or "").lower() == "call" else "P"
        return {"symbol": sym, "expiration": exp, "strike": src["strike"], "right": right}
    if name == "update_working_memory":
        return {"section": "lessons_today", "entry": "(tool smoke) ping — safe to delete."}
    return None


def _broker_soft_error(err: str | None) -> bool:
    if not err:
        return False
    e = err.lower()
    needles = (
        "no position",
        "no open position",
        "not found",
        "cash-only",
        "outside the allowed trading universe",
        "insufficient",
        "duplicate",
        "risk guard",
        "duplicate blocked",
        "must provide",
        "required:",
        "invalid",
        "cannot",
        "no quote",
        "no option",
        "no contracts",
        "market closed",
        "wait",
        "unknown multi-leg",
        "unknown action",
        "order_id",
        "no long position",
        "pdt",
    )
    return any(n in e for n in needles)


def _pick_otm_call_vertical(ctx: SmokeContext) -> tuple[str, float, float] | None:
    """Return (expiration_yyyymmdd, long_strike, short_strike) for bull call or None."""
    c = ctx.sample_call
    if not c:
        return None
    exp = _norm_exp(c.get("expiration"))
    if not exp or ctx.last is None or float(ctx.last) <= 0:
        return None
    price = float(ctx.last)
    # Prefer OTM call pair: long lower strike, short higher, both >= spot (bull call OTM)
    lo = float(c.get("strike") or 0)
    if lo <= 0:
        return None
    width = max(1.0, round(price * 0.02, 0))  # ~2% width, min $1
    long_s = round(lo, 2)
    short_s = round(lo + width, 2)
    if short_s <= long_s:
        short_s = long_s + 1.0
    if long_s < price * 0.98:  # nudge OTM-ish for calls
        long_s = round(max(long_s, price * 1.01), 2)
        short_s = round(long_s + width, 2)
    return exp, long_s, short_s


def params_for_broker_tool(
    name: str,
    ctx: SmokeContext,
    ref: dict[str, Any],
    *,
    allow_market: bool,
    allow_option_orders: bool,
) -> dict | None:
    """Return params for a broker-mutating tool, or None to skip."""
    sym = ctx.symbol
    last = ctx.last
    if last is None or last <= 0:
        return None

    far_limit = max(0.01, round(float(last) * 0.5, 2))
    near_cap = round(float(last) * 1.02, 2)

    if name in NEVER_AUTOTEST:
        return None

    if name == "cancel_order":
        # Handled explicitly after the broker loop (uses place_oid).
        return None

    if name == "modify_stop":
        oid = ref.get("stop_oid")
        if oid is None:
            return None
        return {"order_id": oid, "new_stop_price": round(float(last) * 0.45, 2)}

    if name == "cancel_stops":
        return {"symbol": sym}

    if name == "limit_order":
        if ref.get("place_oid") is not None:
            return None
        return {"symbol": sym, "side": "BUY", "quantity": 1, "limit_price": far_limit}

    if name == "market_order":
        if not allow_market:
            return None
        return {"symbol": sym, "side": "BUY", "quantity": 1}

    if name in ("adaptive_order",):
        return {
            "symbol": sym,
            "side": "BUY",
            "quantity": 1,
            "order_type": "LMT",
            "limit_price": far_limit,
            "priority": "Patient",
        }

    if name in ("midprice_order", "snap_mid_order"):
        return {"symbol": sym, "side": "BUY", "quantity": 1, "price_cap": near_cap}

    if name == "relative_order":
        return {
            "symbol": sym,
            "side": "BUY",
            "quantity": 1,
            "offset": 0.05,
            "limit_price": far_limit,
        }

    if name == "stop_order":
        return {
            "symbol": sym,
            "side": "SELL",
            "quantity": 1,
            "stop_price": far_limit,
        }

    if name == "stop_limit":
        sp = round(float(last) * 0.48, 2)
        lp = round(float(last) * 0.47, 2)
        return {"symbol": sym, "side": "SELL", "quantity": 1, "stop_price": sp, "limit_price": lp}

    if name == "trailing_stop":
        return {"symbol": sym, "quantity": 1, "direction": "LONG", "trail_percent": 0.05}

    if name == "trailing_stop_limit":
        return {
            "symbol": sym,
            "quantity": 1,
            "direction": "LONG",
            "trail_percent": 0.05,
            "limit_offset": 0.1,
        }

    if name == "bracket_order":
        return {
            "symbol": sym,
            "side": "BUY",
            "quantity": 1,
            "limit_price": far_limit,
            "stop_loss": round(float(last) * 0.4, 2),
            "take_profit": near_cap,
        }

    if name == "oca_order":
        return {
            "symbol": sym,
            "quantity": 1,
            "direction": "LONG",
            "stop_price": round(float(last) * 0.45, 2),
            "target_price": near_cap,
        }

    if name in ("moc_order", "moo_order"):
        return {"symbol": sym, "side": "BUY", "quantity": 1}

    if name in ("loc_order", "loo_order"):
        return {"symbol": sym, "side": "BUY", "quantity": 1, "limit_price": far_limit}

    if name == "gtd_order":
        return {
            "symbol": sym,
            "side": "BUY",
            "quantity": 1,
            "limit_price": far_limit,
            "good_till_date": "20261231 16:00:00",
        }

    if name in ("fok_order", "ioc_order"):
        return {"symbol": sym, "side": "BUY", "quantity": 1, "limit_price": far_limit}

    if name in ("vwap_order", "twap_order"):
        return {"symbol": sym, "side": "BUY", "quantity": 1}

    if name == "iceberg_order":
        return {
            "symbol": sym,
            "side": "BUY",
            "total_quantity": 3,
            "display_size": 1,
            "limit_price": far_limit,
        }

    if name == "multi_leg":
        picked = _pick_otm_call_vertical(ctx)
        if not picked:
            return None
        exp, ls, ss = picked
        return {
            "type": "debit_spread",
            "symbol": sym,
            "expiration": exp,
            "long_strike": min(ls, ss),
            "short_strike": max(ls, ss),
            "right": "C",
            "quantity": 1,
        }

    if not allow_option_orders:
        return None

    if name == "buy_option":
        src = ctx.sample_call
        if not src:
            return None
        exp = _norm_exp(src.get("expiration"))
        if not exp:
            return None
        return {
            "symbol": sym,
            "expiration": exp,
            "strike": float(src["strike"]),
            "right": "C",
            "quantity": 1,
        }

    if name == "vertical_spread":
        picked = _pick_otm_call_vertical(ctx)
        if not picked:
            return None
        exp, ls, ss = picked
        lo, hi = min(ls, ss), max(ls, ss)
        return {
            "symbol": sym,
            "expiration": exp,
            "long_strike": lo,
            "short_strike": hi,
            "right": "C",
            "quantity": 1,
            "limit_price": round(float(last) * 0.05, 2),
        }

    # Income / complex structures — often blocked without stock or wide cash; still exercise handler
    def _exp() -> str | None:
        if not ctx.sample_call:
            return None
        e = _norm_exp(ctx.sample_call.get("expiration"))
        return e or None

    exp0 = _exp()
    if name == "iron_condor":
        if not exp0:
            return None
        return {
            "symbol": sym,
            "expiration": exp0,
            "put_long_strike": round(float(last) * 0.85, 2),
            "put_short_strike": round(float(last) * 0.88, 2),
            "call_short_strike": round(float(last) * 1.12, 2),
            "call_long_strike": round(float(last) * 1.15, 2),
            "quantity": 1,
        }

    if name == "iron_butterfly":
        if not exp0:
            return None
        mid = round(float(last), 2)
        w = max(1.0, round(float(last) * 0.02, 2))
        return {
            "symbol": sym,
            "expiration": exp0,
            "center_strike": mid,
            "wing_width": w,
            "quantity": 1,
        }

    if name == "straddle":
        if not exp0:
            return None
        return {
            "symbol": sym,
            "expiration": exp0,
            "strike": round(float(last), 2),
            "quantity": 1,
        }

    if name == "strangle":
        if not exp0:
            return None
        return {
            "symbol": sym,
            "expiration": exp0,
            "put_strike": round(float(last) * 0.95, 2),
            "call_strike": round(float(last) * 1.05, 2),
            "quantity": 1,
        }

    if name == "calendar_spread":
        if not ctx.sample_call:
            return None
        exp = _norm_exp(ctx.sample_call.get("expiration"))
        if not exp:
            return None
        return {
            "symbol": sym,
            "strike": float(ctx.sample_call["strike"]),
            "near_expiration": exp,
            "far_expiration": exp,
            "right": "C",
            "quantity": 1,
        }

    if name == "diagonal_spread":
        if not ctx.sample_call:
            return None
        exp = _norm_exp(ctx.sample_call.get("expiration"))
        if not exp:
            return None
        return {
            "symbol": sym,
            "near_strike": float(ctx.sample_call["strike"]),
            "far_strike": float(ctx.sample_call["strike"]) + 2,
            "near_expiration": exp,
            "far_expiration": exp,
            "right": "C",
            "quantity": 1,
        }

    if name == "butterfly":
        if not exp0:
            return None
        mid = round(float(last), 2)
        w = max(1.0, round(float(last) * 0.015, 2))
        return {
            "symbol": sym,
            "expiration": exp0,
            "lower_strike": mid - w,
            "middle_strike": mid,
            "upper_strike": mid + w,
            "right": "C",
            "quantity": 1,
        }

    if name == "ratio_spread":
        if not exp0:
            return None
        return {
            "symbol": sym,
            "expiration": exp0,
            "long_strike": round(float(last) * 1.02, 2),
            "short_strike": round(float(last) * 1.05, 2),
            "right": "C",
            "ratio": [1, 2],
            "quantity": 1,
        }

    if name == "jade_lizard":
        if not exp0:
            return None
        return {
            "symbol": sym,
            "expiration": exp0,
            "put_strike": round(float(last) * 0.9, 2),
            "call_short_strike": round(float(last) * 1.05, 2),
            "call_long_strike": round(float(last) * 1.1, 2),
            "quantity": 1,
        }

    if name == "collar":
        if not exp0:
            return None
        return {
            "symbol": sym,
            "expiration": exp0,
            "put_strike": round(float(last) * 0.92, 2),
            "call_strike": round(float(last) * 1.08, 2),
            "shares": 100,
        }

    if name == "covered_call":
        if not exp0:
            return None
        return {
            "symbol": sym,
            "expiration": exp0,
            "strike": round(float(last) * 1.1, 2),
            "shares": 100,
        }

    if name == "cash_secured_put":
        if not ctx.sample_put:
            return None
        exp_p = _norm_exp(ctx.sample_put.get("expiration"))
        if not exp_p:
            return None
        return {
            "symbol": sym,
            "expiration": exp_p,
            "strike": round(float(last) * 0.85, 2),
            "contracts": 1,
        }

    if name == "protective_put":
        if not ctx.sample_put:
            return None
        exp_p = _norm_exp(ctx.sample_put.get("expiration"))
        if not exp_p:
            return None
        return {
            "symbol": sym,
            "expiration": exp_p,
            "strike": round(float(last) * 0.9, 2),
            "shares": 100,
        }

    if name == "close_option":
        if not ctx.sample_call:
            return None
        return {
            "symbol": sym,
            "expiration": _norm_exp(ctx.sample_call.get("expiration")),
            "strike": float(ctx.sample_call["strike"]),
            "right": "C",
            "limit_price": far_limit,
        }

    if name == "close_spread":
        return {"symbol": sym}

    if name == "roll_option":
        if not ctx.sample_call:
            return None
        exp = _norm_exp(ctx.sample_call.get("expiration"))
        return {
            "symbol": sym,
            "old_expiration": exp,
            "old_strike": float(ctx.sample_call["strike"]),
            "new_expiration": exp,
            "new_strike": float(ctx.sample_call["strike"]) + 1,
            "right": "C",
            "quantity": 1,
        }

    return None


async def run_safe_phase(
    executor: ToolExecutor,
    ctx: SmokeContext,
    *,
    with_research: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []

    for action in sorted(safe_names()):
        if action == "clear_working_memory_entry":
            continue
        if action == "research" and not with_research:
            skipped.append((action, "omit --with-research"))
            continue
        params = params_for_safe_tool(action, ctx)
        if params is None:
            skipped.append((action, "no sample contract / params"))
            continue
        try:
            tr = await executor.execute(action, params)
            payload = tr.data if isinstance(tr.data, dict) else {}
            if action == "update_working_memory" and payload.get("success") and payload.get("entry_id"):
                ctx.wm_clear.append((payload.get("section", "lessons_today"), int(payload["entry_id"])))
            soft_ok = action in (
                "get_position",
                "signal_breakdown",
                "enter_option",
                "plan_order",
                "option_quote",
                "option_greeks",
            )
            if not tr.success and not soft_ok:
                err = payload.get("error")
                if err:
                    failures.append(f"{action}: {err}")
            if not tr.success and soft_ok:
                logger.info("(soft) %s success=%s err=%s", action, tr.success, payload.get("error"))
        except Exception as e:
            failures.append(f"{action}: EXC {e}")

    if ctx.wm_clear:
        entry = ctx.wm_clear[-1]
        trc = await executor.execute(
            "clear_working_memory_entry",
            {"section": entry[0], "entry_id": entry[1]},
        )
        if not trc.success:
            logger.warning("clear_working_memory_entry cleanup: %s", trc.data)

    return {"failures": failures, "skipped": skipped}


async def run_broker_phase(
    executor: ToolExecutor,
    ctx: SmokeContext,
    *,
    allow_market: bool,
    allow_option_orders: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    broker_soft: list[str] = []
    skipped: list[tuple[str, str]] = []
    ref: dict[str, Any] = {"place_oid": None, "stop_oid": None}

    raw = sorted(n for n in broker_mutating_names() if n not in NEVER_AUTOTEST and n != "cancel_order")

    def _sort_key(n: str) -> tuple[int, str]:
        if n == "limit_order":
            return (0, n)
        return (1, n)

    names = sorted(raw, key=_sort_key)
    for action in names:
        params = params_for_broker_tool(
            action, ctx, ref, allow_market=allow_market, allow_option_orders=allow_option_orders
        )
        if params is None:
            skipped.append((action, "no safe params / skip rule"))
            continue
        try:
            tr = await executor.execute(action, params)
            payload = tr.data if isinstance(tr.data, dict) else {}
            err = payload.get("error") if isinstance(payload, dict) else None
            if action == "limit_order" and tr.success:
                oid = payload.get("order_id") if isinstance(payload, dict) else None
                if oid is not None:
                    ref["place_oid"] = int(oid)
            if not tr.success:
                if _broker_soft_error(str(err) if err else ""):
                    broker_soft.append(f"{action}: {err}")
                else:
                    if err:
                        failures.append(f"{action}: {err}")
                    else:
                        failures.append(f"{action}: failed (no error text)")
            else:
                if action == "stop_order" and isinstance(payload, dict) and payload.get("order_id"):
                    ref["stop_oid"] = payload.get("order_id")
        except Exception as e:
            failures.append(f"{action}: EXC {e}")

    # Cancel working smoke limit if still open (also exercises cancel_order)
    oid = ref.get("place_oid")
    if oid is not None:
        cx = await executor.execute("cancel_order", {"order_id": oid})
        if not cx.success:
            err = (cx.data or {}).get("error") if isinstance(cx.data, dict) else None
            if err and not _broker_soft_error(str(err)):
                failures.append(f"cancel_order(cleanup): {err}")

    return {"failures": failures, "broker_soft": broker_soft, "skipped": skipped}
