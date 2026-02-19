"""Instrument selection tool — helps the agent discover ALL available
entry methods for a given symbol and directional outlook.

This replaces the need to list every tool in the system prompt.
The agent calls this to see what's available, then picks.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# All instrument categories and their tools — single source of truth
_INSTRUMENTS = {
    "stock_entry": {
        "label": "Stock Entry",
        "tools": {
            "plan_order": {
                "desc": "Auto-selects optimal order type + stop based on spread, ATR, session. RECOMMENDED for most entries.",
                "params": "symbol, side (BUY=long entry, SELL=short entry), quantity, execute=true",
                "directions": ["bullish", "bearish"],
            },
            "market_order": {
                "desc": "Immediate fill, no price control",
                "params": "symbol, side, quantity",
                "directions": ["bullish", "bearish"],
            },
            "limit_order": {
                "desc": "Fill at limit price or better",
                "params": "symbol, side, quantity, limit_price",
                "directions": ["bullish", "bearish"],
            },
            "bracket_order": {
                "desc": "Entry + stop loss + take profit in one order",
                "params": "symbol, side, quantity, limit_price, stop_loss, take_profit",
                "directions": ["bullish", "bearish"],
            },
            "adaptive_order": {
                "desc": "IBKR adaptive algo — optimizes fill quality",
                "params": "symbol, side, quantity, priority (Patient/Normal/Urgent)",
                "directions": ["bullish", "bearish"],
            },
            "midprice_order": {
                "desc": "Pegged to bid/ask midpoint — good for wide spreads",
                "params": "symbol, side, quantity, price_cap?",
                "directions": ["bullish", "bearish"],
            },
        },
    },
    "stock_algos": {
        "label": "Algorithmic Stock Entry (large orders)",
        "tools": {
            "vwap_order": {
                "desc": "Volume-weighted average price execution",
                "params": "symbol, side, quantity, start_time?, end_time?",
                "directions": ["bullish", "bearish"],
            },
            "twap_order": {
                "desc": "Time-weighted average price execution",
                "params": "symbol, side, quantity, start_time?, end_time?",
                "directions": ["bullish", "bearish"],
            },
            "iceberg_order": {
                "desc": "Shows only partial size to market",
                "params": "symbol, side, total_quantity, display_size, limit_price",
                "directions": ["bullish", "bearish"],
            },
        },
    },
    "stock_timed": {
        "label": "Timed Stock Orders",
        "tools": {
            "moo_order": {
                "desc": "Market-on-Open — fills at opening auction",
                "params": "symbol, side, quantity",
                "directions": ["bullish", "bearish"],
            },
            "moc_order": {
                "desc": "Market-on-Close — fills at closing auction",
                "params": "symbol, side, quantity",
                "directions": ["bullish", "bearish"],
            },
            "loo_order": {
                "desc": "Limit-on-Open",
                "params": "symbol, side, quantity, limit_price",
                "directions": ["bullish", "bearish"],
            },
            "loc_order": {
                "desc": "Limit-on-Close",
                "params": "symbol, side, quantity, limit_price",
                "directions": ["bullish", "bearish"],
            },
            "gtd_order": {
                "desc": "Good-Till-Date — auto-expires",
                "params": "symbol, side, quantity, limit_price, good_till_date",
                "directions": ["bullish", "bearish"],
            },
        },
    },
    "options_directional": {
        "label": "Options — Directional",
        "tools": {
            "enter_option(long_call)": {
                "desc": "Buy call — bullish, limited risk to premium",
                "params": 'symbol, strategy="long_call", quantity, execute=true',
                "directions": ["bullish"],
            },
            "enter_option(long_put)": {
                "desc": "Buy put — bearish, limited risk to premium",
                "params": 'symbol, strategy="long_put", quantity, execute=true',
                "directions": ["bearish"],
            },
            "enter_option(bull_call_spread)": {
                "desc": "Vertical spread — bullish, defined risk & reward",
                "params": 'symbol, strategy="bull_call_spread", quantity, execute=true',
                "directions": ["bullish"],
            },
            "enter_option(bear_put_spread)": {
                "desc": "Vertical spread — bearish, defined risk & reward",
                "params": 'symbol, strategy="bear_put_spread", quantity, execute=true',
                "directions": ["bearish"],
            },
        },
    },
    "options_income": {
        "label": "Options — Income / Neutral",
        "tools": {
            "enter_option(covered_call)": {
                "desc": "Sell call against shares — income on existing long position",
                "params": 'symbol, strategy="covered_call", quantity, execute=true',
                "directions": ["neutral", "mildly_bullish"],
            },
            "enter_option(cash_secured_put)": {
                "desc": "Sell put for premium — income if stock stays above strike",
                "params": 'symbol, strategy="cash_secured_put", quantity, execute=true',
                "directions": ["neutral", "mildly_bullish"],
            },
            "enter_option(iron_condor)": {
                "desc": "Sell put+call spreads — income in range-bound markets, defined risk",
                "params": 'symbol, strategy="iron_condor", quantity, execute=true',
                "directions": ["neutral"],
            },
        },
    },
    "options_volatility": {
        "label": "Options — Volatility / Big Move Expected",
        "tools": {
            "enter_option(straddle)": {
                "desc": "Buy ATM call + put — profits from big move in either direction",
                "params": 'symbol, strategy="straddle", quantity, execute=true',
                "directions": ["volatile"],
            },
            "enter_option(strangle)": {
                "desc": "Buy OTM call + put — cheaper than straddle, needs bigger move",
                "params": 'symbol, strategy="strangle", quantity, execute=true',
                "directions": ["volatile"],
            },
        },
    },
    "options_protection": {
        "label": "Options — Portfolio Protection",
        "tools": {
            "enter_option(protective_put)": {
                "desc": "Buy put to protect existing long position",
                "params": 'symbol, strategy="protective_put", quantity, execute=true',
                "directions": ["protection"],
            },
            "collar": {
                "desc": "Buy protective put + sell covered call — low/zero cost hedge",
                "params": "symbol, expiration, put_strike, call_strike, shares",
                "directions": ["protection"],
            },
        },
    },
    "options_advanced_spreads": {
        "label": "Options — Advanced Spreads (direct execution)",
        "tools": {
            "vertical_spread": {
                "desc": "Bull/bear spread with explicit strikes",
                "params": "symbol, expiration, long_strike, short_strike, right (C/P), quantity",
                "directions": ["bullish", "bearish"],
            },
            "iron_butterfly": {
                "desc": "ATM butterfly — max premium collection, narrow profit zone",
                "params": "symbol, expiration, center_strike, wing_width, quantity",
                "directions": ["neutral"],
            },
            "calendar_spread": {
                "desc": "Same strike, different expirations — profits from time decay",
                "params": "symbol, strike, near_expiration, far_expiration, right?, quantity",
                "directions": ["neutral", "mildly_bullish", "mildly_bearish"],
            },
            "diagonal_spread": {
                "desc": "Different strikes + expirations — directional + time decay",
                "params": "symbol, near_strike, far_strike, near_exp, far_exp, right?, quantity",
                "directions": ["bullish", "bearish"],
            },
            "butterfly": {
                "desc": "3-strike butterfly — low cost, profits if stock pins at middle strike",
                "params": "symbol, expiration, lower_strike, middle_strike, upper_strike, right?, quantity",
                "directions": ["neutral"],
            },
            "ratio_spread": {
                "desc": "Unequal legs — cheap/free entry, unlimited risk on one side",
                "params": "symbol, expiration, long_strike, short_strike, right?, ratio?, quantity",
                "directions": ["bullish", "bearish"],
            },
            "jade_lizard": {
                "desc": "Short put + short call spread — no upside risk, premium collection",
                "params": "symbol, expiration, put_strike, call_short_strike, call_long_strike, quantity",
                "directions": ["neutral", "mildly_bullish"],
            },
        },
    },
}

# Direction → relevant outlook keywords
_DIRECTION_MAP = {
    "bullish": ["bullish", "mildly_bullish", "volatile", "protection"],
    "bearish": ["bearish", "mildly_bearish", "volatile", "protection"],
    "neutral": ["neutral", "mildly_bullish", "mildly_bearish", "volatile"],
    "volatile": ["volatile", "bullish", "bearish"],
}


async def handle_instrument_selector(executor, params: dict) -> Any:
    """Return all available instruments, optionally filtered by outlook.
    
    Args:
        symbol: Stock ticker (optional — used to add context like IV, price)
        outlook: bullish | bearish | neutral | volatile (optional — highlights relevant instruments)
    """
    symbol = params.get("symbol")
    outlook = (params.get("outlook") or "").lower().strip()
    
    # Gather optional context for the symbol
    context = {}
    if symbol and executor.data_provider:
        symbol = symbol.upper()
        try:
            q = executor.data_provider.get_quote(symbol)
            if q and q.last:
                context["price"] = q.last
                context["change_pct"] = q.change_pct
        except Exception:
            pass
        try:
            iv_dte_min = params.get("iv_dte_min")
            iv_dte_max = params.get("iv_dte_max")
            iv_strike_pct = params.get("iv_strike_pct")
            if iv_strike_pct is not None:
                iv_strike_pct = float(iv_strike_pct)
            if iv_dte_min is not None and iv_dte_max is not None:
                iv = executor.data_provider.get_iv_info(
                    symbol, dte_min=int(iv_dte_min), dte_max=int(iv_dte_max),
                    strike_pct=iv_strike_pct
                )
                if iv:
                    context["iv_current"] = iv.iv_current
                    context["iv_rank"] = iv.iv_rank
        except Exception:
            pass
        try:
            atr = executor.data_provider.get_atr(symbol)
            if atr:
                context["atr_pct"] = round(atr.value / context.get("price", 1) * 100, 1) if context.get("price") else None
        except Exception:
            pass

    # Get regime for context (simplified — no LiveState regime tracker)
    regime_label = None

    # Build output
    result = {
        "symbol": symbol,
        "outlook": outlook or "all (specify outlook to highlight relevant instruments)",
        "market_regime": regime_label,
    }
    
    if context:
        result["symbol_context"] = context
    
    # Determine which directions to highlight
    relevant_dirs = set()
    if outlook and outlook in _DIRECTION_MAP:
        relevant_dirs = set(_DIRECTION_MAP[outlook])
    
    categories = []
    for cat_key, cat_data in _INSTRUMENTS.items():
        cat_out = {
            "category": cat_data["label"],
            "instruments": [],
        }
        for tool_name, tool_info in cat_data["tools"].items():
            tool_dirs = set(tool_info["directions"])
            is_relevant = bool(relevant_dirs & tool_dirs) if relevant_dirs else True

            # CASH-ONLY: hide instruments that create short stock positions
            if executor.cash_only:
                # Block plan_order/market_order/limit_order etc. with bearish direction
                # (SELL side = short entry) in stock categories
                if cat_key in ("stock_entry", "stock_algos", "stock_timed"):
                    if "bearish" in tool_dirs:
                        # Only show if also has bullish direction AND outlook is bullish/neutral
                        if outlook in ("bearish",):
                            continue  # Skip entirely for bearish outlook on stock tools
                        # For dual-direction tools, note they are BUY-only
                        entry = {
                            "tool": tool_name,
                            "description": tool_info["desc"] + " [CASH-ONLY: BUY side only]",
                            "params": tool_info["params"],
                        }
                        if relevant_dirs:
                            entry["matches_outlook"] = is_relevant
                        cat_out["instruments"].append(entry)
                        continue
            
            entry = {
                "tool": tool_name,
                "description": tool_info["desc"],
                "params": tool_info["params"],
            }
            if relevant_dirs:
                entry["matches_outlook"] = is_relevant
            
            cat_out["instruments"].append(entry)
        
        # If filtering by outlook, only include categories with at least 1 relevant instrument
        if relevant_dirs:
            has_relevant = any(i.get("matches_outlook") for i in cat_out["instruments"])
            if has_relevant:
                categories.append(cat_out)
        else:
            categories.append(cat_out)
    
    result["categories"] = categories
    
    # Add smart suggestions based on context
    suggestions = []
    if context.get("iv_rank") and context["iv_rank"] > 50:
        suggestions.append("High IV rank — selling premium (iron condor, covered call, CSP) may be favorable")
    if context.get("iv_rank") and context["iv_rank"] < 20:
        suggestions.append("Low IV rank — buying options (long call/put, straddle) is relatively cheap")
    if context.get("atr_pct") and context["atr_pct"] > 3:
        suggestions.append(f"High ATR ({context['atr_pct']}%) — consider wider stops or defined-risk spreads")
    if regime_label in ("LIQUIDATION", "RISK-OFF", "SELLING"):
        suggestions.append(f"Regime is {regime_label} — bearish instruments, puts, and hedges may be appropriate")
    if regime_label in ("RISK-ON",):
        suggestions.append(f"Regime is {regime_label} — bullish entries and momentum plays favorable")
    
    if suggestions:
        result["suggestions"] = suggestions

    # Cash-only advisory
    if executor.cash_only:
        result["cash_only_notice"] = (
            "CASH-ONLY MODE: Short selling is disabled. "
            "For bearish views, use long puts, bear put spreads, or protective puts. "
            "Stock entries must be BUY side only."
        )

    result["note"] = "Use plan_order for stock entries (auto-selects order type + stop). Use enter_option for options (auto-selects contract). Use direct tools (vertical_spread, etc.) when you want explicit control."
    
    return result


HANDLERS = {
    "instrument_selector": handle_instrument_selector,
}
