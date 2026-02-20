"""Research and market data tool handlers."""

import datetime
import logging
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)


# VIX-like symbols that need special handling
_VIX_ALIASES = {"VIX", "^VIX", "$VIX", "VIX9D"}


async def handle_quote(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        # VIX fallback: MarketData.app doesn't support VIX directly.
        # Try UVXY as a VIX proxy, or estimate from SPY options IV.
        if symbol.upper().replace("^", "").replace("$", "") in {"VIX", "VIX9D"}:
            return await _vix_fallback(executor)

        quote = executor.data_provider.get_quote(symbol)
        if quote:
            result = asdict(quote)
            # Tag data source for transparency
            src = quote.source or ''
            result["source"] = src
            return result
        return {"error": f"No quote for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def _vix_fallback(executor) -> dict:
    """Estimate VIX from SPY option IV or UVXY price as proxy."""
    # Try UVXY as VIX proxy
    try:
        uvxy = executor.data_provider.get_quote("UVXY")
        if uvxy:
            result = asdict(uvxy)
            result["note"] = "UVXY used as VIX proxy (VIX not directly available)"
            result["symbol"] = "VIX (via UVXY)"
            return result
    except Exception:
        pass

    # Fallback: estimate from SPY ATM options IV
    try:
        from data.marketdata_client import get_marketdata_client
        client = get_marketdata_client()
        iv_data = await client.get_iv_rank("SPY", dte_min=20, dte_max=40)
        if iv_data and iv_data.get("iv_current"):
            return {
                "symbol": "VIX (estimated from SPY IV)",
                "last": round(iv_data["iv_current"], 1),
                "note": "Estimated from SPY ATM call IV. Not true VIX.",
                "source": "spy_iv_estimate",
            }
    except Exception:
        pass

    # Final fallback: estimate volatility from SPY ATR
    try:
        atr = executor.data_provider.get_atr("SPY", 14)
        quote = executor.data_provider.get_quote("SPY")
        if atr and quote and quote.last and quote.last > 0:
            # Annualized ATR as % ≈ daily ATR% × √252 — rough VIX proxy
            atr_pct = atr.value / quote.last
            annualized = round(atr_pct * (252 ** 0.5) * 100, 1)
            return {
                "symbol": "VIX (estimated from SPY ATR)",
                "last": annualized,
                "atr_daily": round(atr.value, 2),
                "spy_price": round(quote.last, 2),
                "note": "Annualized from SPY 14-day ATR. Rough proxy only.",
                "source": "spy_atr_estimate",
            }
    except Exception:
        pass

    return {"error": "VIX unavailable. Use UVXY quote or SPY option IV as proxy."}


async def handle_candles(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        days = int(params.get("days", 30))
        resolution = str(params.get("resolution", "D")).upper().strip()
        # Normalize common aliases
        _res_aliases = {
            "DAILY": "D", "DAY": "D", "1D": "D",
            "HOURLY": "H", "HOUR": "H", "60": "H", "1H": "H",
            "5MIN": "5", "5M": "5",
            "15MIN": "15", "15M": "15",
            "1MIN": "1", "1M": "1",
            "WEEKLY": "W", "WEEK": "W", "1W": "W",
            "MONTHLY": "M", "MONTH": "M", "1MO": "M",
        }
        resolution = _res_aliases.get(resolution, resolution)
        valid_resolutions = {"D": "daily", "H": "hourly", "5": "5min", "15": "15min", "1": "1min", "W": "weekly", "M": "monthly"}
        if resolution not in valid_resolutions:
            return {"error": f"Invalid resolution '{resolution}'. Use: D (daily), H (hourly), 5 (5min), 15 (15min), 1 (1min), W (weekly), M (monthly)"}
        
        candles = executor.data_provider.get_candles(symbol, resolution=resolution, days_back=days)
        if candles and len(candles) > 0:
            candle_list = []
            n = len(candles)
            for i in range(n):
                ts = candles.timestamps[i]
                if resolution == 'D':
                    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else ''
                else:
                    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else ''
                candle_list.append({
                    "date": dt,
                    "open": round(candles.open[i], 2),
                    "high": round(candles.high[i], 2),
                    "low": round(candles.low[i], 2),
                    "close": round(candles.close[i], 2),
                    "volume": candles.volume[i]
                })
            return {
                "symbol": symbol,
                "resolution": valid_resolutions[resolution],
                "bars": n,
                "candles": candle_list
            }
        return {"error": f"No candle data for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_fundamentals(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_fundamentals(symbol)
        if result:
            return asdict(result)
        return {"error": f"No fundamentals for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_earnings(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_earnings_info(symbol)
        if result:
            return asdict(result)
        return {"error": f"No earnings data for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_atr(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        period = int(params.get("period", 14))
        result = executor.data_provider.get_atr(symbol, period)
        if result:
            data = asdict(result)
            quote = executor.data_provider.get_quote(symbol)
            if quote and quote.last and quote.last > 0:
                data["atr_pct"] = round(result.value / quote.last * 100, 2)
            return data
        return {"error": f"No ATR data for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_iv_info(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}

    # Agent-controlled IV sampling parameters — no defaults
    dte_min = params.get("dte_min")
    dte_max = params.get("dte_max")
    if dte_min is None or dte_max is None:
        return {
            "error": "dte_min and dte_max are required",
            "available_params": {
                "symbol": "required - underlying ticker",
                "dte_min": "required - minimum days to expiration",
                "dte_max": "required - maximum days to expiration",
                "strike_pct": "optional - strike range as decimal (0.15 = ±15%). Auto-adaptive if omitted."
            }
        }
    dte_min = int(dte_min)
    dte_max = int(dte_max)
    strike_pct = params.get("strike_pct")  # e.g. 0.15 = ±15%; None = auto
    if strike_pct is not None:
        strike_pct = float(strike_pct)

    try:
        result = executor.data_provider.get_iv_info(
            symbol, dte_min=dte_min, dte_max=dte_max, strike_pct=strike_pct
        )
        if result:
            data = asdict(result)
            data["params_used"] = {
                "dte_min": dte_min, "dte_max": dte_max,
                "strike_pct": strike_pct or "auto"
            }
            return data
        return {
            "error": f"No IV data for {symbol}",
            "params_used": {
                "dte_min": dte_min, "dte_max": dte_max,
                "strike_pct": strike_pct or "auto"
            },
            "available_params": {
                "symbol": "required - underlying ticker",
                "dte_min": "optional - min DTE for IV sampling (default 14)",
                "dte_max": "optional - max DTE for IV sampling (default 45)",
                "strike_pct": "optional - strike range as decimal (0.15 = ±15%). Auto if omitted."
            }
        }
    except Exception as e:
        return {"error": str(e)}


async def handle_news(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_news(symbol)
        if result:
            return {
                "symbol": result.symbol,
                "sentiment": result.sentiment,
                "headlines": [
                    {"title": item.title, "publisher": item.publisher}
                    for item in result.items[:10]
                ]
            }
        return {"error": f"No news for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_analysts(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_analyst_data(symbol)
        if result:
            return asdict(result)
        return {"error": f"No analyst data for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_extended_fundamentals(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_extended_fundamentals(symbol)
        if result:
            return asdict(result)
        return {"error": f"No extended fundamentals for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_institutional_data(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_institutional_data(symbol)
        if result:
            data = {
                "symbol": result.symbol,
                "insider_pct": result.insider_pct,
                "institutional_pct": result.institutional_pct,
                "top_holders": [
                    {"holder": h.holder, "shares": h.shares, "pct": h.pct_held}
                    for h in (result.top_holders or [])
                ],
            }
            return data
        return {"error": f"No institutional data for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_insider_data(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_insider_data(symbol)
        if result:
            data = {
                "symbol": result.symbol,
                "recent_buys": result.recent_buys,
                "recent_sells": result.recent_sells,
                "net_sentiment": result.net_sentiment,
                "total_buy_value": result.total_buy_value,
                "total_sell_value": result.total_sell_value,
                "transactions": [
                    {
                        "insider": t.insider,
                        "relation": t.relation,
                        "type": t.transaction_type,
                        "shares": t.shares,
                        "value": t.value,
                    }
                    for t in (result.transactions or [])[:5]
                ],
            }
            return data
        return {"error": f"No insider data for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_peer_comparison(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    try:
        result = executor.data_provider.get_peer_comparison(symbol)
        if result:
            return asdict(result)
        return {"error": f"No peer comparison for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_market_hours(executor, params: dict) -> Any:
    from data.market_hours import get_market_hours_provider
    provider = get_market_hours_provider()
    return provider.get_session_info()


async def handle_budget(executor, params: dict) -> Any:
    from data.cost_tracker import get_cost_tracker
    tracker = get_cost_tracker()
    summary = tracker.get_budget_summary()
    return {"budget": summary.to_llm_string()}


async def handle_economic_calendar(executor, params: dict) -> Any:
    """Return today's macro events + 3-day look-ahead."""
    from data.economic_calendar import get_todays_events, get_upcoming_events
    today_events = get_todays_events()
    upcoming = get_upcoming_events(days=3)
    return {
        "today": [e.to_dict() for e in today_events],
        "upcoming_3d": [e.to_dict() for e in upcoming],
        "count_today": len(today_events),
        "count_upcoming": len(upcoming),
    }


HANDLERS = {
    "quote": handle_quote,
    "candles": handle_candles,
    "fundamentals": handle_fundamentals,
    "earnings": handle_earnings,
    "atr": handle_atr,
    "iv_info": handle_iv_info,
    "news": handle_news,
    "analysts": handle_analysts,
    "extended_fundamentals": handle_extended_fundamentals,
    "institutional_data": handle_institutional_data,
    "insider_data": handle_insider_data,
    "peer_comparison": handle_peer_comparison,
    "market_hours": handle_market_hours,
    "budget": handle_budget,
    "economic_calendar": handle_economic_calendar,
}
