"""Research and market data tool handlers."""

import datetime
from dataclasses import asdict
from datetime import date, timezone
from datetime import datetime as dt_datetime
from typing import Any

from core.log_context import get_logger
from core.runtime.operating_context import get_operating_context

logger = get_logger(__name__)


# VIX-like symbols that need special handling
_VIX_ALIASES = {"VIX", "^VIX", "$VIX", "VIX9D"}


def _normalize_resolution(value: Any) -> str:
    """Normalize resolution aliases to canonical code."""
    raw = str(value or "D").strip().upper()
    aliases = {
        "DAILY": "D", "DAY": "D", "1D": "D", "D": "D",
        "HOURLY": "H", "HOUR": "H", "60": "H", "1H": "H", "H": "H",
        "5MIN": "5", "5M": "5", "5": "5",
        "15MIN": "15", "15M": "15", "15": "15",
        "1MIN": "1", "1M": "1", "1": "1",
        "WEEKLY": "W", "WEEK": "W", "1W": "W", "W": "W",
        "MONTHLY": "M", "MONTH": "M", "1MO": "M", "M": "M",
    }
    return aliases.get(raw, raw)


def _parse_atr_period_and_resolution(params: dict) -> tuple[int, str]:
    """Parse ATR period safely, never int()-parsing non-numeric timeframe strings."""
    raw_period = params.get("period", 14)
    raw_resolution = params.get("resolution")

    if raw_resolution is None and isinstance(raw_period, str):
        token = raw_period.strip()
        if token and not token.replace(".", "", 1).isdigit():
            raw_resolution = token
            raw_period = 14

    if raw_period is None:
        period = 14
    elif isinstance(raw_period, (int, float)):
        period = int(raw_period)
    else:
        token = str(raw_period).strip()
        period = int(float(token)) if token.replace(".", "", 1).isdigit() else 14

    period = max(1, period)
    resolution = _normalize_resolution(raw_resolution or "D")
    return period, resolution


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
            # Determine real-time status from source tag
            src = (quote.source or '').lower()
            is_realtime = 'realtime' in src or 'hybrid' in src or src.startswith('ibkr')
            data_warning = None if is_realtime else "DELAYED DATA — prices may be 15+ minutes old. Do not use for entry timing."
            ts = None
            if quote.source_updated:
                ts = datetime.datetime.utcfromtimestamp(quote.source_updated).strftime('%Y-%m-%dT%H:%M:%SZ')
            elif quote.timestamp:
                ts = quote.timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

            return {
                "symbol": quote.symbol,
                "last": quote.last,
                "bid": quote.bid,
                "ask": quote.ask,
                "volume": quote.volume,
                "change_pct": quote.change_pct,
                "is_realtime": is_realtime,
                "data_warning": data_warning,
                "timestamp": ts,
                "source": quote.source,
            }
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
    except Exception as e:
        logger.debug(f"UVXY quote fallback failed: {e}")

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
    except Exception as e:
        logger.debug(f"SPY IV fallback failed: {e}")

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
                "note": "VIX unavailable — using SPY ATR as proxy.",
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
            n = len(candles)
            dates = []
            for i in range(n):
                ts = candles.timestamps[i]
                if resolution == 'D':
                    dates.append(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else '')
                else:
                    dates.append(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else '')
            return {
                "symbol": symbol,
                "resolution": valid_resolutions[resolution],
                "bars": n,
                "is_realtime": True,
                "data_warning": None,
                "dates": dates,
                "open": [round(candles.open[i], 2) for i in range(n)],
                "high": [round(candles.high[i], 2) for i in range(n)],
                "low": [round(candles.low[i], 2) for i in range(n)],
                "close": [round(candles.close[i], 2) for i in range(n)],
                "volume": [candles.volume[i] for i in range(n)],
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
        period, resolution = _parse_atr_period_and_resolution(params)
        result = executor.data_provider.get_atr(symbol, period)
        if result:
            data = asdict(result)
            quote = executor.data_provider.get_quote(symbol)
            if quote and quote.last and quote.last > 0:
                data["atr_pct"] = round(result.value / quote.last * 100, 2)
            data["resolution"] = resolution
            data["is_realtime"] = True
            data["data_warning"] = None
            data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            return data
        return {"error": f"No ATR data for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_iv_info(executor, params: dict) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}

    # Agent-friendly defaults: 7-45 DTE covers near-term front-month IV
    # (the most actionable window for short-vol/long-vol decisions). Agent
    # may override either bound when investigating term-structure or LEAPS.
    dte_min = params.get("dte_min")
    dte_max = params.get("dte_max")
    if dte_min is None:
        dte_min = 7
    if dte_max is None:
        dte_max = 45
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
    """Return today's macro events + look-ahead. Always surface the next
    upcoming event so an empty short window is not mistaken for missing data.
    """
    try:
        from data.economic_calendar import get_todays_events, get_upcoming_events
        today_events = get_todays_events()
        upcoming_3d = get_upcoming_events(days=3)
        # Look further out to find the next high-impact event if short window is empty.
        upcoming_14d = get_upcoming_events(days=14)
        next_event = None
        for e in upcoming_14d:
            if e.date > (today_events[0].date if today_events else date.today()):
                next_event = e.to_dict()
                break
        if next_event is None and upcoming_14d:
            next_event = upcoming_14d[0].to_dict()
        result = {
            "today": [e.to_dict() for e in today_events],
            "upcoming_3d": [e.to_dict() for e in upcoming_3d],
            "count_today": len(today_events),
            "count_upcoming": len(upcoming_3d),
            "next_high_impact": next_event,
        }
        if not today_events and not upcoming_3d:
            result["note"] = "No high-impact US macro events in the next 3 days."
        return result
    except Exception as e:
        logger.warning(f"economic_calendar failed: {e}")
        return {"today": [], "upcoming_3d": [], "count_today": 0, "count_upcoming": 0, "warning": str(e)}


# ── Briefing tool (hierarchical drill-down) ─────────────────────


def _query_briefing_data() -> dict:
    """Query all briefing data from signal combination engine."""
    from signals.briefing import query_briefing_data
    return query_briefing_data()


def _briefing_summary(data: dict) -> dict:
    """Composite briefing: signal quality, top composites, recommended trades."""
    from signals.briefing import briefing_summary
    return briefing_summary(data)


def _briefing_signals(data: dict) -> dict:
    """Full recommended trade list with composite + template details."""
    from signals.briefing import briefing_signals
    return briefing_signals(data)


def _briefing_strategies(data: dict) -> dict:
    """Template performance table."""
    from signals.briefing import briefing_strategies
    return briefing_strategies(data)


def _briefing_feedback(data: dict) -> dict:
    """Full trade feedback with stats."""
    from research.simulator import format_confidence_line
    feedback = data.get("feedback", [])
    if not feedback:
        return {"feedback": "No trade feedback in last 7 days"}

    avg_gap = sum(r.get("execution_gap") or 0 for r in feedback) / len(feedback)
    avg_pnl = sum(r.get("actual_pnl") or 0 for r in feedback) / len(feedback)
    sim_returns = [r["simulated_return"] for r in feedback if r.get("simulated_return") is not None]
    actual_pnls = [r["actual_pnl"] for r in feedback if r.get("actual_pnl") is not None]
    gaps = [r["execution_gap"] for r in feedback if r.get("execution_gap") is not None]

    result: dict = {
        "trades": len(feedback),
        "avg_pnl": round(avg_pnl, 2),
        "avg_gap": round(avg_gap, 3),
    }
    for label, vals, kind in [
        ("sim_return_ci", sim_returns, "pct"),
        ("actual_pnl_ci", actual_pnls, "usd"),
        ("gap_ci", gaps, None),
    ]:
        line = format_confidence_line(label, vals, kind=kind or "raw")
        if line:
            result[label] = line

    # Per-symbol execution cost breakdown
    try:
        from memory import get_execution_cost
        top = get_execution_cost()  # top-10 symbols by trade count
        if top:
            result["per_symbol_cost"] = top
    except Exception:
        pass

    return result


def _briefing_environment(data: dict) -> dict:
    """Full environment regimes + signal health + N_eff trend."""
    from signals.briefing import briefing_environment
    return briefing_environment(data)


async def handle_briefing(executor, params: dict) -> Any:
    """Hierarchical research briefing tool.

    detail=None or "summary" -> compact overview (~100 tokens)
    detail="signals"         -> full live signals
    detail="strategies"      -> template performance
    detail="feedback"        -> trade feedback stats
    detail="environment"     -> full regimes + fit scores + history
    """
    detail = params.get("detail", "summary")
    data = _query_briefing_data()
    if not data:
        return {"briefing": "No research data available yet. Run research pipeline first."}

    # Add Independent Mode awareness
    try:
        ctx = get_operating_context()
        if ctx.is_independent_mode:
            data["note"] = "Running in Independent Mode (researcher unavailable). Data may be stale."
    except Exception:
        pass

    if detail == "signals":
        return _briefing_signals(data)
    elif detail == "strategies":
        return _briefing_strategies(data)
    elif detail == "feedback":
        return _briefing_feedback(data)
    elif detail == "environment":
        return _briefing_environment(data)
    else:
        return _briefing_summary(data)


async def handle_prior_research(executor, params: dict) -> Any:
    """Return the agent's cached research results for reuse."""
    agent = getattr(executor, '_agent', None)
    if agent is None:
        return {"prior_research": [], "note": "No research cache available"}

    cache = getattr(agent, '_research_results', {})
    if not cache:
        return {"prior_research": [], "note": "No prior research cached this session"}

    entries = []
    for query, entry in cache.items():
        entries.append({
            "query": query,
            "summary": entry.get("summary", "")[:600],
            "time": entry.get("ts", "?"),
            "category": entry.get("category", "?"),
        })

    return {"prior_research": entries, "count": len(entries)}


# ═══════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME CHART TOOLS
# ═══════════════════════════════════════════════════════════════

def _format_chart_frame(candles, resolution: str, label: str) -> dict | None:
    """Format a single timeframe into compact columnar output with analytics."""
    if not candles or len(candles) == 0:
        return None
    n = len(candles)
    closes = candles.close
    highs = candles.high
    lows = candles.low
    volumes = candles.volume

    # Derived analytics
    atr_vals = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        atr_vals.append(tr)
    atr_14 = round(sum(atr_vals[-14:]) / min(len(atr_vals), 14), 4) if atr_vals else 0

    # Trend label (simple: last close vs first close)
    trend = "up" if closes[-1] > closes[0] else "down" if closes[-1] < closes[0] else "flat"

    # Relative volume (last bar vs average)
    avg_vol = sum(volumes) / n if n > 0 else 1
    rel_vol = round(volumes[-1] / avg_vol, 2) if avg_vol > 0 else 0

    # Date formatting
    is_intraday = resolution not in ("D", "W", "M")
    dates = []
    for i in range(n):
        ts = candles.timestamps[i]
        if ts:
            fmt = '%Y-%m-%d %H:%M' if is_intraday else '%Y-%m-%d'
            dates.append(datetime.datetime.fromtimestamp(ts).strftime(fmt))
        else:
            dates.append('')

    # Keep chart payload compact for LLM context. Intraday lookbacks can be
    # very large (thousands of bars), which can explode token usage.
    max_bars = 120
    start = max(0, n - max_bars)
    returned_n = n - start

    return {
        "frame": label,
        "bars_total": n,
        "bars": returned_n,
        "truncated": n > returned_n,
        "dates": dates[start:],
        "open": [round(candles.open[i], 2) for i in range(start, n)],
        "high": [round(highs[i], 2) for i in range(start, n)],
        "low": [round(lows[i], 2) for i in range(start, n)],
        "close": [round(closes[i], 2) for i in range(start, n)],
        "volume": list(volumes[start:]),
        "atr_14": atr_14,
        "trend": trend,
        "rel_volume": rel_vol,
    }


async def _fetch_multi_frames(executor, symbol: str, frames: list[tuple[str, int, str]]) -> dict:
    """Fetch multiple timeframes concurrently and return formatted result.

    frames: list of (resolution, bars, label) tuples.
    """
    import asyncio

    async def _get(res, bars, label):
        c = await asyncio.to_thread(
            executor.data_provider.get_candles, symbol, resolution=res, days_back=bars
        )
        return _format_chart_frame(c, res, label)

    tasks = [_get(res, bars, label) for res, bars, label in frames]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    chart_frames = []
    for r in results:
        if isinstance(r, dict):
            chart_frames.append(r)

    if not chart_frames:
        return {"error": f"No chart data for {symbol}"}
    return {"symbol": symbol, "frames": chart_frames}


async def handle_chart_intraday(executor, params: dict) -> Any:
    """1min 30 bars + 5min 30 bars + 15min 20 bars. For active day-trade entries."""
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    return await _fetch_multi_frames(executor, symbol, [
        ("1", 30, "1min"), ("5", 30, "5min"), ("15", 20, "15min"),
    ])


async def handle_chart_swing(executor, params: dict) -> Any:
    """Hourly 20 bars + daily 30 bars + weekly 12 bars. For swing/multi-day setups."""
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    return await _fetch_multi_frames(executor, symbol, [
        ("H", 20, "hourly"), ("D", 30, "daily"), ("W", 12, "weekly"),
    ])


async def handle_chart_full(executor, params: dict) -> Any:
    """5min 30 + hourly 20 + daily 30. Validate a strong thesis across timeframes."""
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    return await _fetch_multi_frames(executor, symbol, [
        ("5", 30, "5min"), ("H", 20, "hourly"), ("D", 30, "daily"),
    ])


async def handle_chart_quick(executor, params: dict) -> Any:
    """Daily 10 bars + derived analytics. Fast screening tool."""
    symbol = params.get("symbol")
    if not symbol:
        return {"error": "symbol required"}
    return await _fetch_multi_frames(executor, symbol, [
        ("D", 10, "daily"),
    ])


async def handle_execution_status(executor, params: dict) -> Any:
    """Return execution autoresearch status: snapshot stats, graduated params, calibrated slippage."""
    try:
        from datetime import datetime, timedelta, timezone

        from memory import get_calibrated_slippage, get_db, get_graduated_params

        db = get_db()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        # Snapshot summary
        row = db.execute("SELECT COUNT(*) as n FROM execution_snapshots WHERE status = 'filled'").fetchone()
        snap_total = row["n"] if row else 0
        row = db.execute(
            "SELECT COUNT(*) as n FROM execution_snapshots WHERE status = 'filled' AND ts > ?",
            (cutoff,),
        ).fetchone()
        snap_recent = row["n"] if row else 0

        # Graduated params
        grads = get_graduated_params(active_only=True)
        grad_list = [
            {"key": g["param_key"], "value": g["param_value"], "improvement_bps": g["improvement_bps"], "p_value": g["p_value"]}
            for g in grads
        ]

        # Calibrated slippage
        cal = get_calibrated_slippage()
        cal_list = [
            {"order_type": k[0], "time_bucket": k[1], "atr_bucket": k[2], "median_bps": round(v, 2)}
            for k, v in cal.items()
        ]

        return {
            "snapshots_total": snap_total,
            "snapshots_7d": snap_recent,
            "graduated_params": grad_list,
            "calibrated_slippage": cal_list,
        }
    except Exception as e:
        return {"error": str(e)}


async def handle_open_hypotheses(executor, params: dict) -> Any:
    """Return open trader hypotheses for review."""
    try:
        from memory import get_db
        db = get_db()
        rows = db.execute(
            """SELECT id, hypothesis_type, description, suggested_action, priority, ts
               FROM trader_hypotheses
               WHERE status = 'open'
               ORDER BY priority ASC, ts DESC LIMIT 15""",
        ).fetchall()
        hyps = [
            {
                "id": r["id"],
                "type": r["hypothesis_type"],
                "desc": r["description"][:200],
                "action": (r["suggested_action"] or "")[:100],
                "priority": r["priority"],
                "date": r["ts"][:10],
            }
            for r in rows
        ]
        return {"hypotheses": hyps, "count": len(hyps)}
    except Exception as e:
        return {"error": str(e)}


_EVOLUTION_DAEMON_ONLY = (
    "Template evolution runs only in python -m research — "
    "not started or controlled from the trader process."
)


async def handle_research_engine(executor, params: dict) -> dict:
    """Agent-controlled in-process scorer only.

    params.action in {"status", "start", "pause", "resume", "stop"}.
    scope='scorer' | 'both' affects the scorer. scope='evolution' is ignored for
    control actions (evolution is owned by python -m research); status still reports policy.
    """
    from core.config import TRADER_IN_PROCESS_SCORER_NEVER
    from signals import scorer as _sc
    from signals import template_evolution as _te

    action = str(params.get("action", "status")).lower().strip()
    scope = str(params.get("scope", "both")).lower().strip()
    want_scorer = scope in ("both", "scorer")
    want_evo = scope in ("both", "evolution")

    result: dict[str, Any] = {}
    if action == "status":
        pass  # fall through to unified status report below
    elif TRADER_IN_PROCESS_SCORER_NEVER and action in ("start", "resume") and want_scorer:
        result["error"] = (
            "In-process scorer is disabled (TRADER_IN_PROCESS_SCORER=never). "
            "Run python -m research on the research host."
        )
    elif action == "start":
        if want_scorer and not _sc.is_scorer_running():
            _sc.run_research_threaded(verbose=False)
            result["scorer_started"] = True
        if want_evo:
            result["evolution_note"] = _EVOLUTION_DAEMON_ONLY
    elif action == "pause":
        if want_scorer:
            result["scorer_paused"] = _sc.pause_scorer()
        if want_evo:
            result["evolution_note"] = _EVOLUTION_DAEMON_ONLY
    elif action == "resume":
        if want_scorer:
            result["scorer_resumed"] = _sc.resume_scorer()
        if want_evo:
            result["evolution_note"] = _EVOLUTION_DAEMON_ONLY
    elif action == "stop":
        if want_scorer:
            result["scorer_stopped"] = _sc.stop_scorer()
        if want_evo:
            result["evolution_note"] = _EVOLUTION_DAEMON_ONLY
    else:
        return {"error": f"unknown action '{action}'. Use: status|start|pause|resume|stop"}

    result["scorer"] = {
        "running": _sc.is_scorer_running(),
        "paused": _sc.is_scorer_paused(),
    }
    result["evolution"] = {
        "running_in_trader_process": _te.is_evolution_running(),
        "paused_in_trader_process": _te.is_evolution_paused(),
        "policy": _EVOLUTION_DAEMON_ONLY,
    }
    return result


# ── Context Quality Tool (Independent Mode support) ──────────────────────────


async def handle_context_quality(executor, params: dict) -> dict:
    """Return the current quality and reliability of the trader's information.

    The agent should call this explicitly when it wants to understand how good
    its own context is (especially in Independent Mode when the researcher is down).
    """
    ctx = get_operating_context()
    q = ctx.quality

    return {
        "researcher_available": q.researcher_available,
        "memory_source": q.memory_source,
        "last_research_update_minutes_ago": (
            (dt_datetime.now(timezone.utc) - q.last_research_update).total_seconds() / 60
            if q.last_research_update else None
        ),
        "working_memory_completeness": round(q.working_memory_completeness, 2),
        "hypotheses_available": q.hypotheses_available,
        "overall_quality": q.overall_quality,
        "summary": q.to_prompt_block(ctx.risk_multiplier),
    }


# ── Quality inspection tools (read-only; enforcement is host-side) ─────────────
# quality_status, quality_for_symbol, provenance_audit — see docs/operations/independent-mode.md


def _validate_symbol(raw: Any) -> str | None:
    """Local symbol validator (keeps the quality tools self-contained)."""
    if not isinstance(raw, str):
        return None
    sym = raw.strip().upper()
    if not sym or len(sym) > 10:
        return None
    if not all(c.isalnum() or c in ".-" for c in sym):
        return None
    return sym


def _recent_feedback_for_symbol(sym: str, limit: int = 30) -> list[dict]:
    """Recent trade_feedback rows for a symbol (robust)."""
    try:
        from memory import get_db
        db = get_db()
        cur = db.execute(
            "SELECT * FROM trade_feedback WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
            (sym, limit),
        )
        rows: list[dict] = []
        for r in cur.fetchall():
            if hasattr(r, "keys"):
                rows.append(dict(r))
            else:
                rows.append({"symbol": sym})
        return rows
    except Exception as e:
        logger.debug("quality feedback read failed for %s: %s", sym, e)
        return []


def _maybe_refresh_matrix(svc) -> None:
    """Re-populate matrix if older than five minutes (best-effort)."""
    try:
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        m = svc.get_matrix()
        age_min = (_dt.now(_tz.utc) - m.last_populated).total_seconds() / 60
        if age_min > 5:
            svc.populate()
    except Exception:
        pass


async def handle_quality_status(executor, params: dict) -> dict:
    """Read-only system quality snapshot from QualityMatrix + operating context.

    Primary fields (authoritative from QualityMatrix when available):
      - overall_quality, risk_multiplier, blocked_tool_categories,
        force_conservative_reasoning, suggested model config
      - matrix_llm_config (the exact temp/max_tokens/reasoning_bias the host will use)
      - matrix_can_new_risk, matrix_recent_provenance_count
      - summary: the compact to_prompt_block() string
      - aggregate_execution from trade_feedback
      - researcher/memory status for context

    This is the #1 tool the agent must call early in Independent Mode (see
    agent.py INDEPENDENT MODE RULES). It tells the agent the hard policy surface
    without the agent having to guess.

    Pure inspection. Never causes trades, research, or state changes.
    """
    ctx = get_operating_context()
    q = ctx.quality

    # Lightweight aggregate (synthesizes what a future QualityMatrix will provide)
    agg: dict[str, Any] = {"n_recent_trades": 0, "avg_gap": None, "symbols_covered": 0}
    try:
        from memory import get_db
        db = get_db()
        rows = db.execute(
            "SELECT symbol, execution_gap FROM trade_feedback ORDER BY ts DESC LIMIT 150"
        ).fetchall()
        syms: set[str] = set()
        gaps: list[float] = []
        for r in rows:
            d = dict(r) if hasattr(r, "keys") else {}
            s = d.get("symbol") or (r[0] if isinstance(r, (list, tuple)) and len(r) > 0 else None)
            if s:
                syms.add(str(s).upper())
            g = d.get("execution_gap")
            if g is not None:
                try:
                    gaps.append(float(g))
                except Exception:
                    pass
        agg = {
            "n_recent_trades": len(rows),
            "avg_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
            "symbols_covered": len(syms),
        }
    except Exception as e:
        logger.debug("quality_status agg failed: %s", e)

    last_age = None
    if q.last_research_update:
        last_age = (dt_datetime.now(timezone.utc) - q.last_research_update).total_seconds() / 60

    matrix_enrichment: dict[str, Any] = {}
    try:
        from core.quality.quality_matrix import get_quality_matrix_service
        svc = get_quality_matrix_service()
        _maybe_refresh_matrix(svc)
        m = svc.get_matrix()
        m.recommended_policies()
        llm_cfg = m.get_llm_call_config()
        matrix_enrichment = {
            "matrix_overall_quality": m.overall_quality,
            "matrix_risk_multiplier": round(m.risk_multiplier, 3),
            "matrix_blocked_categories": list(m.blocked_tool_categories),
            "matrix_force_conservative": bool(m.force_conservative_reasoning),
            "matrix_llm_config": {
                "temperature": llm_cfg.get("temperature"),
                "max_tokens": llm_cfg.get("max_tokens"),
                "reasoning_bias": llm_cfg.get("reasoning_bias"),
                "source": llm_cfg.get("source"),
            },
            "matrix_can_new_risk": m.can_initiate_new_risk(),
            "matrix_recent_provenance_count": len(m.recent_provenance),
            "matrix_recent_tool_count": len(m.recent_tool_usage),
            "matrix_global_exec_quality": getattr(m, "global_execution_quality", None),
            "matrix_last_populated_minutes_ago": round(
                (dt_datetime.now(timezone.utc) - m.last_populated).total_seconds() / 60, 1
            ),
        }
        # Prefer matrix values for the primary keys when available (source of truth)
        overall_for_return = m.overall_quality
        rm_for_return = m.risk_multiplier
        summary_block = m.to_prompt_block(m.risk_multiplier)
    except Exception as _mat_err:
        logger.debug("quality_status matrix enrichment failed: %s", _mat_err)
        overall_for_return = q.overall_quality
        rm_for_return = ctx.risk_multiplier
        summary_block = q.to_prompt_block(ctx.risk_multiplier)

    return {
        "overall_quality": overall_for_return,
        "researcher_available": q.researcher_available,
        "memory_source": q.memory_source,
        "last_research_update_minutes_ago": last_age,
        "working_memory_completeness": round(q.working_memory_completeness, 2),
        "hypotheses_available": q.hypotheses_available,
        "risk_multiplier": round(rm_for_return, 3),
        "aggregate_execution": agg,
        "summary": summary_block,
        "matrix": matrix_enrichment or {"note": "core matrix unavailable this call (fallback to legacy context)"},
        "enforcement_note": (
            "Host enforces model config, tool blocks, and quantity scaling via QualityMatrix. "
            "Use provenance_audit() for recent tool and decision history."
        ),
    }


async def handle_quality_for_symbol(executor, params: dict) -> dict:
    """Read-only per-symbol quality from recent trade_feedback.

    Returns per-symbol quality metrics derived from trade_feedback (the same source
    that feeds SymbolQuality in the canonical matrix).

    Response includes:
      - quality_score (0.3–0.95 heuristic: lower |avg_gap| + higher feedback count = higher)
      - avg_execution_gap_pct, feedback_count, recent_pnl_samples, last_feedback_ts

    In Independent Mode the agent is expected to call this for every name it is
    considering before using local WM or signals for conviction or sizing.

    Pure read-only. Complements quality_status() (system) and provenance_audit()
    (tool history on that symbol's decisions).
    """
    sym = _validate_symbol(params.get("symbol"))
    if sym is None:
        return {"error": "symbol required (1-10 alphanumeric chars)"}

    rows = _recent_feedback_for_symbol(sym, limit=25)

    if not rows:
        return {
            "symbol": sym,
            "feedback_count": 0,
            "quality_score": 0.45,
            "notes": "No recent trade feedback. Treat any thesis for this symbol as lower conviction in Independent Mode.",
            "last_feedback": None,
        }

    gaps: list[float] = []
    pnls: list[float] = []
    last_ts = None
    for r in rows:
        g = r.get("execution_gap")
        if g is not None:
            try:
                gaps.append(float(g))
            except Exception:
                pass
        p = r.get("actual_pnl")
        if p is not None:
            try:
                pnls.append(float(p))
            except Exception:
                pass
        if last_ts is None and r.get("ts"):
            last_ts = r.get("ts")

    avg_gap = round(sum(gaps) / len(gaps), 2) if gaps else None
    score = 0.72
    if avg_gap is not None:
        penalty = min(0.35, max(0.0, (abs(avg_gap) - 5.0) / 40.0))
        score -= penalty
    score += min(0.12, len(rows) * 0.004)
    score = max(0.30, min(0.95, round(score, 3)))

    return {
        "symbol": sym,
        "feedback_count": len(rows),
        "avg_execution_gap_pct": avg_gap,
        "recent_pnl_samples": [round(p, 2) for p in pnls[:5]],
        "quality_score": score,
        "last_feedback_ts": last_ts,
        "notes": (
            "Lower |gap| and higher sample count increase local trust. "
            "Always cross-check against quality_status() (global posture) + "
            "provenance_audit(window, symbol) (what tools + decisions were actually recorded for this name)."
        ),
    }


async def handle_current_constraints(executor, params: dict) -> dict:
    """Legacy constraints summary; prefer quality_status and provenance_audit.

    New code and Independent Mode prompts should prefer the quality inspect tools
    (quality_status for full matrix policy, quality_for_symbol for per-name, and
    especially provenance_audit for the heavy tool-usage + decision history).
    This is a lightweight derived view only.
    """
    ctx = get_operating_context()
    q = ctx.quality
    mult = ctx.risk_multiplier
    is_ind = ctx.is_independent_mode

    if mult < 0.5:
        pass
    elif mult < 0.8:
        pass
    else:
        pass

    return {
        "risk_multiplier": round(mult, 3),
        "effective_mode": "independent" if is_ind else "full_research",
        "overall_quality": q.overall_quality,
        "new_entry_policy": "high_conviction_only" if is_ind else "standard",
        "prefer_position_management": bool(mult < 0.75),
        "max_risk_scaling_hint": round(1.5 * mult, 2),
        "reason": (
            "Researcher unavailable or memory on local fallback; automatic risk reduction active."
            if is_ind
            else "Full research context available."
        ),
        "enforcement_note": "Legacy view. Use quality_status() + provenance_audit() for authoritative policy and provenance.",
    }


async def handle_provenance_audit(executor, params: dict) -> dict:
    """Read-only audit of recent tool usage and decision provenance snapshots.

    Returns two arrays from QualityMatrix (populated
    by record_tool_usage + record_decision_snapshot calls throughout the host):

    recent_tool_usage: list of ToolUsageRecord dicts
        {tool_name, called_at, symbol, success, latency_ms, source, context}

    recent_provenance: list of DecisionProvenanceSnapshot dicts
        {ts, cycle_id, decision_type, symbol, tools_used (list of names active at decision),
         context_quality, outcome, notes, quality_state_keys}

    Supports:
      - window (int, default 12, clamped 1-30): how many most-recent items
      - symbol (str): filter both lists to records mentioning that symbol

    This is the mechanism that gives the agent real visibility into "what tools
    did we actually invoke and what was the decision context at the time?"
    Critical for Independent Mode staleness detection, over-reliance diagnosis,
    and deciding whether local WM + signals have solid historical tool backing.

    Pure read-only. The act of calling this tool itself is recorded (via executor)
    so it appears in future provenance_audit calls — useful for self-audit.

    See QualityMatrixService.record_* and the dataclass definitions in
    core/quality/quality_matrix.py.
    """
    try:
        from core.quality.quality_matrix import get_quality_matrix_service
        svc = get_quality_matrix_service()
        _maybe_refresh_matrix(svc)
        m = svc.get_matrix()

        # Parse window safely
        raw_window = params.get("window")
        try:
            window = int(raw_window) if raw_window is not None else 12
        except Exception:
            window = 12
        window = max(1, min(30, window))

        # Symbol filter (reuse validator)
        sym_filter = _validate_symbol(params.get("symbol")) if params.get("symbol") else None

        # Recent tool usage (most recent last; apply filter + slice)
        tools_src = m.recent_tool_usage
        if sym_filter:
            tools_src = [t for t in tools_src if (t.symbol or "").upper() == sym_filter]
        tools_recent = tools_src[-window:] if tools_src else []

        tool_records = []
        for t in tools_recent:
            tool_records.append({
                "tool_name": t.tool_name,
                "called_at": getattr(t.called_at, "isoformat", lambda: str(t.called_at))(),
                "symbol": t.symbol,
                "success": bool(t.success),
                "latency_ms": round(float(t.latency_ms or 0), 1),
                "source": getattr(t, "source", None) or "executor",
                "context": getattr(t, "context", {}) or {},
            })

        # Recent provenance snapshots (decision-scoped tool sets)
        prov_src = m.recent_provenance
        if sym_filter:
            prov_src = [p for p in prov_src if (p.symbol or "").upper() == sym_filter]
        prov_recent = prov_src[-window:] if prov_src else []

        prov_records = []
        for p in prov_recent:
            tools_in_snap = [tt.tool_name for tt in (p.tools_used or [])[-6:]]
            prov_records.append({
                "ts": getattr(p.ts, "isoformat", lambda: str(p.ts))(),
                "cycle_id": int(p.cycle_id or 0),
                "decision_type": p.decision_type or "cycle_decision",
                "symbol": p.symbol,
                "tools_used": tools_in_snap,
                "context_quality": p.context_quality or "full",
                "outcome": p.outcome,
                "notes": (p.notes or "")[:120],
                "quality_state_keys": list((p.quality_state or {}).keys())[:5],
            })

        return {
            "window": window,
            "symbol_filter": sym_filter,
            "total_tool_records": len(m.recent_tool_usage),
            "total_provenance_snapshots": len(m.recent_provenance),
            "returned_tool_records": len(tool_records),
            "returned_provenance": len(prov_records),
            "recent_tool_usage": tool_records,
            "recent_provenance": prov_records,
            "summary": (
                f"{len(tool_records)} tool calls + {len(prov_records)} decision snapshots "
                f"(filter={sym_filter or 'all'}, window={window})."
            ),
            "enforcement_note": (
                "Inspect-only. Host enforces blocks, scaling, and model params via QualityMatrix."
            ),
        }
    except Exception as exc:
        logger.debug("provenance_audit failed: %s", exc)
        return {
            "error": str(exc),
            "note": "Provenance unavailable (first run, disabled, or schema pending).",
            "recent_tool_usage": [],
            "recent_provenance": [],
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
    "briefing": handle_briefing,
    "prior_research": handle_prior_research,
    "chart_intraday": handle_chart_intraday,
    "chart_swing": handle_chart_swing,
    "chart_full": handle_chart_full,
    "chart_quick": handle_chart_quick,
    "execution_status": handle_execution_status,
    "trader_rules": handle_execution_status,
    "open_hypotheses": handle_open_hypotheses,
    "research_engine": handle_research_engine,
    "context_quality": handle_context_quality,
}


HANDLERS["quality_status"] = handle_quality_status
HANDLERS["quality_for_symbol"] = handle_quality_for_symbol
HANDLERS["provenance_audit"] = handle_provenance_audit
HANDLERS["current_constraints"] = handle_current_constraints




