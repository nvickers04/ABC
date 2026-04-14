"""Research and market data tool handlers."""

import datetime
import logging
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)


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
            data["timestamp"] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
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
    try:
        from data.economic_calendar import get_todays_events, get_upcoming_events
        today_events = get_todays_events()
        upcoming = get_upcoming_events(days=3)
        return {
            "today": [e.to_dict() for e in today_events],
            "upcoming_3d": [e.to_dict() for e in upcoming],
            "count_today": len(today_events),
            "count_upcoming": len(upcoming),
        }
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
    from research.simulator import compute_sample_confidence, format_confidence_line
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
        line = format_confidence_line(label, vals, kind=kind)
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

    return {
        "frame": label,
        "bars": n,
        "dates": dates,
        "open": [round(candles.open[i], 2) for i in range(n)],
        "high": [round(highs[i], 2) for i in range(n)],
        "low": [round(lows[i], 2) for i in range(n)],
        "close": [round(closes[i], 2) for i in range(n)],
        "volume": list(volumes),
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
        from memory import get_db, get_graduated_params, get_calibrated_slippage
        db = get_db()

        # Snapshot summary
        row = db.execute("SELECT COUNT(*) as n FROM execution_snapshots WHERE status = 'filled'").fetchone()
        snap_total = row["n"] if row else 0
        row = db.execute(
            "SELECT COUNT(*) as n FROM execution_snapshots WHERE status = 'filled' AND ts > datetime('now', '-7 days')"
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
}
