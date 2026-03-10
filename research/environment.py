"""
Market Environment Assessment — regime detection from candle data.

Computes a structured snapshot of market conditions that the meta-learning
selector uses to match strategies to environments and optimize slot allocation.

Pure pandas/numpy — no LLM calls. Runs once per research round on the shared
candle universe.
"""

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Regime labels ───────────────────────────────────────────────

VOLATILITY_REGIMES = ("low", "normal", "high", "extreme")
TREND_REGIMES = ("strong_down", "down", "sideways", "up", "strong_up")
BREADTH_REGIMES = ("bearish", "mixed", "neutral", "bullish")


# ── Per-symbol feature extraction ───────────────────────────────

def _symbol_features(days: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute features for a single symbol across its trading days.

    Args:
        days: {date_str: candle_df} for one symbol

    Returns:
        Feature dict for this symbol.
    """
    if not days:
        return {}

    # Build daily OHLCV summary from intraday bars
    daily_rows = []
    for date_str in sorted(days.keys()):
        df = days[date_str]
        if df.empty:
            continue
        daily_rows.append({
            "date": date_str,
            "open": df["open"].iloc[0],
            "high": df["high"].max(),
            "low": df["low"].min(),
            "close": df["close"].iloc[-1],
            "volume": df["volume"].sum(),
            "bar_count": len(df),
            # Intraday range as % of open
            "intraday_range_pct": (df["high"].max() - df["low"].min()) / df["open"].iloc[0] * 100,
            # Average bar range (proxy for intraday volatility)
            "avg_bar_range": ((df["high"] - df["low"]) / df["open"].iloc[0] * 100).mean(),
        })

    if len(daily_rows) < 2:
        return {}

    daily = pd.DataFrame(daily_rows)

    # ATR (Average True Range) as % of price
    daily["prev_close"] = daily["close"].shift(1)
    daily["tr"] = np.maximum(
        daily["high"] - daily["low"],
        np.maximum(
            abs(daily["high"] - daily["prev_close"]),
            abs(daily["low"] - daily["prev_close"]),
        ),
    )
    atr = daily["tr"].dropna().mean()
    avg_price = daily["close"].mean()
    atr_pct = (atr / avg_price * 100) if avg_price > 0 else 0

    # Daily returns
    daily["return_pct"] = daily["close"].pct_change() * 100
    returns = daily["return_pct"].dropna()
    cumulative_return = ((1 + returns / 100).prod() - 1) * 100 if len(returns) > 0 else 0

    # Trend: SMA crossover direction
    closes = daily["close"].values
    if len(closes) >= 5:
        sma_fast = np.mean(closes[-3:])  # 3-day
        sma_slow = np.mean(closes[-5:])  # 5-day (short window for 10-day data)
        trend_strength = (sma_fast - sma_slow) / sma_slow * 100
    else:
        trend_strength = 0

    # Momentum: recent vs earlier returns
    if len(returns) >= 4:
        recent = returns.iloc[-2:].mean()
        earlier = returns.iloc[:-2].mean()
        momentum_shift = recent - earlier
    else:
        momentum_shift = 0

    # Volume trend
    volumes = daily["volume"].values
    if len(volumes) >= 4:
        recent_vol = np.mean(volumes[-2:])
        earlier_vol = np.mean(volumes[:-2])
        vol_ratio = recent_vol / earlier_vol if earlier_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    # Intraday volatility mean
    avg_intraday_range = daily["intraday_range_pct"].mean()

    return {
        "atr_pct": round(atr_pct, 4),
        "avg_intraday_range": round(avg_intraday_range, 4),
        "cumulative_return": round(cumulative_return, 4),
        "trend_strength": round(trend_strength, 4),
        "momentum_shift": round(momentum_shift, 4),
        "vol_ratio": round(vol_ratio, 4),
        "daily_return_std": round(returns.std(), 4) if len(returns) > 1 else 0,
        "last_close": round(closes[-1], 2),
        "avg_daily_volume": round(daily["volume"].mean(), 0),
        "num_days": len(daily),
    }


# ── Universe-level environment snapshot ─────────────────────────

def compute_environment(
    universe: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    """Compute a comprehensive market environment snapshot.

    Args:
        universe: {symbol: {date_str: candle_df}} — the shared candle data

    Returns:
        Environment snapshot dict with regime classifications and raw metrics.
    """
    if not universe:
        return {"regime": "unknown", "symbols_analyzed": 0}

    # Compute per-symbol features
    sym_features: dict[str, dict] = {}
    for sym, days in universe.items():
        feat = _symbol_features(days)
        if feat:
            sym_features[sym] = feat

    if not sym_features:
        return {"regime": "unknown", "symbols_analyzed": 0}

    n = len(sym_features)

    # ── Volatility regime ───────────────────────────────────────
    atr_values = [f["atr_pct"] for f in sym_features.values()]
    intraday_ranges = [f["avg_intraday_range"] for f in sym_features.values()]
    avg_atr = np.mean(atr_values)
    avg_intraday = np.mean(intraday_ranges)

    # Classify volatility
    if avg_atr < 1.0:
        vol_regime = "low"
    elif avg_atr < 2.0:
        vol_regime = "normal"
    elif avg_atr < 3.5:
        vol_regime = "high"
    else:
        vol_regime = "extreme"

    # ── Trend regime (breadth-based) ────────────────────────────
    trends = [f["trend_strength"] for f in sym_features.values()]
    cum_returns = [f["cumulative_return"] for f in sym_features.values()]

    up_trending = sum(1 for t in trends if t > 0.3)
    down_trending = sum(1 for t in trends if t < -0.3)
    sideways = n - up_trending - down_trending

    pct_up = up_trending / n
    pct_down = down_trending / n

    if pct_up > 0.6:
        trend_regime = "strong_up"
    elif pct_up > 0.4:
        trend_regime = "up"
    elif pct_down > 0.6:
        trend_regime = "strong_down"
    elif pct_down > 0.4:
        trend_regime = "down"
    else:
        trend_regime = "sideways"

    # ── Breadth ─────────────────────────────────────────────────
    positive_returns = sum(1 for r in cum_returns if r > 0)
    negative_returns = sum(1 for r in cum_returns if r < 0)
    advance_decline = (positive_returns - negative_returns) / n if n > 0 else 0

    if advance_decline > 0.3:
        breadth_regime = "bullish"
    elif advance_decline > 0.0:
        breadth_regime = "neutral"
    elif advance_decline > -0.3:
        breadth_regime = "mixed"
    else:
        breadth_regime = "bearish"

    # ── Dispersion (how differently symbols are behaving) ───────
    return_std = np.std(cum_returns) if len(cum_returns) > 1 else 0
    trend_std = np.std(trends) if len(trends) > 1 else 0
    dispersion = round(float(return_std + trend_std), 4)

    # ── Momentum regime ─────────────────────────────────────────
    momentum_shifts = [f["momentum_shift"] for f in sym_features.values()]
    avg_momentum = np.mean(momentum_shifts)
    # Positive = recent days stronger than earlier, negative = fading
    if avg_momentum > 0.5:
        momentum_regime = "accelerating"
    elif avg_momentum > -0.5:
        momentum_regime = "steady"
    else:
        momentum_regime = "decelerating"

    # ── Volume regime ───────────────────────────────────────────
    vol_ratios = [f["vol_ratio"] for f in sym_features.values()]
    avg_vol_ratio = np.mean(vol_ratios)
    if avg_vol_ratio > 1.3:
        volume_regime = "expanding"
    elif avg_vol_ratio > 0.8:
        volume_regime = "normal"
    else:
        volume_regime = "contracting"

    # ── Strategy fit scores ─────────────────────────────────────
    # Score how favorable the environment is for different strategy types
    strategy_fit = _compute_strategy_fit(
        vol_regime=vol_regime,
        trend_regime=trend_regime,
        breadth_regime=breadth_regime,
        momentum_regime=momentum_regime,
        dispersion=dispersion,
        avg_atr=avg_atr,
        avg_intraday=avg_intraday,
    )

    # ── Best symbols for each approach ──────────────────────────
    # Sorted by suitability for momentum (high trend + high vol)
    momentum_candidates = sorted(
        sym_features.keys(),
        key=lambda s: sym_features[s]["trend_strength"] * sym_features[s]["atr_pct"],
        reverse=True,
    )[:5]

    # Mean reversion candidates (low trend strength + high intraday range)
    reversion_candidates = sorted(
        sym_features.keys(),
        key=lambda s: sym_features[s]["avg_intraday_range"] / max(0.01, abs(sym_features[s]["trend_strength"])),
        reverse=True,
    )[:5]

    # Options candidates (high ATR = high IV proxy)
    options_candidates = sorted(
        sym_features.keys(),
        key=lambda s: sym_features[s]["atr_pct"],
        reverse=True,
    )[:5]

    snapshot = {
        "timestamp": date.today().isoformat(),
        "symbols_analyzed": n,

        # Regime classifications
        "volatility_regime": vol_regime,
        "trend_regime": trend_regime,
        "breadth_regime": breadth_regime,
        "momentum_regime": momentum_regime,
        "volume_regime": volume_regime,

        # Raw metrics
        "avg_atr_pct": round(float(avg_atr), 4),
        "avg_intraday_range_pct": round(float(avg_intraday), 4),
        "avg_cumulative_return": round(float(np.mean(cum_returns)), 4),
        "advance_decline_ratio": round(float(advance_decline), 4),
        "dispersion": dispersion,
        "avg_momentum_shift": round(float(avg_momentum), 4),
        "avg_volume_ratio": round(float(avg_vol_ratio), 4),
        "pct_trending_up": round(pct_up, 2),
        "pct_trending_down": round(pct_down, 2),

        # Strategy-environment fit scores (0-1)
        "strategy_fit": strategy_fit,

        # Symbol recommendations
        "momentum_candidates": momentum_candidates,
        "reversion_candidates": reversion_candidates,
        "options_candidates": options_candidates,

        # Per-symbol features (for detailed analysis)
        "per_symbol": sym_features,
    }

    logger.info(
        f"Environment: vol={vol_regime} trend={trend_regime} breadth={breadth_regime} "
        f"momentum={momentum_regime} volume={volume_regime} "
        f"dispersion={dispersion:.2f}"
    )

    return snapshot


def _compute_strategy_fit(
    *,
    vol_regime: str,
    trend_regime: str,
    breadth_regime: str,
    momentum_regime: str,
    dispersion: float,
    avg_atr: float,
    avg_intraday: float,
) -> dict[str, float]:
    """Score how favorable the current environment is for each strategy archetype.

    Returns dict of {strategy_type: score} where score is 0.0-1.0.
    Higher = more favorable environment for that strategy type.
    """
    scores = {}

    # ── Momentum / Breakout strategies ──────────────────────────
    # Favor: trending markets, expanding volume, higher vol
    momentum_score = 0.5
    if trend_regime in ("strong_up", "strong_down"):
        momentum_score += 0.3
    elif trend_regime in ("up", "down"):
        momentum_score += 0.15
    elif trend_regime == "sideways":
        momentum_score -= 0.2
    if momentum_regime == "accelerating":
        momentum_score += 0.15
    if vol_regime in ("high", "extreme"):
        momentum_score += 0.1
    elif vol_regime == "low":
        momentum_score -= 0.15
    scores["momentum_breakout"] = max(0, min(1, momentum_score))

    # ── Mean Reversion strategies ───────────────────────────────
    # Favor: sideways markets, normal vol, high intraday range
    reversion_score = 0.5
    if trend_regime == "sideways":
        reversion_score += 0.25
    elif trend_regime in ("strong_up", "strong_down"):
        reversion_score -= 0.2
    if vol_regime == "normal":
        reversion_score += 0.15
    elif vol_regime == "extreme":
        reversion_score -= 0.2
    if avg_intraday > 2.0:
        reversion_score += 0.1
    scores["mean_reversion"] = max(0, min(1, reversion_score))

    # ── VWAP strategies ─────────────────────────────────────────
    # Favor: normal-high vol, trending with pullbacks
    vwap_score = 0.5
    if vol_regime in ("normal", "high"):
        vwap_score += 0.15
    if trend_regime in ("up", "down"):
        vwap_score += 0.15
    elif trend_regime == "sideways":
        vwap_score += 0.1
    scores["vwap"] = max(0, min(1, vwap_score))

    # ── Options — Long calls/puts (directional) ────────────────
    # Favor: strong trend, expanding vol (IV rise helps)
    long_options_score = 0.5
    if trend_regime in ("strong_up", "strong_down"):
        long_options_score += 0.25
    elif trend_regime in ("up", "down"):
        long_options_score += 0.1
    if vol_regime == "high":
        long_options_score += 0.1  # some vol good for movement
    elif vol_regime == "extreme":
        long_options_score -= 0.1  # IV too high = expensive
    elif vol_regime == "low":
        long_options_score += 0.05  # cheap entry but less movement
    scores["long_options"] = max(0, min(1, long_options_score))

    # ── Options — Short premium (strangles, iron condors) ──────
    # Favor: sideways + high vol (sell into IV), contracting momentum
    short_premium_score = 0.5
    if trend_regime == "sideways":
        short_premium_score += 0.25
    elif trend_regime in ("strong_up", "strong_down"):
        short_premium_score -= 0.25
    if vol_regime in ("high", "extreme"):
        short_premium_score += 0.2  # rich premium to sell
    elif vol_regime == "low":
        short_premium_score -= 0.15  # not enough premium
    if momentum_regime == "decelerating":
        short_premium_score += 0.1
    scores["short_premium"] = max(0, min(1, short_premium_score))

    # ── Options — Straddle (volatility play) ────────────────────
    # Favor: low vol (cheap) with expected expansion, high dispersion
    straddle_score = 0.5
    if vol_regime == "low":
        straddle_score += 0.2  # cheap entry
    elif vol_regime == "extreme":
        straddle_score -= 0.2  # too expensive
    if dispersion > 3.0:
        straddle_score += 0.15  # stocks moving independently = events
    if momentum_regime == "accelerating":
        straddle_score += 0.1
    scores["straddle"] = max(0, min(1, straddle_score))

    # ── Vertical spreads ────────────────────────────────────────
    # Favor: moderate directional trend, moderate vol
    spread_score = 0.5
    if trend_regime in ("up", "down"):
        spread_score += 0.2
    elif trend_regime in ("strong_up", "strong_down"):
        spread_score += 0.1
    if vol_regime in ("normal", "high"):
        spread_score += 0.15
    scores["vertical_spread"] = max(0, min(1, spread_score))

    # ── Bracket / range-bound ───────────────────────────────────
    # Similar to mean reversion but with tighter parameters
    bracket_score = 0.5
    if trend_regime == "sideways":
        bracket_score += 0.2
    if vol_regime == "normal":
        bracket_score += 0.15
    if avg_intraday > 1.5:
        bracket_score += 0.1
    scores["bracket"] = max(0, min(1, bracket_score))

    return {k: round(v, 3) for k, v in scores.items()}


def format_environment_for_prompt(snapshot: dict) -> str:
    """Format environment snapshot as a concise text block for LLM prompts.

    This gets injected into the research system prompt and analysis prompt
    so each slot's LLM knows what regime it's optimizing for.
    """
    if not snapshot or snapshot.get("regime") == "unknown":
        return "(Environment data unavailable)"

    fit = snapshot.get("strategy_fit", {})
    fit_lines = []
    for stype, score in sorted(fit.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        fit_lines.append(f"  {stype:<20s} {bar} {score:.2f}")

    candidates = snapshot.get("momentum_candidates", [])
    rev_candidates = snapshot.get("reversion_candidates", [])
    opt_candidates = snapshot.get("options_candidates", [])

    return f"""CURRENT MARKET ENVIRONMENT (computed from {snapshot['symbols_analyzed']} symbols, last {snapshot.get('num_days', '?')} trading days):
  Volatility:  {snapshot['volatility_regime']} (ATR {snapshot['avg_atr_pct']:.2f}%, intraday range {snapshot['avg_intraday_range_pct']:.2f}%)
  Trend:       {snapshot['trend_regime']} ({snapshot['pct_trending_up']:.0%} up, {snapshot['pct_trending_down']:.0%} down)
  Breadth:     {snapshot['breadth_regime']} (A/D ratio {snapshot['advance_decline_ratio']:+.2f})
  Momentum:    {snapshot['momentum_regime']} (shift {snapshot['avg_momentum_shift']:+.2f})
  Volume:      {snapshot['volume_regime']} (ratio {snapshot['avg_volume_ratio']:.2f}x)
  Dispersion:  {snapshot['dispersion']:.2f}

STRATEGY-ENVIRONMENT FIT (higher = better match for current conditions):
{chr(10).join(fit_lines)}

TOP SYMBOL CANDIDATES:
  Momentum:       {', '.join(candidates)}
  Mean reversion: {', '.join(rev_candidates)}
  Options:        {', '.join(opt_candidates)}"""
