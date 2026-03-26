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

VOLATILITY_REGIMES = ("low", "normal", "high")
TREND_REGIMES = ("down", "flat", "up")
BREADTH_REGIMES = ("bearish", "neutral", "bullish")


def make_environment_key(snapshot: dict[str, Any]) -> str:
    """Build a compact regime key for grouping strategy results by condition."""
    return (
        f"vol={snapshot.get('volatility_regime', 'unknown')}|"
        f"trend={snapshot.get('trend_regime', 'unknown')}|"
        f"breadth={snapshot.get('breadth_regime', 'unknown')}"
    )


def compute_session_environment(session_universe: dict[str, pd.DataFrame], session_date: str) -> dict[str, Any]:
    """Compute a same-day environment snapshot across all symbols for one session."""
    rows: list[dict[str, float]] = []
    for sym, df in session_universe.items():
        if df is None or df.empty:
            continue
        open_px = float(df["open"].iloc[0])
        close_px = float(df["close"].iloc[-1])
        if open_px <= 0:
            continue

        first_n = max(1, min(len(df), len(df) // 3 or 1))
        last_n = max(1, min(len(df), len(df) // 3 or 1))
        morning_close = float(df["close"].iloc[first_n - 1])
        afternoon_open = float(df["open"].iloc[-last_n])
        first_vol = float(df["volume"].iloc[:first_n].sum())
        last_vol = float(df["volume"].iloc[-last_n:].sum())

        rows.append({
            "symbol": sym,
            "session_return_pct": (close_px - open_px) / open_px * 100,
            "intraday_range_pct": (float(df["high"].max()) - float(df["low"].min())) / open_px * 100,
            "morning_return_pct": (morning_close - open_px) / open_px * 100,
            "afternoon_return_pct": (
                (close_px - afternoon_open) / afternoon_open * 100 if afternoon_open > 0 else 0.0
            ),
            "late_volume_ratio": (last_vol / first_vol) if first_vol > 0 else 1.0,
        })

    if not rows:
        snapshot = {
            "timestamp": session_date,
            "symbols_analyzed": 0,
            "volatility_regime": "unknown",
            "trend_regime": "unknown",
            "breadth_regime": "unknown",
            "momentum_regime": "unknown",
            "volume_regime": "unknown",
            "dispersion": 0.0,
            "strategy_fit": {},
        }
        snapshot["env_key"] = make_environment_key(snapshot)
        return snapshot

    frame = pd.DataFrame(rows)
    n = len(frame)
    avg_intraday = float(frame["intraday_range_pct"].mean())
    returns = frame["session_return_pct"].tolist()
    positive = sum(1 for r in returns if r > 0.15)
    negative = sum(1 for r in returns if r < -0.15)
    pct_up = positive / n if n else 0.0
    pct_down = negative / n if n else 0.0
    advance_decline = (positive - negative) / n if n else 0.0
    avg_accel = float((frame["afternoon_return_pct"] - frame["morning_return_pct"]).mean())
    avg_late_vol = float(frame["late_volume_ratio"].mean())
    dispersion = float(frame["session_return_pct"].std(ddof=0) + frame["intraday_range_pct"].std(ddof=0))

    # 3-level volatility (low / normal / high) — 27 total bins
    if avg_intraday < 1.5:
        vol_regime = "low"
    elif avg_intraday < 3.0:
        vol_regime = "normal"
    else:
        vol_regime = "high"

    # 3-level trend (down / flat / up)
    if pct_down > 0.45:
        trend_regime = "down"
    elif pct_up > 0.45:
        trend_regime = "up"
    else:
        trend_regime = "flat"

    # 3-level breadth (bearish / neutral / bullish)
    if advance_decline > 0.2:
        breadth_regime = "bullish"
    elif advance_decline < -0.2:
        breadth_regime = "bearish"
    else:
        breadth_regime = "neutral"

    # Momentum & volume are stored as continuous values and kept as
    # descriptive labels for the LLM prompt, but excluded from env_key
    # to keep bins manageable (3×3×3 = 27).
    if avg_accel > 0.15:
        momentum_regime = "accelerating"
    elif avg_accel > -0.15:
        momentum_regime = "steady"
    else:
        momentum_regime = "decelerating"

    if avg_late_vol > 1.15:
        volume_regime = "expanding"
    elif avg_late_vol < 0.85:
        volume_regime = "contracting"
    else:
        volume_regime = "normal"

    snapshot = {
        "timestamp": session_date,
        "symbols_analyzed": n,
        "volatility_regime": vol_regime,
        "trend_regime": trend_regime,
        "breadth_regime": breadth_regime,
        "momentum_regime": momentum_regime,
        "volume_regime": volume_regime,
        "avg_intraday_range_pct": round(avg_intraday, 4),
        "advance_decline_ratio": round(advance_decline, 4),
        "dispersion": round(dispersion, 4),
        "avg_momentum_shift": round(avg_accel, 4),
        "avg_volume_ratio": round(avg_late_vol, 4),
        "pct_trending_up": round(pct_up, 2),
        "pct_trending_down": round(pct_down, 2),
        "strategy_fit": _compute_strategy_fit(
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            breadth_regime=breadth_regime,
            momentum_regime=momentum_regime,
            dispersion=dispersion,
            avg_atr=avg_intraday,
            avg_intraday=avg_intraday,
        ),
    }
    snapshot["env_key"] = make_environment_key(snapshot)
    return snapshot


def compute_environment_by_date(universe: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, Any]]:
    """Compute one same-day environment snapshot per eval date across the universe."""
    date_map: dict[str, dict[str, pd.DataFrame]] = {}
    for sym, days in universe.items():
        for day_str, df in days.items():
            date_map.setdefault(day_str, {})[sym] = df

    return {
        day_str: compute_session_environment(day_universe, day_str)
        for day_str, day_universe in date_map.items()
    }


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
        "daily_returns": returns.tolist(),  # for cross-asset correlation
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

    # 3-level volatility (low / normal / high) — keeps total bins at 3×3×3 = 27
    if avg_atr < 1.5:
        vol_regime = "low"
    elif avg_atr < 3.0:
        vol_regime = "normal"
    else:
        vol_regime = "high"

    # ── Trend regime (breadth-based) ────────────────────────────
    trends = [f["trend_strength"] for f in sym_features.values()]
    cum_returns = [f["cumulative_return"] for f in sym_features.values()]

    up_trending = sum(1 for t in trends if t > 0.3)
    down_trending = sum(1 for t in trends if t < -0.3)
    sideways = n - up_trending - down_trending

    pct_up = up_trending / n
    pct_down = down_trending / n

    # 3-level trend (down / flat / up)
    if pct_down > 0.45:
        trend_regime = "down"
    elif pct_up > 0.45:
        trend_regime = "up"
    else:
        trend_regime = "flat"

    # Regime confidence: distance from the 0.45 decision boundary, 0→1.
    if trend_regime == "down":
        trend_confidence = min(1.0, (pct_down - 0.45) / 0.30)
    elif trend_regime == "up":
        trend_confidence = min(1.0, (pct_up - 0.45) / 0.30)
    else:
        trend_confidence = min(1.0, (0.45 - max(pct_up, pct_down)) / 0.20)
    trend_confidence = max(0.0, trend_confidence)

    # ── Cross-asset correlation ─────────────────────────────────
    # High = macro/systematic regime; low = idiosyncratic/stock-picking.
    cross_corr = 0.5  # fallback
    if len(cum_returns) > 3:
        all_daily: list[list[float]] = []
        for f in sym_features.values():
            dr = f.get("daily_returns")
            if dr is not None and len(dr) > 2:
                all_daily.append(dr)
        if len(all_daily) >= 3:
            min_len = min(len(d) for d in all_daily)
            if min_len >= 3:
                mat = np.array([d[:min_len] for d in all_daily])
                corr_matrix = np.corrcoef(mat)
                n_sym = corr_matrix.shape[0]
                off_diag = []
                for ci in range(n_sym):
                    for cj in range(ci + 1, n_sym):
                        v = corr_matrix[ci, cj]
                        if np.isfinite(v):
                            off_diag.append(v)
                if off_diag:
                    cross_corr = float(np.mean(off_diag))
    advance_decline = (up_trending - down_trending) / n if n > 0 else 0

    # 3-level breadth (bearish / neutral / bullish)
    if advance_decline > 0.2:
        breadth_regime = "bullish"
    elif advance_decline < -0.2:
        breadth_regime = "bearish"
    else:
        breadth_regime = "neutral"

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

        # Regime confidence & correlation
        "trend_confidence": round(trend_confidence, 4),
        "cross_asset_correlation": round(cross_corr, 4),

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
        f"Environment: vol={vol_regime} trend={trend_regime}({trend_confidence:.0%}) "
        f"breadth={breadth_regime} momentum={momentum_regime} volume={volume_regime} "
        f"dispersion={dispersion:.2f} cross_corr={cross_corr:.2f}"
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
    # Use continuous avg_atr for granularity within the 3-level labels
    momentum_score = 0.5
    if trend_regime in ("up", "down"):
        momentum_score += 0.25
    elif trend_regime == "flat":
        momentum_score -= 0.2
    if momentum_regime == "accelerating":
        momentum_score += 0.15
    if avg_atr > 2.5:
        momentum_score += 0.1
    elif avg_atr < 1.0:
        momentum_score -= 0.15
    scores["momentum_breakout"] = max(0, min(1, momentum_score))

    # ── Mean Reversion strategies ───────────────────────────────
    # Favor: flat markets, moderate vol, wide intraday range
    reversion_score = 0.5
    if trend_regime == "flat":
        reversion_score += 0.25
    elif trend_regime in ("up", "down"):
        reversion_score -= 0.15
    if 1.0 <= avg_atr <= 2.5:
        reversion_score += 0.15
    elif avg_atr > 3.5:
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
    elif trend_regime == "flat":
        vwap_score += 0.1
    scores["vwap"] = max(0, min(1, vwap_score))

    # ── Options — Long calls/puts (directional) ────────────────
    # Favor: trending, moderate vol (cheap enough to buy, enough movement)
    long_options_score = 0.5
    if trend_regime in ("up", "down"):
        long_options_score += 0.2
    if 1.5 <= avg_atr <= 3.0:
        long_options_score += 0.1  # affordable IV + movement
    elif avg_atr > 3.5:
        long_options_score -= 0.1  # IV too expensive
    scores["long_options"] = max(0, min(1, long_options_score))

    # ── Options — Short premium (strangles, iron condors) ──────
    # Favor: flat + elevated vol (sell into IV), contracting momentum
    short_premium_score = 0.5
    if trend_regime == "flat":
        short_premium_score += 0.25
    elif trend_regime in ("up", "down"):
        short_premium_score -= 0.15
    if avg_atr > 2.0:
        short_premium_score += 0.2  # rich premium to sell
    elif avg_atr < 1.0:
        short_premium_score -= 0.15  # not enough premium
    if momentum_regime == "decelerating":
        short_premium_score += 0.1
    scores["short_premium"] = max(0, min(1, short_premium_score))

    # ── Options — Straddle (volatility play) ────────────────────
    # Favor: low vol (cheap) with expected expansion, high dispersion
    straddle_score = 0.5
    if avg_atr < 1.5:
        straddle_score += 0.2  # cheap entry
    elif avg_atr > 3.5:
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
    if vol_regime in ("normal", "high"):
        spread_score += 0.15
    scores["vertical_spread"] = max(0, min(1, spread_score))

    # ── Bracket / range-bound ───────────────────────────────────
    # Similar to mean reversion but with tighter parameters
    bracket_score = 0.5
    if trend_regime == "flat":
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

    # Regime confidence & correlation (may not be present in older snapshots)
    trend_conf = snapshot.get("trend_confidence", 0.0)
    cross_corr = snapshot.get("cross_asset_correlation", 0.5)
    corr_label = "macro/systematic" if cross_corr > 0.6 else "idiosyncratic" if cross_corr < 0.3 else "mixed"

    return f"""CURRENT MARKET ENVIRONMENT (computed from {snapshot['symbols_analyzed']} symbols, last {snapshot.get('num_days', '?')} trading days):

  CONTINUOUS METRICS (use these for nuanced strategy design, not just the labels):
    ATR:             {snapshot['avg_atr_pct']:.2f}%  [low <1.0 | normal 1-2 | high 2-3.5 | extreme >3.5]
    Intraday range:  {snapshot['avg_intraday_range_pct']:.2f}%
    Cumulative ret:  {snapshot.get('avg_cumulative_return', 0):.2f}%
    A/D ratio:       {snapshot['advance_decline_ratio']:+.2f}  [bearish <-0.3 | mixed -0.3-0 | neutral 0-0.3 | bullish >0.3]
    Momentum shift:  {snapshot['avg_momentum_shift']:+.2f}  [decel <-0.5 | steady -0.5-0.5 | accel >0.5]
    Volume ratio:    {snapshot['avg_volume_ratio']:.2f}x  [contracting <0.8 | normal 0.8-1.3 | expanding >1.3]
    % trending up:   {snapshot['pct_trending_up']:.0%}
    % trending down: {snapshot['pct_trending_down']:.0%}
    Dispersion:      {snapshot['dispersion']:.2f}
    Trend confidence:{trend_conf:.0%}  [0%=borderline | 100%=decisive regime]
    Cross-asset corr:{cross_corr:.2f}  [{corr_label}: <0.3=stock-picking | >0.6=macro-driven]

  REGIME LABELS (summary of the above):
    Volatility={snapshot['volatility_regime']}  Trend={snapshot['trend_regime']}  Breadth={snapshot['breadth_regime']}
    Momentum={snapshot['momentum_regime']}  Volume={snapshot['volume_regime']}

STRATEGY-ENVIRONMENT FIT (higher = better match for current conditions):
{chr(10).join(fit_lines)}

TOP SYMBOL CANDIDATES:
  Momentum:       {', '.join(candidates)}
  Mean reversion: {', '.join(rev_candidates)}
  Options:        {', '.join(opt_candidates)}"""
