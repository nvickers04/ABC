"""
Slot 04 — Regime-Adaptive Bearish-Skew Iron Condor (High-Vol Downtrend)

Mandate: bearish / neutral / options_premium
Allowed order_types: butterfly, iron_condor, straddle, strangle

Adapted to CURRENT environment: Volatility=high (ATR 5.15%), Trend=down,
Breadth=bearish (A/D -0.56), Momentum=decelerating (-2.22), Volume=normal.
Short-premium fit 0.65. Uses env to adapt skew, ROC cap, and risk based on
breadth_regime (tighter upside in bearish breadth to avoid neutral-chop failures).

Core logic:
- Price below SMA50 respects bearish backdrop
- Adaptive one-sided momentum filter (stricter upside cap when breadth=bearish)
- Realized-range vs ATR filter (stricter 1.5× multiplier for premium safety)
- Downside-skewed iron condor: tighter call wing + wider put wing when bearish
- Strikes are percentage-based then rounded to realistic increments
- Earlier entry window (bar 30+) + moderate spacing for sufficient signal count
- Tighter risk: reduced max hold, smaller vol-scaled stop (addresses loose-stop losers)
- Target has mild downward bias consistent with bearish mandate and current regime
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 50
ROC_BARS = 5
RANGE_BARS = 25
MIN_BAR = 30
MAX_HOLD_BARS = 120
MAX_ROC_UP_BASE = 0.025
SIGNAL_INTERVAL = 15


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _round_strike(price: float) -> float:
    """Realistic option strike rounding based on price level."""
    if price < 25:
        return round(price / 0.5) * 0.5
    elif price < 60:
        return round(price / 1.0) * 1.0
    else:
        return round(price / 5.0) * 5.0


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < MIN_BAR + ATR_PERIOD + SMA_PERIOD + 30:
        return []

    signals = []
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    sma50 = close.rolling(SMA_PERIOD).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)

    # Regime adaptation from env (fallback to current bearish context)
    breadth_regime = env.get("breadth_regime", "bearish") if env is not None else "bearish"
    is_bearish_breadth = breadth_regime == "bearish"
    max_roc_up = 0.015 if is_bearish_breadth else MAX_ROC_UP_BASE
    call_mult = 0.018 if is_bearish_breadth else 0.027
    put_mult = 0.052 if is_bearish_breadth else 0.040

    for i in range(MIN_BAR, len(candles) - 35, SIGNAL_INTERVAL):
        if (pd.isna(sma50.iloc[i]) or pd.isna(atr.iloc[i]) or 
            pd.isna(roc.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = float(close.iloc[i])
        a = float(atr.iloc[i])

        # Bearish context
        if price >= sma50.iloc[i]:
            continue

        # Adaptive momentum filter: block strong upside (stricter when bearish)
        if roc.iloc[i] > max_roc_up:
            continue

        # Realized range filter — prefer periods not excessively stretched
        start = max(0, i - RANGE_BARS)
        recent_range = float(high.iloc[start:i+1].max() - low.iloc[start:i+1].min())
        expected_range = a * np.sqrt(RANGE_BARS) * 1.5
        if recent_range > expected_range:
            continue

        # Percentage-based strikes with adaptive downside skew
        call_short_dist = call_mult * price
        put_short_dist = put_mult * price
        wing_width = 0.019 * price

        short_call = _round_strike(price + call_short_dist)
        long_call = _round_strike(short_call + wing_width)
        short_put = _round_strike(price - put_short_dist)
        long_put = _round_strike(short_put - wing_width)

        # Avoid degenerate strikes
        if short_call <= price or short_put >= price or long_call <= short_call or long_put >= short_put:
            continue

        # Tighter volatility-scaled exits (addresses previous loose-stop problem)
        vol_move = 1.0 * a * np.sqrt(20)
        target_price = price - 0.4 * vol_move
        stop_price = price + 1.15 * vol_move

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": price,
            "target_price": float(target_price),
            "stop_price": float(stop_price),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "regime_adaptive_bearish_skew_iron_condor",
            "legs_json": {
                "strategy": "iron_condor",
                "expiration": "nearest_weekly",
                "put_long_strike": float(long_put),
                "put_short_strike": float(short_put),
                "call_short_strike": float(short_call),
                "call_long_strike": float(long_call)
            },
        })

    return signals