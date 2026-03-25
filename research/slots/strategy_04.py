"""
Slot 04 — Bearish-Skew Iron Condor (High-Vol Downtrend Adaptation)

Mandate: bearish / neutral / options_premium
Allowed order_types: butterfly, iron_condor, straddle, strangle

Adapted to CURRENT environment: Volatility=high (ATR 5.12%), Trend=down (76% trending down),
Breadth=bearish (A/D -0.64), Momentum=decelerating (-1.10), Volume=normal (0.97x).
Short-premium fit is 0.65. Staying strictly inside mandate.

Core logic:
- Price below SMA50 (respects decisive bearish backdrop)
- One-sided momentum filter: allow downward drift but block strong upside moves (roc < +2.5%)
- Realized range filter vs ATR (enter when not excessively expanded — better for premium)
- Iron condor with clear downside skew: wider buffer on put side (more room for bearish drift)
- Strikes are percentage-based then rounded to realistic option increments (avoids fixed $5 issues)
- Entry after open settles (bar 45+), moderate spacing for signal count without over-trading
- Long hold for theta decay, forced exit near close
- Target/stop scaled to realized volatility (wider in high-vol regime)
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 50
ROC_BARS = 5
RANGE_BARS = 25
MIN_BAR = 45
MAX_HOLD_BARS = 240
MAX_ROC_UP = 0.025
SIGNAL_INTERVAL = 18


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


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + ATR_PERIOD + SMA_PERIOD + 30:
        return []

    signals = []
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    sma50 = close.rolling(SMA_PERIOD).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)

    for i in range(MIN_BAR, len(candles) - 35, SIGNAL_INTERVAL):
        if (pd.isna(sma50.iloc[i]) or pd.isna(atr.iloc[i]) or 
            pd.isna(roc.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = float(close.iloc[i])
        a = float(atr.iloc[i])

        # Bearish context
        if price >= sma50.iloc[i]:
            continue

        # Allow bearish momentum, block strong upside only (inverted from prior range-bound logic)
        if roc.iloc[i] > MAX_ROC_UP:
            continue

        # Realized expansion filter — prefer periods that are not already too stretched (premium-friendly)
        start = max(0, i - RANGE_BARS)
        recent_range = float(high.iloc[start:i+1].max() - low.iloc[start:i+1].min())
        # Approximate expected range under random walk (ATR scaled by sqrt(time))
        expected_range = a * np.sqrt(RANGE_BARS) * 1.8
        if recent_range > expected_range:
            continue

        # Percentage-based strikes with downside skew (more room on put side)
        call_short_dist = 0.027 * price
        put_short_dist = 0.048 * price   # wider buffer for bearish regime
        wing_width = 0.019 * price

        short_call = _round_strike(price + call_short_dist)
        long_call = _round_strike(short_call + wing_width)
        short_put = _round_strike(price - put_short_dist)
        long_put = _round_strike(short_put - wing_width)

        # Avoid degenerate strikes
        if short_call <= price or short_put >= price or long_call <= short_call or long_put >= short_put:
            continue

        # Volatility-scaled exit levels (wider in high-vol regime)
        vol_move = 1.4 * a * np.sqrt(25)   # scaled move for ~25 bars
        target_price = price - 0.3 * vol_move
        stop_price = price + 2.2 * vol_move

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": price,
            "target_price": float(target_price),
            "stop_price": float(stop_price),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "bearish_skew_iron_condor_atr",
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