"""
Slot 04 — Regime-Gated Bearish-Skew Iron Condor (High-Vol Downtrend)

Mandate: bearish / neutral / options_premium
Allowed order_types: butterfly, iron_condor, straddle, strangle

Adapted to CURRENT environment: Volatility=high (ATR 5.15%), Trend=down,
Breadth=bearish (A/D -0.56), Momentum=decelerating (-2.22), Volume=normal.
Short-premium fit 0.65. Hard-gates on bearish breadth + down trend,
requires mild negative ROC, dynamic high-vol skew, tightened stops/hold.

Core logic:
- Hard regime gate for bearish breadth + down trend (removes mismatched-regime noise)
- Price below SMA50 respects bearish backdrop
- Momentum filter requires mild bearish drift (roc < -0.005) + upside cap
- Realized-range vs ATR filter (loosened multiplier for signal count in high vol)
- Dynamic downside-skewed iron condor: tighter call wing / wider put wing in high ATR
- Strikes percentage-based then rounded to realistic increments
- Moderate entry window (bar 30+) + SIGNAL_INTERVAL=10 for sufficient signals
- Tighter volatility-scaled exits (stop 0.75×, target 0.55×) for high-vol regime
- Reduced max hold (60 bars) to capture theta faster and limit large losers
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 50
ROC_BARS = 5
RANGE_BARS = 25
MIN_BAR = 30
MAX_HOLD_BARS = 60
MAX_ROC_UP_BASE = 0.025
SIGNAL_INTERVAL = 10


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

    # Hard regime gate - only trade in the proven high-expectancy bucket
    if env is not None:
        breadth_regime = env.get("breadth_regime", "")
        trend_regime = env.get("trend_regime", "")
        if breadth_regime != "bearish" or trend_regime != "down":
            return []

    signals = []
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    sma50 = close.rolling(SMA_PERIOD).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)

    breadth_regime = env.get("breadth_regime", "bearish") if env is not None else "bearish"
    is_bearish_breadth = breadth_regime == "bearish"
    max_roc_up = 0.015 if is_bearish_breadth else MAX_ROC_UP_BASE

    for i in range(MIN_BAR, len(candles) - 35, SIGNAL_INTERVAL):
        if (pd.isna(sma50.iloc[i]) or pd.isna(atr.iloc[i]) or 
            pd.isna(roc.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = float(close.iloc[i])
        a = float(atr.iloc[i])

        # Bearish context
        if price >= sma50.iloc[i]:
            continue

        # Momentum filter: require mild bearish drift (per analysis) + upside cap
        if roc.iloc[i] > max_roc_up or roc.iloc[i] > -0.005:
            continue

        # Realized range filter — prefer periods not excessively stretched (slightly loosened)
        start = max(0, i - RANGE_BARS)
        recent_range = float(high.iloc[start:i+1].max() - low.iloc[start:i+1].min())
        expected_range = a * np.sqrt(RANGE_BARS) * 1.8
        if recent_range > expected_range:
            continue

        # Dynamic skew based on realized volatility (tighter call wing in high vol)
        vol_pct = a / price if price > 0 else 0
        call_mult = 0.015 if vol_pct > 0.04 else 0.018
        put_mult = 0.058 if vol_pct > 0.04 else 0.052
        wing_width = 0.019 * price

        short_call = _round_strike(price + call_mult * price)
        long_call = _round_strike(short_call + wing_width)
        short_put = _round_strike(price - put_mult * price)
        long_put = _round_strike(short_put - wing_width)

        # Avoid degenerate strikes
        if (short_call <= price or short_put >= price or 
            long_call <= short_call or long_put >= short_put):
            continue

        # Tighter volatility-scaled exits for high-ATR regime (addresses loose-stop losers)
        vol_move = 1.0 * a * np.sqrt(20)
        target_price = price - 0.55 * vol_move
        stop_price = price + 0.75 * vol_move

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": price,
            "target_price": float(target_price),
            "stop_price": float(stop_price),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "regime_gated_bearish_skew_iron_condor",
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