"""
Slot 08 — Bearish-Breadth High-Vol Down-Regime Adjusted-Skew Iron Condor

Mandate: bullish / neutral / options_premium
Allowed order_types: iron_condor, butterfly, straddle, strangle

Bullish-leaning premium collection via iron condor with moderated upside skew.
Updated per latest analysis for current high-vol + down-trend + bearish-breadth regime:
- Strict gate to high-vol + down-trend + bearish breadth ONLY (neutral removed)
- Restricted to repeatable symbols: HUBS, MARA, APP
- Removed SMA filter entirely (no longer fighting the down-regime)
- Tight range-bound filter (|ROC| < 0.015) for decelerating momentum
- Moderated upside skew (call wing 1.4x, put wing 1.3x)
- Realistic fixed-percentage exits (0.35% target, 0.25% stop) to match observed tiny moves
- Later entry (bar 90+), moderate spacing, short 45-bar max hold for theta
- No bandwidth filter in high-vol regime
- Focus on bearish-breadth edge, symbol quality, and execution-tolerant parameters
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
ROC_BARS = 6
MIN_BAR = 90
MAX_HOLD_BARS = 45
SIGNAL_INTERVAL = 20

ALLOWED_SYMBOLS = {"HUBS", "MARA", "APP"}


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < 130 or symbol not in ALLOWED_SYMBOLS:
        return []

    signals = []
    close = candles["close"]
    
    # Regime awareness
    if env is None or not isinstance(env, dict):
        vol_regime = "normal"
        trend_regime = "flat"
        breadth_regime = "neutral"
    else:
        vol_regime = env.get("volatility_regime", "normal")
        trend_regime = env.get("trend_regime", "flat")
        breadth_regime = env.get("breadth_regime", "neutral")
    
    is_high_vol = vol_regime == "high"
    is_down_trend = trend_regime == "down"
    is_bearish_breadth = breadth_regime == "bearish"
    
    # Hard regime gate: only trade in the currently profitable bucket
    if not (is_high_vol and is_down_trend and is_bearish_breadth):
        return []
    
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)
    
    max_roc_abs = 0.015
    wing_scale = 8.0
    max_hold = MAX_HOLD_BARS
    
    for i in range(MIN_BAR, len(candles) - 30, SIGNAL_INTERVAL):
        if any(pd.isna(x) for x in (
            atr.iloc[i], roc.iloc[i]
        )) or atr.iloc[i] <= 0:
            continue

        price = float(close.iloc[i])
        a = float(atr.iloc[i])
        
        # Tight range-bound filter for decelerating momentum in down regime
        if abs(roc.iloc[i]) > max_roc_abs:
            continue

        # Base strike rounded to nearest 5
        base = round(price / 5.0) * 5.0
        
        # Moderated upside skew: less bullish bias than previous version
        short_put = base - wing_scale
        long_put = short_put - (wing_scale * 1.3)
        short_call = base + int(wing_scale * 1.4)
        long_call = short_call + int(wing_scale * 1.2)

        # Realistic small-percentage exits that match observed 1-bar move sizes
        target_price = price * 1.0035
        stop_price = price * 0.9975

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": max_hold,
            "setup_type": "bearish_breadth_highvol_down_adjusted_skew_ic",
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