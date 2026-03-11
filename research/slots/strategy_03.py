"""
Slot 03 — Short-Biased VWAP Rejection (ATR-Scaled, Trend-Filtered)

Evolved for current regime: extreme volatility (ATR ~5.85%), downtrend,
decelerating momentum, neutral breadth. Prioritizes short_premium (0.80)
and momentum_breakout (0.75) alignment.

Core logic:
- VWAP cross-down rejections (price crosses below VWAP on elevated volume)
- Strict trend filter: shorts ONLY when below EMA20, longs ONLY when above
- ATR(14) scaled stops/targets to survive 5.35% intraday ranges (no more 0.4% chops)
- Volume confirmation at 1.8x 20-bar average
- Morning/midday filter only (before bar 210 ≈ 13:00 ET)
- Reduced max hold to 20 bars for faster regime
- Short bias via EMA filter naturally suppresses counter-trend longs
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 20
MIN_BAR = 50
VOL_MULT = 1.8
EMA_SPAN = 20
ATR_PERIOD = 14
STOP_ATR_MULT = 0.85
TARGET_ATR_MULT = 2.1
MAX_ENTRY_BAR = 210


def calculate_atr(candles: pd.DataFrame, period: int) -> pd.Series:
    high_low = candles["high"] - candles["low"]
    high_close = np.abs(candles["high"] - candles["close"].shift(1))
    low_close = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + 30:
        return []

    signals = []

    # VWAP
    cum_vol = candles["volume"].cumsum()
    cum_pv = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    # EMA20 for trend filter
    ema = candles["close"].ewm(span=EMA_SPAN, adjust=False).mean()

    # Volume average
    vol_avg = candles["volume"].rolling(20).mean()

    # ATR
    atr = calculate_atr(candles, ATR_PERIOD)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(vwap.iloc[i]) or pd.isna(ema.iloc[i]) or
            pd.isna(vol_avg.iloc[i]) or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = candles.iloc[i]["close"]
        prev_price = candles.iloc[i - 1]["close"]
        v = vwap.iloc[i]
        e = ema.iloc[i]
        a = atr.iloc[i]
        vol = candles.iloc[i]["volume"]
        vol_threshold = vol_avg.iloc[i] * VOL_MULT

        # Short rejection: cross below VWAP on high volume while below EMA (trend-aligned)
        if (prev_price > v and price < v and
            vol > vol_threshold and price < e):
            entry = price
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "vwap",
                "entry_price": entry,
                "target_price": entry - TARGET_ATR_MULT * a,
                "stop_price": entry + STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "vwap_rejection_trend_short",
                "legs_json": None,
            })

        # Long bounce: cross above VWAP on high volume while above EMA (rare in downtrend)
        elif (prev_price < v and price > v and
              vol > vol_threshold and price > e):
            entry = price
            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "vwap",
                "entry_price": entry,
                "target_price": entry + TARGET_ATR_MULT * a,
                "stop_price": entry - STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "vwap_bounce_trend_aligned",
                "legs_json": None,
            })

    return signals