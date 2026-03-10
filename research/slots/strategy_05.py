"""
Slot 05 — Bull Call Vertical Spread on Early Filtered Breakout

Breakout above 20-bar high in first 90 minutes with:
- volume > 1.8x average
- price above both 10 and 50 SMA (counters decelerating momentum)
- strong bar close (>=60% of range)
- minimum 0.15 ATR penetration
- positive MACD histogram
Uses defined-risk bull call vertical spread (long ~ATM, +5 short) to match
0.60 vertical_spread fit, reduce theta drag in extreme vol, and capture
quick 1.25:0.75 ATR moves. 35-bar max hold focuses on morning momentum
in strong_up / bullish breadth regime. Realistic targets for decelerating
momentum environment.
"""

import pandas as pd
import numpy as np


LOOKBACK = 20
ATR_PERIOD = 14
VOLUME_MULT = 1.8
MIN_BAR = 30
MAX_HOLD_BARS = 35
TREND_PERIOD = 50
SHORT_TREND = 10
TARGET_ATR_MULT = 1.25
STOP_ATR_MULT = 0.75
RANGE_THRESH = 0.60
PENETRATION_ATR = 0.15
SPREAD_WIDTH = 5.0


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    min_bars = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, SHORT_TREND, MIN_BAR) + 20
    if len(candles) < min_bars + 1:
        return []

    signals = []
    
    # Core indicators
    rolling_high = candles["high"].rolling(LOOKBACK).max()
    vol_avg = candles["volume"].rolling(LOOKBACK).mean().shift(1)
    sma50 = candles["close"].rolling(TREND_PERIOD).mean()
    sma10 = candles["close"].rolling(SHORT_TREND).mean()
    
    # ATR
    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(np.abs(candles["high"] - prev_close), np.abs(candles["low"] - prev_close)),
    )
    atr = tr.rolling(ATR_PERIOD).mean()
    
    # MACD histogram to filter decelerating momentum
    ema12 = candles["close"].ewm(span=12, adjust=False).mean()
    ema26 = candles["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    start = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, SHORT_TREND, MIN_BAR) + 10
    for i in range(start, len(candles) - 1):
        row = candles.iloc[i]
        if pd.isna(rolling_high.iloc[i - 1]) or pd.isna(atr.iloc[i]) or pd.isna(sma50.iloc[i]) or pd.isna(sma10.iloc[i]) or pd.isna(macd_hist.iloc[i]):
            continue
        
        # Time-of-day filter: first 90 minutes only (strongest momentum window)
        if i > 89:
            continue
        
        # Breakout
        if row["close"] <= rolling_high.iloc[i - 1]:
            continue
        
        # Minimum penetration
        if (row["close"] - rolling_high.iloc[i - 1]) < (PENETRATION_ATR * atr.iloc[i]):
            continue
        
        # Volume confirmation
        if row["volume"] < vol_avg.iloc[i] * VOLUME_MULT:
            continue
        
        # Trend filter (above both SMAs)
        if row["close"] <= sma50.iloc[i] or row["close"] <= sma10.iloc[i]:
            continue
        
        # Momentum filter (positive and rising MACD hist)
        if macd_hist.iloc[i] <= 0 or macd_hist.iloc[i] <= macd_hist.iloc[i - 1]:
            continue
        
        # Strong bar close
        rng = row["high"] - row["low"]
        if rng <= 0 or (row["close"] - row["low"]) / rng < RANGE_THRESH:
            continue
        
        entry = row["close"]
        cur_atr = atr.iloc[i]
        
        # Strikes for bull call spread (~ATM long leg)
        long_strike = round(entry)
        short_strike = long_strike + SPREAD_WIDTH
        
        target_price = entry + cur_atr * TARGET_ATR_MULT
        stop_price = entry - cur_atr * STOP_ATR_MULT
        
        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "vertical_spread",
            "entry_price": entry,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "bull_call_vertical_momentum",
            "legs_json": {
                "strategy": "vertical_spread",
                "expiration": "nearest_weekly",
                "long_strike": float(long_strike),
                "short_strike": float(short_strike),
                "right": "C",
            },
        })
    return signals