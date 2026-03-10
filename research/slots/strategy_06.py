"""
Slot 06 — Momentum Breakout Call Debit Spread

Bullish vertical call spread on break above 20-bar high in strong_up regime.
Requires volume confirmation (>1.8x avg), price above SMA(50), SMA(20)>SMA(50)
for trend alignment, and short-term positive momentum (close > close[5]).
Uses realistic intraday ATR multiples (target 1.0 ATR, stop 0.6 ATR) given
extreme volatility and ~30-minute typical hold window. Focused on morning
momentum in high-fit momentum_breakout + long_options environment.
Strikes rounded to nearest 5 for realistic option chaining.
"""

import pandas as pd
import numpy as np

LOOKBACK = 20
ATR_PERIOD = 14
VOLUME_MULT = 1.8
MIN_BAR = 30
MAX_HOLD_BARS = 25
TREND_PERIOD = 50
FAST_MA = 20
TARGET_ATR_MULT = 1.0
STOP_ATR_MULT = 0.6


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    min_bars = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, FAST_MA, MIN_BAR) + 10
    if len(candles) < min_bars + 1:
        return []

    signals = []
    rolling_high = candles["high"].rolling(LOOKBACK).max()
    vol_avg = candles["volume"].rolling(LOOKBACK).mean().shift(1)
    sma = candles["close"].rolling(TREND_PERIOD).mean()
    sma_fast = candles["close"].rolling(FAST_MA).mean()

    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(np.abs(candles["high"] - prev_close), np.abs(candles["low"] - prev_close)),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, FAST_MA, MIN_BAR) + 5
    for i in range(start, len(candles) - 1):
        row = candles.iloc[i]
        if (pd.isna(rolling_high.iloc[i - 1]) or pd.isna(vol_avg.iloc[i]) or
            pd.isna(sma.iloc[i]) or pd.isna(sma_fast.iloc[i]) or pd.isna(atr.iloc[i])):
            continue

        # Momentum breakout conditions aligned with strong_up environment
        if row["close"] <= rolling_high.iloc[i - 1]:
            continue
        if row["volume"] <= vol_avg.iloc[i] * VOLUME_MULT:
            continue
        if row["close"] <= sma.iloc[i]:
            continue
        if sma_fast.iloc[i] <= sma.iloc[i]:
            continue
        # Short-term momentum filter (rising into the breakout)
        if i < 5 or row["close"] <= candles["close"].iloc[i - 5]:
            continue

        entry = row["close"]
        cur_atr = atr.iloc[i]

        # ATM-ish strikes rounded to nearest 5 for realistic option liquidity
        base_strike = round(entry / 5.0) * 5.0
        long_strike = base_strike
        short_strike = base_strike + 5.0

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "vertical_spread",
            "entry_price": entry,
            "target_price": entry + cur_atr * TARGET_ATR_MULT,
            "stop_price": entry - cur_atr * STOP_ATR_MULT,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "momentum_breakout_call_spread",
            "legs_json": {
                "strategy": "vertical_spread",
                "expiration": "nearest_weekly",
                "long_strike": float(long_strike),
                "short_strike": float(short_strike),
                "right": "C",
            },
        })
    return signals