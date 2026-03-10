"""
Slot 06 — Long Puts on Breakdown

Buys ATM put option when price breaks below 20-bar low with volume
and trend confirmation (below SMA).
"""

import pandas as pd
import numpy as np

LOOKBACK = 20
ATR_PERIOD = 14
VOLUME_MULT = 2.0
MIN_BAR = 45
MAX_HOLD_BARS = 60
TREND_PERIOD = 50


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    min_bars = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 10
    if len(candles) < min_bars + 1:
        return []

    signals = []
    rolling_low = candles["low"].rolling(LOOKBACK).min()
    vol_avg = candles["volume"].rolling(LOOKBACK).mean().shift(1)
    sma = candles["close"].rolling(TREND_PERIOD).mean()

    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(np.abs(candles["high"] - prev_close), np.abs(candles["low"] - prev_close)),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 5
    for i in range(start, len(candles) - 1):
        row = candles.iloc[i]
        if pd.isna(rolling_low.iloc[i - 1]) or pd.isna(atr.iloc[i]) or pd.isna(sma.iloc[i]):
            continue
        if row["close"] >= rolling_low.iloc[i - 1]:
            continue
        if row["volume"] < vol_avg.iloc[i] * VOLUME_MULT:
            continue
        if row["close"] >= sma.iloc[i]:
            continue

        entry = row["close"]
        cur_atr = atr.iloc[i]
        strike = round(entry)
        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "vertical_spread",
            "entry_price": entry,
            "target_price": entry - cur_atr * 3.0,
            "stop_price": entry + cur_atr * 1.5,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "long_put_breakdown",
            "legs_json": {
                "strategy": "long_put",
                "expiration": "nearest_weekly",
                "long_strike": strike,
                "right": "P",
            },
        })
    return signals
