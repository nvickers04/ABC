"""
Slot 04 — Opening Range Breakout (Stop Entry)

First 15-min range breakout with buy-stop above the opening high.
"""

import pandas as pd
import numpy as np

OPENING_BARS = 15
STOP_PCT = 0.5
TARGET_PCT = 1.0
MAX_HOLD_BARS = 60
MIN_VOLUME_RATIO = 1.2


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < OPENING_BARS + 30:
        return []

    signals = []
    opening = candles.iloc[:OPENING_BARS]
    or_high = opening["high"].max()
    or_low = opening["low"].min()
    or_range = or_high - or_low
    if or_range <= 0:
        return []

    vol_avg = candles["volume"].rolling(10).mean()

    for i in range(OPENING_BARS, len(candles) - 1):
        if pd.isna(vol_avg.iloc[i]):
            continue
        row = candles.iloc[i]
        if row["high"] > or_high and row["volume"] > vol_avg.iloc[i] * MIN_VOLUME_RATIO:
            entry = or_high
            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "stop_entry",
                "entry_price": entry,
                "target_price": entry + or_range * 2,
                "stop_price": entry - or_range * 0.5,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "opening_range_breakout",
                "legs_json": None,
            })
            break  # one signal per day for ORB
    return signals
