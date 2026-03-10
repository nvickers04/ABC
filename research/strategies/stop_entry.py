"""
Seed Strategy — Stop entry breakout triggers.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Place a buy-stop above resistance. The order triggers only if
price breaks through the level, confirming the breakout before entry.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
LOOKBACK = 20           # bars to define resistance
ATR_PERIOD = 14         # ATR for stops/targets
STOP_ATR_MULT = 1.5     # stop below breakout level
TARGET_ATR_MULT = 3.0   # target above entry
MAX_HOLD_BARS = 45      # max hold
MIN_BAR = 30            # skip first 30 min
BUFFER_PCT = 0.001      # place stop-entry 0.1% above resistance


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for stop-entry breakout setups.

    Logic:
    - Find the highest high over the lookback window
    - Place a buy-stop just above that resistance level
    - Enters only if price actually breaks through (simulator handles the trigger)
    - Volume must be above average
    """
    min_needed = max(LOOKBACK, ATR_PERIOD, MIN_BAR) + 10
    if len(candles) < min_needed + 1:
        return []

    signals = []

    rolling_high = candles["high"].rolling(LOOKBACK).max()
    vol_avg = candles["volume"].rolling(LOOKBACK).mean().shift(1)

    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close),
        ),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start_idx = max(LOOKBACK, ATR_PERIOD, MIN_BAR) + 5

    for i in range(start_idx, len(candles) - 1):
        row = candles.iloc[i]
        resistance = rolling_high.iloc[i - 1]
        current_atr = atr.iloc[i]
        avg_v = vol_avg.iloc[i]

        if pd.isna(resistance) or pd.isna(current_atr) or pd.isna(avg_v):
            continue
        if current_atr <= 0:
            continue

        # Price must be near but below resistance (coiling)
        if row["close"] >= resistance:
            continue  # already broken out
        if row["close"] < resistance * 0.995:
            continue  # too far from resistance

        # Volume building
        if row["volume"] < avg_v * 1.2:
            continue

        entry_price = resistance * (1 + BUFFER_PCT)
        stop_price = entry_price - STOP_ATR_MULT * current_atr
        target_price = entry_price + TARGET_ATR_MULT * current_atr

        if stop_price >= entry_price or target_price <= entry_price:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "stop_entry",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "resistance_breakout_stop",
            "legs_json": None,
        })

    return signals
