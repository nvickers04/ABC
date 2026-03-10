"""
Seed Strategy — Bracket order entries with built-in risk management.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Enter on range breakout with bracket order — entry + stop + target
all submitted as one order. Tight risk on both sides.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
RANGE_BARS = 15         # bars to define consolidation range
ATR_PERIOD = 14         # ATR for risk sizing
STOP_ATR_MULT = 1.0     # tight stop (bracket = precise)
TARGET_ATR_MULT = 2.5   # target
MAX_HOLD_BARS = 30      # shorter hold — bracket manages exits
MIN_BAR = 30            # skip first 30 min
RANGE_WIDTH_MAX = 0.008 # max range width as % of price (tight = coiling)


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for bracket order setups on tight range breakouts.

    Logic:
    - Identify tight consolidation ranges (low volatility squeeze)
    - When price breaks out of range on volume, enter with bracket
    - Bracket auto-manages stop and target
    """
    min_needed = max(RANGE_BARS, ATR_PERIOD, MIN_BAR) + 10
    if len(candles) < min_needed + 1:
        return []

    signals = []

    rolling_high = candles["high"].rolling(RANGE_BARS).max()
    rolling_low = candles["low"].rolling(RANGE_BARS).min()
    vol_avg = candles["volume"].rolling(20).mean().shift(1)

    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close),
        ),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start_idx = max(RANGE_BARS, ATR_PERIOD, MIN_BAR, 20) + 5

    for i in range(start_idx, len(candles) - 1):
        row = candles.iloc[i]
        range_hi = rolling_high.iloc[i - 1]
        range_lo = rolling_low.iloc[i - 1]
        current_atr = atr.iloc[i]
        avg_v = vol_avg.iloc[i]

        if pd.isna(range_hi) or pd.isna(range_lo) or pd.isna(current_atr) or pd.isna(avg_v):
            continue
        if current_atr <= 0 or row["close"] <= 0:
            continue

        # Range must be tight (consolidation squeeze)
        range_width = (range_hi - range_lo) / row["close"]
        if range_width > RANGE_WIDTH_MAX or range_width <= 0:
            continue

        # Breakout above range on volume
        if row["close"] <= range_hi:
            continue
        if row["volume"] < avg_v * 1.5:
            continue

        entry_price = row["close"]
        stop_price = entry_price - STOP_ATR_MULT * current_atr
        target_price = entry_price + TARGET_ATR_MULT * current_atr

        if stop_price >= entry_price or target_price <= entry_price:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "bracket",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "range_squeeze_bracket",
            "legs_json": None,
        })

    return signals
