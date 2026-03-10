"""
Seed Strategy — Limit order pullback entries.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Identify uptrending stocks pulling back to EMA support.
Place a limit buy below current price at the EMA level.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
EMA_FAST = 9            # fast EMA for pullback level
EMA_SLOW = 21           # slow EMA for trend filter
ATR_PERIOD = 14         # ATR for stops/targets
STOP_ATR_MULT = 1.5     # stop below EMA support
TARGET_ATR_MULT = 3.0   # target above entry
MAX_HOLD_BARS = 50      # max hold before forced exit
MIN_BAR = 30            # skip first 30 min
PULLBACK_THRESHOLD = 0.002  # price must be within 0.2% of fast EMA


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for limit order pullback entries.

    Logic:
    - Stock must be in uptrend (fast EMA > slow EMA)
    - Price pulls back toward fast EMA (within threshold)
    - Place limit buy at fast EMA level
    - Stop below recent swing low, target at 2:1 R:R
    """
    if len(candles) < EMA_SLOW + ATR_PERIOD + MIN_BAR:
        return []

    signals = []

    ema_fast = candles["close"].ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = candles["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close),
        ),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start_idx = max(EMA_SLOW, ATR_PERIOD, MIN_BAR) + 5

    for i in range(start_idx, len(candles) - 1):
        row = candles.iloc[i]
        fast_val = ema_fast.iloc[i]
        slow_val = ema_slow.iloc[i]
        current_atr = atr.iloc[i]

        if pd.isna(fast_val) or pd.isna(slow_val) or pd.isna(current_atr):
            continue
        if current_atr <= 0:
            continue

        # Uptrend: fast EMA above slow EMA
        if fast_val <= slow_val:
            continue

        # Price is near fast EMA (pulling back to support)
        distance = (row["close"] - fast_val) / row["close"]
        if not (-PULLBACK_THRESHOLD <= distance <= PULLBACK_THRESHOLD):
            continue

        # Volume above average (some buying interest)
        if i >= 20:
            avg_vol = candles["volume"].iloc[i - 20:i].mean()
            if row["volume"] < avg_vol * 0.8:
                continue

        entry_price = fast_val  # limit at EMA support
        stop_price = entry_price - STOP_ATR_MULT * current_atr
        target_price = entry_price + TARGET_ATR_MULT * current_atr

        if stop_price >= entry_price or target_price <= entry_price:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "limit",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "ema_pullback_limit",
            "legs_json": None,
        })

    return signals
