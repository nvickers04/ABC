"""
Seed Strategy — Market on Close end-of-day entries.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Enter at market on close based on intraday momentum patterns.
MOC order fills at the last bar's close price. Used for closing imbalance
and end-of-day momentum patterns.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
MOMENTUM_LOOKBACK = 60  # last 60 bars (~1 hour) for momentum check
MIN_MOMENTUM = 0.003    # minimum 0.3% upward move in last hour
STOP_PCT = 0.008        # 0.8% stop (for next-day risk)
TARGET_PCT = 0.015      # 1.5% target
MAX_HOLD_BARS = 390     # full next day (won't matter much for MOC)
VOLUME_RAMP = 1.3       # last-hour volume must exceed day average


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for Market-on-Close momentum entries.

    Logic:
    - Look at last 60 bars of trading day
    - If strong upward momentum in final hour + volume ramping
    - Enter long at close (MOC) — capturing closing imbalance
    """
    if len(candles) < MOMENTUM_LOOKBACK + 30:
        return []

    signals = []

    last_bar_idx = len(candles) - 1
    last_bar = candles.iloc[last_bar_idx]

    # Look at the last 60 bars
    lookback_start = max(0, last_bar_idx - MOMENTUM_LOOKBACK)
    lookback_slice = candles.iloc[lookback_start:last_bar_idx + 1]

    if len(lookback_slice) < MOMENTUM_LOOKBACK // 2:
        return []

    # Momentum: price change over the last hour
    start_price = lookback_slice.iloc[0]["close"]
    end_price = lookback_slice.iloc[-1]["close"]
    momentum = (end_price - start_price) / start_price

    if momentum < MIN_MOMENTUM:
        return []

    # Volume ramp: last-hour avg volume vs all-day avg volume
    day_avg_vol = candles["volume"].mean()
    last_hour_avg_vol = lookback_slice["volume"].mean()
    if day_avg_vol > 0 and last_hour_avg_vol < day_avg_vol * VOLUME_RAMP:
        return []

    entry_price = last_bar["close"]
    stop_price = entry_price * (1 - STOP_PCT)
    target_price = entry_price * (1 + TARGET_PCT)

    signals.append({
        "entry_bar": last_bar_idx,
        "direction": "long",
        "order_type": "moc",
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "max_hold_bars": MAX_HOLD_BARS,
        "setup_type": "closing_momentum_moc",
        "legs_json": None,
    })

    return signals
