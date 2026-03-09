"""
Seed Strategy — the initial strategy the research agent evolves from.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Each dict must contain:
  entry_bar, direction, order_type, entry_price, target_price,
  stop_price, max_hold_bars, setup_type, legs_json (optional)

Seed: 20-bar high breakout with volume confirmation.
"""

import pandas as pd
import numpy as np

# ── Parameters ──────────────────────────────────────────────────
LOOKBACK = 20           # bars to look back for high/low
VOLUME_MULT = 2.0       # volume must be Nx the rolling average
STOP_PCT = 0.008        # 0.8% stop loss
TARGET_PCT = 0.016      # 1.6% target (2:1 R:R)
MAX_HOLD_BARS = 60      # 60 min max hold
MIN_BAR = 30            # skip first 30 min (9:30-10:00 chop)


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan 1-min candles for breakout entries with volume confirmation.

    Strategy logic:
    - Wait until after the first 30 minutes (let the open settle)
    - Look for price closing above the 20-bar high
    - Require volume at least 2x the 20-bar average
    - Enter at market on the next bar
    - Stop at 0.8% below entry, target at 1.6% above (2:1 R:R)
    """
    if len(candles) < LOOKBACK + MIN_BAR + 1:
        return []

    signals = []

    # Pre-compute rolling metrics
    rolling_high = candles["high"].rolling(LOOKBACK).max()
    rolling_vol_avg = candles["volume"].rolling(LOOKBACK).mean()

    for i in range(LOOKBACK + MIN_BAR, len(candles) - 1):
        row = candles.iloc[i]
        prev_high = rolling_high.iloc[i - 1]
        vol_avg = rolling_vol_avg.iloc[i]

        # Breakout: close above prior 20-bar high
        if row["close"] > prev_high and vol_avg > 0 and row["volume"] > vol_avg * VOLUME_MULT:
            entry_price = row["close"]
            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "market",
                "entry_price": entry_price,
                "target_price": entry_price * (1 + TARGET_PCT),
                "stop_price": entry_price * (1 - STOP_PCT),
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "breakout_volume",
                "legs_json": None,
            })

    return signals
