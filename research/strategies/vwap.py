"""
Seed Strategy — VWAP algo entries.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Mean-reversion to VWAP. Buy when price pulls back below VWAP
and shows signs of returning. VWAP order type fills at approximate VWAP.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
ATR_PERIOD = 14         # ATR for stops/targets
STOP_ATR_MULT = 1.5     # stop below entry
TARGET_ATR_MULT = 2.5   # target above entry
MAX_HOLD_BARS = 40      # max hold
MIN_BAR = 45            # need enough bars for VWAP to stabilize
VWAP_DEVIATION = -0.002 # buy when price is 0.2% below VWAP
BOUNCE_BARS = 3         # confirm bounce: close > open for N of last 3 bars


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for VWAP bounce entries.

    Logic:
    - Compute cumulative VWAP from day start
    - When price pulls back below VWAP (oversold vs fair value)
    - And shows bounce (recent bars closing green)
    - Enter with VWAP order type (fills at approx VWAP over next 5 bars)
    """
    if len(candles) < MIN_BAR + ATR_PERIOD + 10:
        return []

    signals = []

    # Compute VWAP
    cum_vol = candles["volume"].cumsum()
    cum_pv = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    # ATR
    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close),
        ),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start_idx = max(MIN_BAR, ATR_PERIOD + 10)

    for i in range(start_idx, len(candles) - 6):  # need 5 bars ahead for VWAP fill
        row = candles.iloc[i]
        current_vwap = vwap.iloc[i]
        current_atr = atr.iloc[i]

        if pd.isna(current_vwap) or pd.isna(current_atr):
            continue
        if current_atr <= 0 or current_vwap <= 0:
            continue

        # Price below VWAP (pullback to fair value)
        deviation = (row["close"] - current_vwap) / current_vwap
        if deviation > VWAP_DEVIATION or deviation < -0.01:
            continue  # not below VWAP, or too far below

        # Bounce confirmation: at least 2 of last 3 bars are green
        green_count = 0
        for j in range(max(0, i - 2), i + 1):
            if candles.iloc[j]["close"] > candles.iloc[j]["open"]:
                green_count += 1
        if green_count < 2:
            continue

        entry_price = current_vwap  # VWAP fill price
        stop_price = entry_price - STOP_ATR_MULT * current_atr
        target_price = entry_price + TARGET_ATR_MULT * current_atr

        if stop_price >= entry_price or target_price <= entry_price:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "vwap",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "vwap_bounce",
            "legs_json": None,
        })

    return signals
