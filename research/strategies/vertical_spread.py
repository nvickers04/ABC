"""
Seed Strategy — Vertical spread directional entries.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Defined-risk directional trade using a bull call vertical spread.
Enter on strong momentum breakout, use spread to cap risk.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
LOOKBACK = 20           # bars for breakout detection
ATR_PERIOD = 14         # ATR for momentum filter
MIN_BAR = 30            # skip first 30 min
SPREAD_WIDTH = 5.0      # $5 wide spread
MAX_HOLD_BARS = 60      # max hold
VOLUME_MULT = 2.0       # volume filter


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for vertical spread entries on momentum breakouts.

    Logic:
    - Detect strong upward breakout with volume
    - Enter bull call vertical (long ATM call, short OTM call)
    - Defined risk = spread width - credit received (approximated)
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
        prev_high = rolling_high.iloc[i - 1]
        current_atr = atr.iloc[i]
        avg_v = vol_avg.iloc[i]

        if pd.isna(prev_high) or pd.isna(current_atr) or pd.isna(avg_v):
            continue
        if current_atr <= 0:
            continue

        # Breakout + volume
        if row["close"] <= prev_high:
            continue
        if row["volume"] < avg_v * VOLUME_MULT:
            continue

        entry_price = row["close"]
        # ATM long call, OTM short call
        long_strike = round(entry_price)  # nearest dollar
        short_strike = long_strike + SPREAD_WIDTH

        # Approximate: stop = lose full debit, target = spread fills up
        debit_approx = SPREAD_WIDTH * 0.4  # ~40% of width for ATM
        stop_price = entry_price - current_atr * 2  # underlying stop
        target_price = entry_price + current_atr * 3  # underlying target

        if stop_price >= entry_price or target_price <= entry_price:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "vertical_spread",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "bull_call_vertical_breakout",
            "legs_json": {
                "strategy": "vertical_spread",
                "expiration": "20260315",  # near-term expiration
                "long_strike": long_strike,
                "short_strike": short_strike,
                "right": "C",
            },
        })

    return signals
