"""
Seed Strategy — Iron condor premium selling.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Sell iron condor on range-bound, low-volatility names. Collect
premium when price stays between short strikes. Profits from time decay.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
ATR_PERIOD = 14         # ATR for sizing strikes
RANGE_BARS = 60         # bars to check for range-bound behavior
RANGE_MAX_PCT = 0.008   # max price range as % (tight = range-bound)
MAX_HOLD_BARS = 120     # hold longer — theta is your friend
MIN_BAR = 60            # wait for range to establish
WING_WIDTH = 5.0        # $5 wings
STRIKE_OFFSET_ATR = 2.0 # short strikes at 2x ATR from current price


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for iron condor entries on range-bound names.

    Logic:
    - Price has been in a tight range for the last 60 bars
    - Low ATR relative to price (not volatile)
    - Sell iron condor: short put + short call inside the range,
      long wings outside for protection
    - Profits if price stays in range (theta decay)
    """
    min_needed = max(ATR_PERIOD, RANGE_BARS, MIN_BAR) + 10
    if len(candles) < min_needed + 1:
        return []

    signals = []

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

    # Rolling range
    rolling_high = candles["high"].rolling(RANGE_BARS).max()
    rolling_low = candles["low"].rolling(RANGE_BARS).min()

    start_idx = max(ATR_PERIOD, RANGE_BARS, MIN_BAR) + 5

    for i in range(start_idx, len(candles) - 1):
        row = candles.iloc[i]
        current_atr = atr.iloc[i]
        range_hi = rolling_high.iloc[i]
        range_lo = rolling_low.iloc[i]

        if pd.isna(current_atr) or pd.isna(range_hi) or pd.isna(range_lo):
            continue
        if current_atr <= 0 or row["close"] <= 0:
            continue

        # Range-bound check
        range_pct = (range_hi - range_lo) / row["close"]
        if range_pct > RANGE_MAX_PCT:
            continue  # too volatile for iron condor

        # Price should be near middle of range
        mid_range = (range_hi + range_lo) / 2
        if abs(row["close"] - mid_range) > (range_hi - range_lo) * 0.4:
            continue  # too close to edge

        entry_price = row["close"]
        offset = current_atr * STRIKE_OFFSET_ATR

        put_short = round(entry_price - offset)
        call_short = round(entry_price + offset)
        put_long = put_short - WING_WIDTH
        call_long = call_short + WING_WIDTH

        # For iron condor: stop = breach of short strike, target = theta decay
        stop_price = entry_price - offset * 1.2  # underlying breaches
        target_price = entry_price  # stays flat = profit

        signals.append({
            "entry_bar": i,
            "direction": "long",  # "long" the iron condor position
            "order_type": "iron_condor",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "range_bound_iron_condor",
            "legs_json": {
                "strategy": "iron_condor",
                "expiration": "20260315",
                "put_short_strike": put_short,
                "put_long_strike": put_long,
                "call_short_strike": call_short,
                "call_long_strike": call_long,
                "wing_width": WING_WIDTH,
            },
        })

    return signals
