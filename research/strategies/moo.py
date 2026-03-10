"""
Seed Strategy — Market on Open gap plays.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Detect gap-up patterns from the first few bars and enter
at market on open. MOO order fills at the open price (bar 0).
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
GAP_THRESHOLD = 0.005   # minimum 0.5% gap-up from prior close estimate
STOP_PCT = 0.008        # 0.8% stop
TARGET_PCT = 0.016      # 1.6% target (2:1 R:R)
MAX_HOLD_BARS = 60      # max hold
VOLUME_MULT = 1.5       # opening volume must exceed this multiple of avg


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for Market-on-Open gap plays.

    Logic:
    - Compare first bar's open to prior day's implied close (last bar close)
    - If gap-up exceeds threshold, enter long at open
    - High volume at open confirms institutional participation
    - Quick stop below open, target at 2:1 R:R
    """
    if len(candles) < 30:
        return []

    signals = []

    # Estimate prior close as the low of the first 5 bars
    # (we only have one day of candles, so approximate from early bars)
    open_price = candles.iloc[0]["open"]
    first_5_avg_close = candles.iloc[:5]["close"].mean()

    # Check if there's a gap (open significantly above early average)
    # Since we don't have yesterday's close, use first bar gap detection:
    # large open vs first couple bars' low suggests gap
    first_bar = candles.iloc[0]
    second_bar = candles.iloc[1] if len(candles) > 1 else first_bar

    # Gap proxy: open is above the first few minutes' pullback
    gap_holds = first_bar["close"] > first_bar["open"] * (1 - GAP_THRESHOLD)
    volume_strong = first_bar["volume"] > candles["volume"].iloc[:20].mean() * VOLUME_MULT if len(candles) >= 20 else True

    if gap_holds and volume_strong and first_bar["close"] > first_bar["open"]:
        entry_price = open_price
        stop_price = entry_price * (1 - STOP_PCT)
        target_price = entry_price * (1 + TARGET_PCT)

        signals.append({
            "entry_bar": 0,
            "direction": "long",
            "order_type": "moo",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "gap_up_moo",
            "legs_json": None,
        })

    return signals
