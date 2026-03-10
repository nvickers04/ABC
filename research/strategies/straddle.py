"""
Seed Strategy — Long straddle volatility expansion entries.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Concept: Buy a straddle when volatility is compressed (squeeze), expecting
a breakout in either direction. Profits from large moves.
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
ATR_PERIOD = 14         # ATR for volatility measurement
BB_PERIOD = 20          # Bollinger Band period
BB_STD = 2.0            # Bollinger Band std dev
SQUEEZE_THRESHOLD = 0.6 # BB width / ATR ratio below this = squeeze
MAX_HOLD_BARS = 60      # max hold (want quick vol expansion)
MIN_BAR = 30            # skip first 30 min


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan for straddle entries on volatility squeeze.

    Logic:
    - Detect Bollinger Band squeeze (bands narrow relative to ATR)
    - Enter long straddle (ATM call + ATM put)
    - Profits if price moves significantly in either direction
    - Time decay is the cost — need fast breakout
    """
    min_needed = max(ATR_PERIOD, BB_PERIOD, MIN_BAR) + 10
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

    # Bollinger Bands
    sma = candles["close"].rolling(BB_PERIOD).mean()
    std = candles["close"].rolling(BB_PERIOD).std()
    bb_upper = sma + BB_STD * std
    bb_lower = sma - BB_STD * std
    bb_width = (bb_upper - bb_lower) / sma  # normalized width

    start_idx = max(ATR_PERIOD, BB_PERIOD, MIN_BAR) + 5

    for i in range(start_idx, len(candles) - 1):
        row = candles.iloc[i]
        current_atr = atr.iloc[i]
        current_bb_width = bb_width.iloc[i]

        if pd.isna(current_atr) or pd.isna(current_bb_width):
            continue
        if current_atr <= 0 or row["close"] <= 0:
            continue

        # Squeeze detection: BB width is narrow relative to recent ATR
        atr_pct = current_atr / row["close"]
        if atr_pct <= 0:
            continue
        squeeze_ratio = current_bb_width / atr_pct
        if squeeze_ratio > SQUEEZE_THRESHOLD:
            continue  # not squeezed enough

        # Volume picking up (early sign of breakout)
        if i >= 20:
            avg_vol = candles["volume"].iloc[i - 20:i].mean()
            if row["volume"] < avg_vol * 1.2:
                continue

        entry_price = row["close"]
        strike = round(entry_price)  # ATM strike

        # Straddle: stop is premium loss, target is large move
        premium_approx = entry_price * 0.04  # ~4% premium for ATM straddle
        stop_price = entry_price * 0.985   # ~1.5% adverse = straddle bleeds
        target_price = entry_price * 1.025  # ~2.5% favorable = straddle profits

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "straddle",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "volatility_squeeze_straddle",
            "legs_json": {
                "strategy": "straddle",
                "expiration": "20260315",
                "strike": strike,
            },
        })

    return signals
