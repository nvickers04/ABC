"""
Slot 09 — Bear Call Vertical Spread in Downtrend + Extreme Vol

Optimized for current regime: extreme volatility (ATR 6.1%), downtrend
(53% down days), high short_premium fit (0.80) and vertical_spread fit (0.70).
Triggers on options/mean-reversion candidates. Requires price below 20-bar SMA
for bearish bias. Short call strike placed 0.75 ATR above price with 7-point wing.
Widened stop (5.0%) and realistic target (2.0%) to match ATR regime and reduce
gamma-driven gaps. Signals thinned to every 22 bars after 60-minute mark to keep
signal count realistic and avoid over-fitting. Builds on #106 (exp=5.3362) by
lowering MIN_BAR, widening allowed symbols to current top candidates, and
increasing stop buffer to survive 6.1% ATR environment.
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 20
MIN_BAR = 60
MAX_HOLD_BARS = 145
SHORT_STRIKE_ATR_MULT = 0.75
WING_WIDTH = 7.0
STOP_PCT = 5.0
TARGET_PCT = 2.0
SIGNAL_INTERVAL = 22

ALLOWED_SYMBOLS = {"MARA", "U", "CELH", "ZS", "APP", "HUBS", "PINS", "CAVA"}


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if symbol not in ALLOWED_SYMBOLS:
        return []
    if len(candles) < ATR_PERIOD + SMA_PERIOD + 50:
        return []

    signals = []
    close = candles["close"]
    sma = close.rolling(SMA_PERIOD).mean()

    prev_close = close.shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close)
        )
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    for i in range(MIN_BAR, len(candles) - 15, SIGNAL_INTERVAL):
        if pd.isna(sma.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        if close.iloc[i] >= sma.iloc[i]:
            continue  # bearish filter only

        entry = close.iloc[i]
        atr_val = atr.iloc[i]
        short_strike = round(entry + atr_val * SHORT_STRIKE_ATR_MULT)
        long_strike = short_strike + WING_WIDTH

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "vertical_spread",
            "entry_price": entry,
            "target_price": entry * (1 - TARGET_PCT / 100),
            "stop_price": entry * (1 + STOP_PCT / 100),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "bear_call_vertical",
            "legs_json": {
                "strategy": "vertical_spread",
                "expiration": "nearest_weekly",
                "long_strike": long_strike,
                "short_strike": short_strike,
                "right": "C"
            },
        })
    return signals