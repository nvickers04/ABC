"""
Slot 08 — Straddle on Volatility Squeeze

Long straddle when Bollinger bandwidth contracts below threshold,
anticipating a volatility expansion. Both ATM call and put.
"""

import pandas as pd
import numpy as np

BB_PERIOD = 20
BB_STD = 2.0
SQUEEZE_THRESHOLD = 0.02
ATR_PERIOD = 14
MIN_BAR = 40
MAX_HOLD_BARS = 60
STOP_PCT = 3.0
TARGET_PCT = 5.0


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < BB_PERIOD + ATR_PERIOD + 20:
        return []

    signals = []
    sma = candles["close"].rolling(BB_PERIOD).mean()
    std = candles["close"].rolling(BB_PERIOD).std()
    bandwidth = (std * BB_STD * 2) / sma

    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(np.abs(candles["high"] - prev_close), np.abs(candles["low"] - prev_close)),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start = max(BB_PERIOD, ATR_PERIOD, MIN_BAR) + 5
    for i in range(start, len(candles) - 1):
        if pd.isna(bandwidth.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        if bandwidth.iloc[i] < SQUEEZE_THRESHOLD:
            entry = candles.iloc[i]["close"]
            strike = round(entry)
            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "straddle",
                "entry_price": entry,
                "target_price": entry * (1 + TARGET_PCT / 100),
                "stop_price": entry * (1 - STOP_PCT / 100),
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "volatility_squeeze_straddle",
                "legs_json": {
                    "strategy": "straddle",
                    "expiration": "nearest_weekly",
                    "strike": strike,
                },
            })
    return signals
