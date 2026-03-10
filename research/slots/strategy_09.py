"""
Slot 09 — Iron Condor on Range-Bound Names

Sells premium when price is range-bound (low ATR/price, BB squeeze).
Wings placed at 1-ATR from current price.
"""

import pandas as pd
import numpy as np

BB_PERIOD = 20
ATR_PERIOD = 14
MIN_BAR = 60
MAX_HOLD_BARS = 120
RANGE_BW_MAX = 0.025
WING_ATR_MULT = 1.5
STOP_PCT = 2.0
TARGET_PCT = 1.5


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < BB_PERIOD + ATR_PERIOD + 20:
        return []

    signals = []
    sma = candles["close"].rolling(BB_PERIOD).mean()
    std = candles["close"].rolling(BB_PERIOD).std()
    bandwidth = (std * 2 * 2) / sma

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

        if bandwidth.iloc[i] > RANGE_BW_MAX:
            continue

        entry = candles.iloc[i]["close"]
        cur_atr = atr.iloc[i]
        call_strike = round(entry + cur_atr * WING_ATR_MULT)
        put_strike = round(entry - cur_atr * WING_ATR_MULT)
        wing_width = 5.0

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "iron_condor",
            "entry_price": entry,
            "target_price": entry * (1 - TARGET_PCT / 100),
            "stop_price": entry * (1 + STOP_PCT / 100),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "range_iron_condor",
            "legs_json": {
                "strategy": "iron_condor",
                "expiration": "nearest_weekly",
                "call_short_strike": call_strike,
                "call_long_strike": call_strike + wing_width,
                "put_short_strike": put_strike,
                "put_long_strike": put_strike - wing_width,
                "wing_width": wing_width,
            },
        })
    return signals
