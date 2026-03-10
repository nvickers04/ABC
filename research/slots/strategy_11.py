"""
Slot 11 — Trailing Stop Breakout (Trailing Stop Exit)

Enters on 20-bar high breakout with volume, uses trailing stop
to let winners run in trending environments.
"""

import pandas as pd
import numpy as np

LOOKBACK = 20
ATR_PERIOD = 14
TRAIL_ATR_MULT = 2.0
TARGET_ATR_MULT = 5.0
MAX_HOLD_BARS = 60
MIN_BAR = 60
TREND_PERIOD = 50
VOLUME_MULT = 1.5


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    min_bars = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 20
    if len(candles) < min_bars + 1:
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
    sma_trend = candles["close"].rolling(TREND_PERIOD).mean()

    start = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 10
    for i in range(start, len(candles) - 1):
        row = candles.iloc[i]
        if (
            pd.isna(rolling_high.iloc[i - 1])
            or pd.isna(vol_avg.iloc[i])
            or pd.isna(atr.iloc[i])
            or pd.isna(sma_trend.iloc[i])
        ):
            continue

        prev_high = rolling_high.iloc[i - 1]
        cur_atr = atr.iloc[i]

        if (
            row["close"] > prev_high
            and row["close"] > sma_trend.iloc[i]
            and row["volume"] > vol_avg.iloc[i] * VOLUME_MULT
            and cur_atr > 0
        ):
            entry = candles.iloc[i + 1]["open"]
            stop = entry - cur_atr * TRAIL_ATR_MULT
            target = entry + cur_atr * TARGET_ATR_MULT

            signals.append({
                "entry_bar": i + 1,
                "direction": "long",
                "order_type": "trailing_stop_exit",
                "entry_price": round(entry, 2),
                "target_price": round(target, 2),
                "stop_price": round(stop, 2),
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "trailing_breakout",
                "legs_json": None,
            })

    return signals
