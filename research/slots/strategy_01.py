"""
Slot 01 — Momentum Breakout (Market Order)

ATR-adaptive 20-bar breakout with volume confirmation, trend filter, and
volatility filter. Enters via market order on the next bar's open.
"""

import pandas as pd
import numpy as np

LOOKBACK = 20
VOLUME_MULT = 2.5
ATR_PERIOD = 14
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 3.0
MAX_HOLD_BARS = 40
MIN_BAR = 60
TREND_PERIOD = 50
VOL_FILTER_LOW = 0.001
VOL_FILTER_HIGH = 0.005


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    min_bars = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 20
    if len(candles) < min_bars + 1:
        return []

    signals = []
    rolling_high = candles["high"].rolling(LOOKBACK).max()
    vol_avg_long = candles["volume"].rolling(LOOKBACK).mean().shift(1)
    vol_avg_short = candles["volume"].rolling(5).mean().shift(1)

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
    vol_ratio = atr / candles["close"]

    start = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 10
    for i in range(start, len(candles) - 1):
        row = candles.iloc[i]
        if (
            pd.isna(rolling_high.iloc[i - 1])
            or pd.isna(vol_avg_long.iloc[i])
            or pd.isna(vol_avg_short.iloc[i])
            or pd.isna(atr.iloc[i])
            or pd.isna(sma_trend.iloc[i])
            or pd.isna(vol_ratio.iloc[i])
        ):
            continue

        prev_high = rolling_high.iloc[i - 1]
        cur_atr = atr.iloc[i]
        cur_vr = vol_ratio.iloc[i]

        if row["close"] <= prev_high:
            continue
        if row["volume"] < max(vol_avg_long.iloc[i], vol_avg_short.iloc[i]) * VOLUME_MULT:
            continue
        if row["close"] <= sma_trend.iloc[i]:
            continue
        if cur_vr < VOL_FILTER_LOW or cur_vr > VOL_FILTER_HIGH:
            continue

        entry = row["close"]
        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "market",
            "entry_price": entry,
            "target_price": entry + cur_atr * TARGET_ATR_MULT,
            "stop_price": entry - cur_atr * STOP_ATR_MULT,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "momentum_breakout",
            "legs_json": None,
        })
    return signals
