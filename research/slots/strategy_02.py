"""
Slot 02 — Mean Reversion (Limit Order)

RSI oversold bounce at intraday support with limit entry below recent low.
"""

import pandas as pd
import numpy as np

RSI_PERIOD = 14
RSI_OVERSOLD = 30
LOOKBACK_LOW = 20
LIMIT_OFFSET_PCT = 0.1
STOP_PCT = 0.5
TARGET_PCT = 1.0
MAX_HOLD_BARS = 30
MIN_BAR = 30


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < RSI_PERIOD + LOOKBACK_LOW + 10:
        return []

    signals = []
    rsi = _rsi(candles["close"], RSI_PERIOD)
    rolling_low = candles["low"].rolling(LOOKBACK_LOW).min()

    for i in range(max(RSI_PERIOD + 5, LOOKBACK_LOW, MIN_BAR), len(candles) - 1):
        if pd.isna(rsi.iloc[i]) or pd.isna(rolling_low.iloc[i]):
            continue
        if rsi.iloc[i] > RSI_OVERSOLD:
            continue

        support = rolling_low.iloc[i]
        entry = support * (1 - LIMIT_OFFSET_PCT / 100)
        if candles.iloc[i]["close"] > support * 1.02:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "limit",
            "entry_price": entry,
            "target_price": entry * (1 + TARGET_PCT / 100),
            "stop_price": entry * (1 - STOP_PCT / 100),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "mean_reversion_rsi",
            "legs_json": None,
        })
    return signals
