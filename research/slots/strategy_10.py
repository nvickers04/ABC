"""
Slot 10 — Cash-Secured Put Selling at Support

Sells OTM put at intraday support level when RSI is neutral-to-bullish.
Collects premium with defined risk.
"""

import pandas as pd
import numpy as np

RSI_PERIOD = 14
LOOKBACK_LOW = 20
MIN_BAR = 45
MAX_HOLD_BARS = 90
PUT_OTM_PCT = 1.5
STOP_PCT = 2.0
TARGET_PCT = 0.8


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < max(RSI_PERIOD, LOOKBACK_LOW) + 20:
        return []

    signals = []
    rsi = _rsi(candles["close"], RSI_PERIOD)
    rolling_low = candles["low"].rolling(LOOKBACK_LOW).min()
    vol_avg = candles["volume"].rolling(20).mean()

    start = max(RSI_PERIOD + 5, LOOKBACK_LOW, MIN_BAR)
    for i in range(start, len(candles) - 1):
        if pd.isna(rsi.iloc[i]) or pd.isna(rolling_low.iloc[i]):
            continue

        # RSI between 40-60 (neutral, not yet oversold)
        if rsi.iloc[i] < 40 or rsi.iloc[i] > 60:
            continue

        price = candles.iloc[i]["close"]
        support = rolling_low.iloc[i]
        dist_to_support = (price - support) / price * 100

        # Price near support (within 1.5%)
        if dist_to_support > 1.5:
            continue

        strike = round(price * (1 - PUT_OTM_PCT / 100))
        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "vertical_spread",
            "entry_price": price,
            "target_price": price * (1 + TARGET_PCT / 100),
            "stop_price": price * (1 - STOP_PCT / 100),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "cash_secured_put",
            "legs_json": {
                "strategy": "short_put",
                "expiration": "nearest_weekly",
                "short_strike": strike,
                "right": "P",
            },
        })
    return signals
