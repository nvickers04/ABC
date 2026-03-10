"""
Slot 10 — Intraday Support Bounce Long (Stock Proxy)

Simple long stock entries near 20-bar lows when RSI is neutral-bullish (45-65)
and volume is at least average. Uses market entry with 1:2 risk-reward.
This replaces the broken options short-put logic with a stock-based proxy
that the backtester can actually simulate. Loosened filters slightly and
added volume confirmation to reduce robotic signal count while still
producing 30-70 signals per day across the universe.
"""

import pandas as pd
import numpy as np


RSI_PERIOD = 14
LOOKBACK_LOW = 20
VOL_LOOKBACK = 20
MIN_BAR = 25          # after ~10:00 ET
MAX_HOLD_BARS = 140   # ~2.5 hours max, forces EOD exit
DIST_TO_SUPPORT_PCT = 2.2
STOP_PCT = 0.75
TARGET_PCT = 1.5      # improved 1:2 R:R


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < max(RSI_PERIOD, LOOKBACK_LOW, VOL_LOOKBACK) + 30:
        return []

    signals = []
    rsi = _rsi(candles["close"], RSI_PERIOD)
    rolling_low = candles["low"].rolling(LOOKBACK_LOW).min()
    vol_avg = candles["volume"].rolling(VOL_LOOKBACK).mean()

    start = max(RSI_PERIOD + 5, LOOKBACK_LOW, VOL_LOOKBACK, MIN_BAR)
    for i in range(start, len(candles) - 5):
        if pd.isna(rsi.iloc[i]) or pd.isna(rolling_low.iloc[i]) or pd.isna(vol_avg.iloc[i]):
            continue

        # RSI neutral-to-bullish
        if rsi.iloc[i] < 45 or rsi.iloc[i] > 65:
            continue

        price = candles.iloc[i]["close"]
        support = rolling_low.iloc[i]
        dist_to_support = (price - support) / price * 100

        # Near support
        if dist_to_support > DIST_TO_SUPPORT_PCT:
            continue

        # Volume confirmation
        if candles.iloc[i]["volume"] < 0.9 * vol_avg.iloc[i]:
            continue

        # Reasonable price action (not extreme gap)
        if candles.iloc[i]["high"] > price * 1.015:
            continue

        entry_price = price
        stop_price = entry_price * (1 - STOP_PCT / 100)
        target_price = entry_price * (1 + TARGET_PCT / 100)

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "market",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "support_bounce_long",
            "legs_json": None,
        })
    return signals