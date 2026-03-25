"""
Slot 03 — Bear Put Debit Spread (Bearish regime)

Mandate: bearish / short / options_directional
Allowed order_types: vertical_spread, diagonal_spread

Defined-risk bear put vertical spread on breakdown below support.
Buys ATM put, sells lower-strike put for defined risk/reward.

Core logic:
- Price below both EMA20 and EMA50 (bearish structure)
- Breakdown below 20-bar low with volume confirmation (1.5x avg)
- Negative 5-bar momentum and EMA50 sloping down
- Bar closes in lower 35% of range (decisive selling)
- Bear put vertical: long ATM put, short put 5 points lower
- ATR-based underlying targets for spread management
- Morning entries only (before bar 150)
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 30
MIN_BAR = 60
VOL_MULT = 1.5
EMA_SHORT_SPAN = 20
EMA_LONG_SPAN = 50
ATR_PERIOD = 14
LOOKBACK_BREAK = 20
MOM_BARS = 5
STOP_ATR_MULT = 0.8
TARGET_ATR_MULT = 1.8
MAX_ENTRY_BAR = 150
SPREAD_WIDTH = 5.0


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + 30:
        return []

    signals = []
    ema20 = candles["close"].ewm(span=EMA_SHORT_SPAN, adjust=False).mean()
    ema50 = candles["close"].ewm(span=EMA_LONG_SPAN, adjust=False).mean()
    vol_avg = candles["volume"].rolling(20).mean()
    atr = _atr(candles, ATR_PERIOD)
    rolling_low = candles["low"].rolling(LOOKBACK_BREAK).min()

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue
        if (pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]) or
            pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0 or
            pd.isna(vol_avg.iloc[i]) or pd.isna(rolling_low.iloc[i - 1])):
            continue
        if i < LOOKBACK_BREAK + MOM_BARS + 5:
            continue

        price = candles["close"].iloc[i]
        a = atr.iloc[i]

        # Bearish structure: below both EMAs
        if price >= ema20.iloc[i] or price >= ema50.iloc[i]:
            continue
        # EMA50 sloping down
        if ema50.iloc[i] >= ema50.iloc[i - 3]:
            continue
        # Breakdown below prior 20-bar low
        if price >= rolling_low.iloc[i - 1]:
            continue
        # Volume confirmation
        if candles["volume"].iloc[i] < vol_avg.iloc[i] * VOL_MULT:
            continue
        # Negative momentum
        if price >= candles["close"].iloc[i - MOM_BARS]:
            continue
        # Decisive selling — close in lower 35% of bar
        bar_range = candles["high"].iloc[i] - candles["low"].iloc[i]
        if bar_range <= 0:
            continue
        if (price - candles["low"].iloc[i]) / bar_range > 0.35:
            continue

        entry = price
        # Bear put vertical: long ATM put, short lower put
        long_strike = round(entry)
        short_strike = long_strike - SPREAD_WIDTH

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "vertical_spread",
            "entry_price": entry,
            "target_price": entry - TARGET_ATR_MULT * a,
            "stop_price": entry + STOP_ATR_MULT * a,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "bear_put_vertical_breakdown",
            "legs_json": {
                "strategy": "vertical_spread",
                "expiration": "nearest_weekly",
                "long_strike": float(long_strike),
                "short_strike": float(short_strike),
                "right": "P",
            },
        })

    return signals