"""
Slot 11 — Neutral Iron Condor (Choppy regime)

Mandate: choppy / neutral / options_premium
Allowed order_types: iron_condor, butterfly, straddle, strangle

Centred premium collection via symmetric iron condor for range-bound
markets. No directional skew — pure theta decay play.

Core logic:
- Price near SMA50 (within 1 ATR — centered, not trending)
- Low directional momentum (abs 5-bar ROC < 1.0%)
- Bollinger bandwidth below median (compressed = range-bound ideal)
- Symmetric iron condor: equal-width wings on both sides
- Entry after opening settles (bar 60+)
- Long hold (rest of session) for theta decay
- 35-bar signal spacing
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 50
BB_PERIOD = 20
BB_STD = 2.0
ROC_BARS = 5
MIN_BAR = 60
MAX_HOLD_BARS = 220
MAX_ROC_ABS = 0.010
SIGNAL_INTERVAL = 35
SMA_PROXIMITY_ATR = 1.0
WING_WIDTH = 7.5


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + SMA_PERIOD + BB_PERIOD + 10:
        return []

    signals = []
    close = candles["close"]
    sma50 = close.rolling(SMA_PERIOD).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)

    sma_bb = close.rolling(BB_PERIOD).mean()
    std_bb = close.rolling(BB_PERIOD).std()
    bandwidth = (std_bb * BB_STD * 2) / sma_bb
    bw_median = bandwidth.rolling(60, min_periods=30).quantile(0.50)

    for i in range(MIN_BAR, len(candles) - 20, SIGNAL_INTERVAL):
        if (pd.isna(sma50.iloc[i]) or pd.isna(atr.iloc[i]) or
            pd.isna(roc.iloc[i]) or pd.isna(bandwidth.iloc[i]) or
            pd.isna(bw_median.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = close.iloc[i]
        a = atr.iloc[i]

        # Centred near SMA50
        if abs(price - sma50.iloc[i]) > SMA_PROXIMITY_ATR * a:
            continue
        # No momentum — range-bound
        if abs(roc.iloc[i]) > MAX_ROC_ABS:
            continue
        # Compressed bandwidth
        if bandwidth.iloc[i] > bw_median.iloc[i]:
            continue

        entry = float(price)

        # Symmetric iron condor
        base = round(entry / 5.0) * 5.0
        short_put = base - 5.0
        long_put = short_put - WING_WIDTH
        short_call = base + 5.0
        long_call = short_call + WING_WIDTH

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": entry,
            "target_price": entry,
            "stop_price": entry + 1.5 * a,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "neutral_iron_condor",
            "legs_json": {
                "strategy": "iron_condor",
                "expiration": "nearest_weekly",
                "long_put": float(long_put),
                "short_put": float(short_put),
                "short_call": float(short_call),
                "long_call": float(long_call),
                "right": "both",
            },
        })

    return signals