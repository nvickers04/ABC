"""
Slot 09 — VWAP Momentum Short (Downtrend Aligned)

Current regime: high vol (ATR 5.51%), strong downtrend (68% down days),
neutral breadth, accelerating momentum (+0.61). Uses highest-fit
momentum_breakout (1.00) and vwap (0.80). Shorts only on top momentum
candidates when price is below both SMA20 and VWAP with clearly negative
5-bar momentum. ATR-based risk (stop 1.4×ATR, target 2.8×ATR) calibrated
for 5.5% ATR regime. Bracket order for clean execution. MIN_BAR=75 and
25-bar spacing avoids opening noise while producing evaluable signal count.
No options — eliminates gamma gaps for better live execution tolerance.
Builds on #771 by widening target:stop, shifting entry later, tightening
symbol list to current top momentum names, and increasing max hold slightly.

Focus: simple, interpretable trend-following shorts that respect the
actual down-trending + accelerating regime.
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 20
MOM_PERIOD = 5
MIN_BAR = 75
MAX_HOLD_BARS = 160
ATR_STOP_MULT = 1.40
ATR_TARGET_MULT = 2.80
SIGNAL_INTERVAL = 25

ALLOWED_SYMBOLS = {"MARA", "CAVA", "DUOL", "PLTR", "NET"}


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if symbol not in ALLOWED_SYMBOLS:
        return []
    if len(candles) < ATR_PERIOD + SMA_PERIOD + 60:
        return []

    signals = []
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    sma = close.rolling(SMA_PERIOD).mean()
    tp = (high + low + close) / 3.0
    vwap = (tp * volume).cumsum() / volume.cumsum()

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()

    mom = close - close.shift(MOM_PERIOD)

    for i in range(MIN_BAR, len(candles) - 20, SIGNAL_INTERVAL):
        if (pd.isna(sma.iloc[i]) or pd.isna(vwap.iloc[i]) or
            pd.isna(atr.iloc[i]) or pd.isna(mom.iloc[i])):
            continue
        if atr.iloc[i] <= 0:
            continue

        if (close.iloc[i] >= sma.iloc[i] or
            close.iloc[i] >= vwap.iloc[i] or
            mom.iloc[i] >= 0):
            continue

        entry = float(close.iloc[i])
        atr_val = float(atr.iloc[i])

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "bracket",
            "entry_price": entry,
            "target_price": entry - atr_val * ATR_TARGET_MULT,
            "stop_price": entry + atr_val * ATR_STOP_MULT,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "vwap_momentum_short",
            "legs_json": None,
        })

    return signals