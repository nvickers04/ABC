"""
Slot 03 — VWAP Bounce (VWAP Order)

Mean-reversion entry when price dips to VWAP and bounces with volume.
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 30
MIN_BAR = 40
STOP_PCT = 0.4
TARGET_PCT = 0.8
VOL_MULT = 1.5


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + 10:
        return []

    signals = []
    cum_vol = candles["volume"].cumsum()
    cum_pv = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    vol_avg = candles["volume"].rolling(20).mean()

    for i in range(MIN_BAR, len(candles) - 1):
        if pd.isna(vwap.iloc[i]) or pd.isna(vol_avg.iloc[i]):
            continue

        price = candles.iloc[i]["close"]
        prev_price = candles.iloc[i - 1]["close"]
        v = vwap.iloc[i]

        # Price crossed below VWAP on prior bar, now bouncing above
        if prev_price < v and price > v and candles.iloc[i]["volume"] > vol_avg.iloc[i] * VOL_MULT:
            entry = price
            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "vwap",
                "entry_price": entry,
                "target_price": entry * (1 + TARGET_PCT / 100),
                "stop_price": entry * (1 - STOP_PCT / 100),
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "vwap_bounce",
                "legs_json": None,
            })
    return signals
