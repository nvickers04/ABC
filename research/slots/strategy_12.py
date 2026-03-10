"""
Slot 12 — Bracket Order Mean Reversion

Enters on RSI oversold + VWAP support with bracket order
(simultaneous stop and target). Catches intraday pullbacks
in otherwise strong names.
"""

import pandas as pd
import numpy as np

RSI_PERIOD = 14
RSI_THRESHOLD = 35
ATR_PERIOD = 14
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 2.5
MAX_HOLD_BARS = 30
MIN_BAR = 60
VWAP_TOLERANCE = 0.003


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    min_bars = max(RSI_PERIOD, ATR_PERIOD, MIN_BAR) + 20
    if len(candles) < min_bars + 1:
        return []

    signals = []
    rsi = _rsi(candles["close"], RSI_PERIOD)

    cum_vol = candles["volume"].cumsum()
    cum_vp = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_vp / cum_vol.replace(0, 1)

    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close),
        ),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    start = max(RSI_PERIOD, ATR_PERIOD, MIN_BAR) + 10
    for i in range(start, len(candles) - 1):
        row = candles.iloc[i]
        if pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(vwap.iloc[i]):
            continue

        cur_atr = atr.iloc[i]
        cur_vwap = vwap.iloc[i]
        price = row["close"]

        if (
            rsi.iloc[i] < RSI_THRESHOLD
            and abs(price - cur_vwap) / cur_vwap < VWAP_TOLERANCE
            and cur_atr > 0
        ):
            entry = candles.iloc[i + 1]["open"]
            stop = entry - cur_atr * STOP_ATR_MULT
            target = entry + cur_atr * TARGET_ATR_MULT

            signals.append({
                "entry_bar": i + 1,
                "direction": "long",
                "order_type": "bracket",
                "entry_price": round(entry, 2),
                "target_price": round(target, 2),
                "stop_price": round(stop, 2),
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "mean_reversion_rsi",
                "legs_json": None,
            })

    return signals
