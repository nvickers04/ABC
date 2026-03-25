"""
Slot 02 — Short Stock Bracket (Bearish regime)

Mandate: bearish / short / stock_bracket
Allowed order_types: bracket, trailing_stop_exit, oca_exit, limit

Short breakdown with bracket order risk management. Enters short on limit
when price rejects resistance and falls below EMA21. Bracket wraps the
entry with a defined stop-loss above the rejection high and trailing
profit target.

Core logic:
- Price must be below VWAP and EMA50 (bearish context)
- EMA21 cross-down on elevated volume (1.5x avg)
- RSI(14) < 45 confirms weakness
- Bracket order: entry at close, stop above recent swing high, target 2.0 ATR
- Morning/midday entries only (before bar 200)
"""

import pandas as pd
import numpy as np


MAX_HOLD_BARS = 45
MIN_BAR = 55
VOL_MULT = 1.5
EMA_SHORT = 21
EMA_LONG = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
STOP_ATR_MULT = 0.9
TARGET_ATR_MULT = 2.0
MAX_ENTRY_BAR = 200
SWING_LOOKBACK = 15
RSI_MAX = 45.0


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < MIN_BAR + 30:
        return []

    signals = []
    ema21 = candles["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    ema50 = candles["close"].ewm(span=EMA_LONG, adjust=False).mean()
    rsi = _rsi(candles["close"], RSI_PERIOD)
    vol_avg = candles["volume"].rolling(20).mean()
    atr = _atr(candles, ATR_PERIOD)
    cum_vol = candles["volume"].cumsum()
    cum_pv = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue
        if pd.isna(ema21.iloc[i]) or pd.isna(ema50.iloc[i]) or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        if pd.isna(rsi.iloc[i]) or pd.isna(vol_avg.iloc[i]) or pd.isna(vwap.iloc[i]):
            continue

        price = candles["close"].iloc[i]
        prev_price = candles["close"].iloc[i - 1]
        a = atr.iloc[i]

        # Bearish context: below VWAP and EMA50
        if price >= vwap.iloc[i] or price >= ema50.iloc[i]:
            continue
        # EMA21 cross-down
        if not (prev_price >= ema21.iloc[i - 1] and price < ema21.iloc[i]):
            continue
        # RSI weakness
        if rsi.iloc[i] >= RSI_MAX:
            continue
        # Volume confirmation
        if candles["volume"].iloc[i] < vol_avg.iloc[i] * VOL_MULT:
            continue

        # Bracket: stop above recent swing high
        swing_high = candles["high"].iloc[max(0, i - SWING_LOOKBACK):i + 1].max()
        stop = max(price + STOP_ATR_MULT * a, swing_high + 0.05 * a)
        target = price - TARGET_ATR_MULT * a

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "bracket",
            "entry_price": price,
            "target_price": target,
            "stop_price": stop,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "bearish_bracket_short",
            "legs_json": None,
        })

    return signals