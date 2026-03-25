"""
Slot 06 — Long Stock Momentum Breakout (Bullish regime)

Mandate: bullish / long / stock_momentum
Allowed order_types: market, vwap, stop_entry, moo

Momentum breakout long entries on trend-following triggers using VWAP
order type. Mirror of Slot 01 (bearish VWAP rejection) but for longs.

Core logic:
- VWAP cross-up reclaim on high volume (>=1.8x 20-bar avg)
- Price above both EMA20 and EMA50 (bullish structure)
- Positive 5-bar ROC (confirms momentum)
- Decisive reclaim: close at least 0.10 ATR above VWAP
- Bar closes in upper 60% of range (buying pressure)
- ATR-based stops/targets (0.5 ATR stop, 1.6 ATR target)
- Morning entries (before bar 180)
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 15
MIN_BAR = 50
VOL_MULT = 1.8
EMA_SHORT_SPAN = 20
EMA_LONG_SPAN = 50
ATR_PERIOD = 14
ROC_BARS = 5
STOP_ATR_MULT = 0.50
TARGET_ATR_MULT = 1.60
MAX_ENTRY_BAR = 180
MIN_CROSS_DEPTH_ATR = 0.10
BAR_CLOSE_MIN_PCT = 0.60


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
    cum_vol = candles["volume"].cumsum()
    cum_pv = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    ema20 = candles["close"].ewm(span=EMA_SHORT_SPAN, adjust=False).mean()
    ema50 = candles["close"].ewm(span=EMA_LONG_SPAN, adjust=False).mean()
    vol_avg = candles["volume"].rolling(20).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = candles["close"].pct_change(ROC_BARS)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue
        if (pd.isna(vwap.iloc[i]) or pd.isna(ema20.iloc[i]) or
            pd.isna(ema50.iloc[i]) or pd.isna(vol_avg.iloc[i]) or
            pd.isna(atr.iloc[i]) or pd.isna(roc.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = candles["close"].iloc[i]
        prev_price = candles["close"].iloc[i - 1]
        v = vwap.iloc[i]
        a = atr.iloc[i]

        # VWAP cross-up reclaim
        if not (prev_price < v and price > v):
            continue
        # Decisive displacement above VWAP
        if (price - v) < (MIN_CROSS_DEPTH_ATR * a):
            continue
        # Bullish structure: above both EMAs
        if price < ema20.iloc[i] or price < ema50.iloc[i]:
            continue
        # Positive momentum
        if roc.iloc[i] <= 0:
            continue
        # Volume confirmation
        if candles["volume"].iloc[i] < vol_avg.iloc[i] * VOL_MULT:
            continue
        # Strong bar close
        bar_range = candles["high"].iloc[i] - candles["low"].iloc[i]
        if bar_range <= 0:
            continue
        close_pct = (price - candles["low"].iloc[i]) / bar_range
        if close_pct < BAR_CLOSE_MIN_PCT:
            continue

        entry = price
        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "vwap",
            "entry_price": entry,
            "target_price": entry + TARGET_ATR_MULT * a,
            "stop_price": entry - STOP_ATR_MULT * a,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "vwap_reclaim_momentum_long",
            "legs_json": None,
        })

    return signals