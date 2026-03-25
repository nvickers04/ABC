"""
Slot 07 — Long Stock Pullback (Bullish regime)

Mandate: bullish / long / stock_pullback
Allowed order_types: limit, midprice

Buy dips into support with limit orders during established uptrends.
Catches controlled pullbacks to EMA9 in stacked-EMA trends with
volume dry-up on the dip (selling exhaustion).

Core logic:
- Price above VWAP and EMA20 (bullish structure)
- EMAs stacked: EMA9 > EMA20 (strong trend)
- Bar low touches or dips below EMA9 but close stays above (pullback, not breakdown)
- Volume dry-up on pullback bar (< 0.85x avg = selling exhaustion)
- Bar close in upper 55% of range (buyers stepping in)
- Limit order entry at current close (buying the dip)
- ATR-based risk: 0.60 ATR stop, 1.50 ATR target (2.5:1 R:R)
- Entry window: bars 30–200
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 20
MIN_BAR = 30
MAX_ENTRY_BAR = 200
EMA_FAST_SPAN = 9
EMA_SLOW_SPAN = 20
ATR_PERIOD = 14
VOL_PERIOD = 20
VOL_CEILING = 0.85
STOP_ATR_MULT = 0.60
TARGET_ATR_MULT = 1.50
BAR_CLOSE_MIN_PCT = 0.55


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + ATR_PERIOD + 10:
        return []

    signals = []
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    cum_vol = volume.cumsum()
    cum_pv = (close * volume).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    ema9 = close.ewm(span=EMA_FAST_SPAN, adjust=False).mean()
    ema20 = close.ewm(span=EMA_SLOW_SPAN, adjust=False).mean()
    atr = _atr(candles, ATR_PERIOD)
    vol_avg = volume.rolling(VOL_PERIOD).mean()

    for i in range(MIN_BAR, min(len(candles), MAX_ENTRY_BAR + 1)):
        if (pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0 or
            pd.isna(vol_avg.iloc[i]) or vol_avg.iloc[i] <= 0 or
            pd.isna(vwap.iloc[i]) or pd.isna(ema9.iloc[i]) or
            pd.isna(ema20.iloc[i])):
            continue

        c = close.iloc[i]
        lo = low.iloc[i]
        hi = high.iloc[i]
        bar_range = hi - lo
        if bar_range <= 0:
            continue

        # Bullish structure: above VWAP and EMA20, stacked EMAs
        if c <= vwap.iloc[i] or c <= ema20.iloc[i]:
            continue
        if ema9.iloc[i] <= ema20.iloc[i]:
            continue

        # Pullback: bar low touches EMA9, close stays above
        if lo > ema9.iloc[i]:
            continue
        if c < ema9.iloc[i]:
            continue

        # Volume dry-up on dip
        if volume.iloc[i] / vol_avg.iloc[i] > VOL_CEILING:
            continue

        # Bar close in upper range
        if (c - lo) / bar_range < BAR_CLOSE_MIN_PCT:
            continue

        a = atr.iloc[i]
        stop = round(c - STOP_ATR_MULT * a, 2)
        target = round(c + TARGET_ATR_MULT * a, 2)
        if stop >= c or target <= c:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "limit",
            "entry_price": round(c, 2),
            "target_price": target,
            "stop_price": stop,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "ema_pullback_long",
            "legs_json": None,
        })

    return signals