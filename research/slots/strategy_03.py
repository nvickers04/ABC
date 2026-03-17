"""
Slot 03 — Short-Only VWAP Rejection with Strict Trend Proxy + Accelerated Momentum
(High-Vol Downtrend Regime)

Evolved for current regime: high volatility (ATR 5.31%), trend=down (70%),
momentum=accelerating (+0.69), volume=normal, breadth=bullish. Aligns with
momentum_breakout (1.00) and vwap (0.80) environment fit.

Core logic:
- Short-only: VWAP cross-down on elevated volume while below both EMA20 and EMA50
  (EMA50 acts as intraday proxy for daily downtrend to filter out mismatched uptrend days)
- Momentum acceleration filter: 5-bar delta negative, more negative than preceding
  5-bar delta by at least 0.3 ATR (material acceleration for +0.69 regime)
- Volume confirmation raised to 2.5× 20-bar average (higher conviction in normal volume)
- ATR(14) scaled stops/targets tightened/widened per observed move sizes:
  STOP_ATR_MULT=0.40 (≈2.1%), TARGET_ATR_MULT=2.0 (≈10.6%) to reduce tiny losers
  while letting winners run in high-vol environment
- Max hold reduced to 12 bars to exit before afternoon chop
- Strict morning filter only (before bar 180 ≈ 12:30 ET)
- Long logic completely removed to eliminate counter-trend drag
- Focus on regime-aligned momentum names (MARA, CAVA, DUOL, NET, DKNG) via logic,
  not hard symbol filter to preserve signal count and avoid single-symbol overfitting
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 12
MIN_BAR = 60
VOL_MULT = 2.5
EMA_SPAN = 20
EMA_LONG_SPAN = 50
ATR_PERIOD = 14
STOP_ATR_MULT = 0.40
TARGET_ATR_MULT = 2.0
MAX_ENTRY_BAR = 180
MOM_ACCEL_ATR_THRESH = 0.3


def calculate_atr(candles: pd.DataFrame, period: int) -> pd.Series:
    high_low = candles["high"] - candles["low"]
    high_close = np.abs(candles["high"] - candles["close"].shift(1))
    low_close = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + 30:
        return []

    signals = []

    # VWAP
    cum_vol = candles["volume"].cumsum()
    cum_pv = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    # EMAs for trend filter (EMA50 as daily trend proxy)
    ema_short = candles["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    ema_long = candles["close"].ewm(span=EMA_LONG_SPAN, adjust=False).mean()

    # Volume average
    vol_avg = candles["volume"].rolling(20).mean()

    # ATR
    atr = calculate_atr(candles, ATR_PERIOD)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(vwap.iloc[i]) or pd.isna(ema_short.iloc[i]) or
            pd.isna(ema_long.iloc[i]) or pd.isna(vol_avg.iloc[i]) or
            pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0):
            continue

        # Need enough history for momentum acceleration
        if i < 12:
            continue

        price = candles.iloc[i]["close"]
        prev_price = candles.iloc[i - 1]["close"]
        v = vwap.iloc[i]
        e_short = ema_short.iloc[i]
        e_long = ema_long.iloc[i]
        a = atr.iloc[i]
        vol = candles.iloc[i]["volume"]
        vol_threshold = vol_avg.iloc[i] * VOL_MULT

        # Momentum acceleration (more negative delta = accelerating down)
        # mom_now < mom_prev - 0.3*ATR ensures material acceleration relative to vol
        mom_now = price - candles.iloc[i - 5]["close"]
        mom_prev = candles.iloc[i - 5]["close"] - candles.iloc[i - 10]["close"]

        # Short rejection: cross below VWAP on high volume, below both EMAs,
        # with material accelerating downward momentum (regime-aligned)
        if (prev_price > v and price < v and
            vol > vol_threshold and
            price < e_short and price < e_long and
            mom_now < 0 and mom_now < mom_prev - MOM_ACCEL_ATR_THRESH * a):
            entry = price
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "vwap",
                "entry_price": entry,
                "target_price": entry - TARGET_ATR_MULT * a,
                "stop_price": entry + STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "vwap_rejection_accel_short_strict",
                "legs_json": None,
            })

    return signals