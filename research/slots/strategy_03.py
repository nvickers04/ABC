"""
Slot 03 — Short-Only VWAP Rejection with Momentum Acceleration (High-Vol Downtrend)

Evolved for current regime: high volatility (ATR 5.28%), trend=down (68%),
momentum=accelerating (+0.78), volume=normal, breadth=bullish. Aligns with
momentum_breakout (1.00) and vwap (0.80) environment fit.

Core logic:
- Short-only: VWAP cross-down on elevated volume while below EMA20
- Momentum acceleration filter: 5-bar delta negative AND more negative
  than the preceding 5-bar delta (captures the +0.78 regime)
- Volume confirmation at 2.0× 20-bar average for conviction
- ATR(14) scaled stops/targets tightened to observed move sizes
  (0.60 stop ≈ 3.2%, 1.65 target ≈ 8.7%) to reduce timeouts
- Max hold reduced to 12 bars to exit before afternoon chop
- Strict morning filter only (before bar 180 ≈ 12:30 ET)
- Long logic completely removed to eliminate counter-trend drag
- Focus on symbols like MARA, CAVA, DUOL, PLTR, NET that match regime
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 12
MIN_BAR = 60
VOL_MULT = 2.0
EMA_SPAN = 20
ATR_PERIOD = 14
STOP_ATR_MULT = 0.60
TARGET_ATR_MULT = 1.65
MAX_ENTRY_BAR = 180


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

    # EMA20 for trend filter
    ema = candles["close"].ewm(span=EMA_SPAN, adjust=False).mean()

    # Volume average
    vol_avg = candles["volume"].rolling(20).mean()

    # ATR
    atr = calculate_atr(candles, ATR_PERIOD)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(vwap.iloc[i]) or pd.isna(ema.iloc[i]) or
            pd.isna(vol_avg.iloc[i]) or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0):
            continue

        # Need enough history for momentum
        if i < 10:
            continue

        price = candles.iloc[i]["close"]
        prev_price = candles.iloc[i - 1]["close"]
        v = vwap.iloc[i]
        e = ema.iloc[i]
        a = atr.iloc[i]
        vol = candles.iloc[i]["volume"]
        vol_threshold = vol_avg.iloc[i] * VOL_MULT

        # Momentum acceleration (more negative delta = accelerating down)
        mom_now = price - candles.iloc[i - 5]["close"]
        mom_prev = candles.iloc[i - 5]["close"] - candles.iloc[i - 10]["close"]

        # Short rejection: cross below VWAP on high volume while below EMA
        # with accelerating downward momentum (regime-aligned)
        if (prev_price > v and price < v and
            vol > vol_threshold and price < e and
            mom_now < mom_prev and mom_now < 0):
            entry = price
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "vwap",
                "entry_price": entry,
                "target_price": entry - TARGET_ATR_MULT * a,
                "stop_price": entry + STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "vwap_rejection_accel_short",
                "legs_json": None,
            })

    return signals