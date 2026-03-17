"""
Slot 10 — Short-Only VWAP Rejection (Tight ATR Risk, Dual-EMA + Momentum Filter)

Evolved for current regime: high volatility (ATR 5.28%), downtrend (68% trending down),
accelerating momentum (+0.78), bullish breadth (+0.28), normal volume (0.86x).
Prioritizes momentum_breakout (1.00) and vwap (0.80). Strict short bias only.

Core logic:
- VWAP cross-down rejections on high volume (>=2.0x 20-bar avg)
- Strong trend filter: shorts ONLY when below BOTH EMA20 and EMA50
- Momentum filter: 5-bar ROC must be negative (aligns with accelerating downtrend)
- Decisive rejection: must close at least 0.12 * ATR below VWAP (avoids wick noise)
- Tight ATR(14) stops/targets for high-vol regime (0.45 stop / 1.65 target)
  — addresses prior issue of never hitting wide stops and bleeding on small moves
- Morning-only entries (before bar 180 ≈ 12:00 ET) to capture highest-edge window
- Reduced max hold (12 bars) to minimize timeout drag in chop
- Completely removed all long logic given dominant downtrend regime
- Naturally surfaces on momentum leaders (MARA, CAVA, DUOL, PLTR, NET)
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 12
MIN_BAR = 50
VOL_MULT = 2.0
EMA_SHORT_SPAN = 20
EMA_LONG_SPAN = 50
ATR_PERIOD = 14
STOP_ATR_MULT = 0.45
TARGET_ATR_MULT = 1.65
MAX_ENTRY_BAR = 180
MIN_CROSS_DEPTH_ATR = 0.12


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

    # EMAs for trend filter
    ema20 = candles["close"].ewm(span=EMA_SHORT_SPAN, adjust=False).mean()
    ema50 = candles["close"].ewm(span=EMA_LONG_SPAN, adjust=False).mean()

    # Volume average
    vol_avg = candles["volume"].rolling(20).mean()

    # ATR
    atr = calculate_atr(candles, ATR_PERIOD)

    # Momentum (5-bar ROC, negative = down momentum)
    roc5 = candles["close"].pct_change(5)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(vwap.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]) or
            pd.isna(vol_avg.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(roc5.iloc[i]) or
            atr.iloc[i] <= 0):
            continue

        price = candles.iloc[i]["close"]
        prev_price = candles.iloc[i - 1]["close"]
        v = vwap.iloc[i]
        e20 = ema20.iloc[i]
        e50 = ema50.iloc[i]
        a = atr.iloc[i]
        vol = candles.iloc[i]["volume"]
        vol_threshold = vol_avg.iloc[i] * VOL_MULT
        momentum = roc5.iloc[i]

        # Short rejection: cross below VWAP on high volume, below both EMAs,
        # negative momentum, and decisive displacement below VWAP
        if (prev_price > v and price < v and
            vol > vol_threshold and
            price < e20 and price < e50 and
            momentum < 0 and
            (v - price) > (MIN_CROSS_DEPTH_ATR * a)):
            
            entry = price
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "vwap",
                "entry_price": entry,
                "target_price": entry - TARGET_ATR_MULT * a,
                "stop_price": entry + STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "vwap_rejection_momentum_short",
                "legs_json": None,
            })

    return signals