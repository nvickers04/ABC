"""
Slot 01 — Short Stock Momentum: Volume Breakdown in Bearish Regime

Evolved for current regime: Volatility=high (ATR 5.15%), Trend=down (68% trending down),
Breadth=bearish (A/D -0.56), Momentum=decelerating (-2.22), Volume=normal (0.97x),
Trend confidence 77%. Prioritizes momentum_breakout (0.85) over vwap (0.80).
Strict short bias only. Uses env to gate non-bearish regimes.

Core logic:
- Regime gate: only trade on trend=down AND breadth=bearish (kills up/bullish toxicity)
- Momentum breakdown: close < low[1] on volume surge (>=1.8x 20-bar avg)
- Trend filter: price below EMA10 (fast response in high-vol regime)
- Momentum filter: 5-bar ROC <= -0.005 AND ROC5 < ROC10 (downside pressure)
- High-vol realistic risk: 0.75 ATR stop / 1.2 ATR target (~1.6:1 R:R, matches realized 0.2-0.4% moves)
- Morning through early afternoon entries (before bar 210 ≈ 13:00 ET)
- Max hold 20 bars to allow follow-through without excessive timeout
- Addresses prior failures: small-sample down-regime noise, fictional large ATR targets,
  1-bar noise losses, weak volume confirmation, and counter-trend triggers.
- Surfaces on momentum leaders (AFRM, PLTR, ROKU, APP, DUOL, MARA)
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 20
MIN_BAR = 60
VOL_MULT = 1.8
EMA_SPAN = 10
ATR_PERIOD = 14
STOP_ATR_MULT = 0.75
TARGET_ATR_MULT = 1.2
MAX_ENTRY_BAR = 210
MIN_ROC = -0.005


def calculate_atr(candles: pd.DataFrame, period: int) -> pd.Series:
    high_low = candles["high"] - candles["low"]
    high_close = np.abs(candles["high"] - candles["close"].shift(1))
    low_close = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < MIN_BAR + 30:
        return []

    # Regime gate - only trade in intended bearish environment
    if env is not None:
        trend = env.get("trend_regime")
        breadth = env.get("breadth_regime")
        if trend != "down" or breadth != "bearish":
            return []

    signals = []

    # Indicators
    vol_avg = candles["volume"].rolling(20).mean()
    ema10 = candles["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    atr = calculate_atr(candles, ATR_PERIOD)
    roc5 = candles["close"].pct_change(5)
    roc10 = candles["close"].pct_change(10)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(vol_avg.iloc[i]) or pd.isna(ema10.iloc[i]) or pd.isna(atr.iloc[i]) or
            pd.isna(roc5.iloc[i]) or pd.isna(roc10.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = candles.iloc[i]["close"]
        prev_low = candles.iloc[i - 1]["low"]
        vol = candles.iloc[i]["volume"]
        a = atr.iloc[i]
        momentum5 = roc5.iloc[i]
        momentum10 = roc10.iloc[i]
        vol_threshold = vol_avg.iloc[i] * VOL_MULT

        # Short momentum breakdown
        if (price < ema10.iloc[i] and
            candles.iloc[i]["close"] < prev_low and
            vol > vol_threshold and
            momentum5 <= MIN_ROC and
            momentum5 < momentum10):
            
            entry = price
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "market",
                "entry_price": entry,
                "target_price": entry - TARGET_ATR_MULT * a,
                "stop_price": entry + STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "momentum_breakdown_volume_short",
                "legs_json": None,
            })

    return signals