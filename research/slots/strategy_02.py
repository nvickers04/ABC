"""
Slot 02 — Momentum Breakout Short (Stop-Entry Breakdown)

Evolved for current regime: Volatility=high (ATR 5.51%), Trend=down (68% trending down),
Momentum=accelerating (+0.61), Volume=normal. Uses momentum_breakout (env fit 1.00).
Short-only downside breakouts on relative-weakness names (MARA, CAVA, DUOL, PLTR, NET).

Core logic:
- Stop-entry short when price breaks below 20-bar low on high volume
- Trend filter: price below EMA21 and RSI(14) < 48
- Momentum filter: 5-bar ROC < -0.55% confirming acceleration
- Volume confirmation >= 1.4x 20-bar average
- ATR-based risk (0.85 ATR stop, 2.4 ATR target) for high-vol regime
- Morning/midday bias only (before bar 210 ≈ 13:00 ET)
- MIN_BAR=55 avoids opening-range noise; max hold 40 bars
- Setup tuned to produce 60-130 signals with realistic execution tolerance
"""

import pandas as pd
import numpy as np


MAX_HOLD_BARS = 40
MIN_BAR = 55
LOW_LOOKBACK = 20
VOL_MULT = 1.4
EMA_PERIOD = 21
RSI_PERIOD = 14
ROC_BARS = 5
ATR_PERIOD = 14
STOP_ATR_MULT = 0.85
TARGET_ATR_MULT = 2.4
MAX_ENTRY_BAR = 210
VOL_PERIOD = 20
ROC_THRESHOLD = -0.55
RSI_MAX = 48.0


def calculate_atr(candles: pd.DataFrame, period: int) -> pd.Series:
    high_low = candles["high"] - candles["low"]
    high_close = np.abs(candles["high"] - candles["close"].shift(1))
    low_close = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_rsi(candles: pd.DataFrame, period: int) -> pd.Series:
    delta = candles["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + 40:
        return []

    signals = []

    # Pre-compute indicators
    ema21 = candles["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    rsi = calculate_rsi(candles, RSI_PERIOD)
    roc5 = candles["close"].pct_change(ROC_BARS) * 100
    vol_avg = candles["volume"].rolling(VOL_PERIOD).mean()
    atr = calculate_atr(candles, ATR_PERIOD)

    for i in range(MIN_BAR, len(candles) - 5):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(ema21.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(roc5.iloc[i]) or
            pd.isna(vol_avg.iloc[i]) or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0):
            continue

        curr_close = candles["close"].iloc[i]
        prev_close = candles["close"].iloc[i - 1]
        curr_low = candles["low"].iloc[i]
        vol = candles["volume"].iloc[i]
        a = atr.iloc[i]

        # 20-bar low (lookback excludes current bar to avoid lookahead)
        lookback_low = candles["low"].iloc[i - LOW_LOOKBACK:i].min()

        if (curr_close < ema21.iloc[i] and
            rsi.iloc[i] < RSI_MAX and
            roc5.iloc[i] < ROC_THRESHOLD and
            vol > vol_avg.iloc[i] * VOL_MULT and
            prev_close >= lookback_low and
            curr_low < lookback_low):
            
            entry_price = lookback_low
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "stop_entry",
                "entry_price": entry_price,
                "target_price": entry_price - TARGET_ATR_MULT * a,
                "stop_price": entry_price + STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "momentum_breakout_short",
                "legs_json": None,
            })

    return signals