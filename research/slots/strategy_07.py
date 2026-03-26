"""
Slot 07 — Long Stock Pullback (Regime-Adaptive RS Absorption)

Mandate: bullish / long / stock_pullback
Allowed order_types: limit, midprice

Role: Long stock pullback — buy dips into support with limit orders.
Heavily adapted to CURRENT high-vol (ATR=5.15%), trend=down, 
bearish-breadth regime (A/D=-0.56, 68% down). Focuses exclusively 
on relative-strength names (close > VWAP + EMA20) that show buyer 
absorption on EMA9 pullbacks. Uses env to switch volume logic and 
risk parameters. Limit entry posted at the EMA9 level itself (true 
dip-buy). Shorter hold, realistic small-target calibration for 
high-vol chop, and strict reversal filters to improve hit rate.

Core logic:
- Relative strength: close > VWAP and close > EMA20 (mandatory)
- Stacked trend relaxed to EMA9 > 0.995 * EMA20 in bearish regime
- Pullback: low undercuts EMA9, close recovers above EMA9
- Regime-aware volume: absorption (>=1.35x avg) in down/bearish days
- Reversal quality: green bar + close >= 62% of range
- RSI(14) between 38-68 to avoid deep oversold or momentum collapse
- High-vol risk: 0.50 ATR stop, 1.10 ATR target (~2.2:1) calibrated 
  to observed 1-6 bar bounce size in this regime
- Entry window: bars 45-220 (avoids open chaos + late-day fade)
- Max hold 11 bars to reduce timeout decay
"""

import pandas as pd
import numpy as np


MAX_HOLD_BARS = 11
MIN_BAR = 45
MAX_ENTRY_BAR = 220
EMA_FAST_SPAN = 9
EMA_SLOW_SPAN = 20
ATR_PERIOD = 14
VOL_PERIOD = 20
RSI_PERIOD = 14
VOL_ABS_MULT = 1.35
STOP_ATR_MULT = 0.50
TARGET_ATR_MULT = 1.10
BAR_CLOSE_MIN_PCT = 0.62


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < MIN_BAR + ATR_PERIOD + VOL_PERIOD + 10:
        return []

    signals = []
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # Regime detection from env (fallback to conservative defaults)
    trend_regime = env.get("trend_regime", "down") if env is not None else "down"
    breadth_regime = env.get("breadth_regime", "bearish") if env is not None else "bearish"
    is_bearish_regime = (trend_regime == "down" or breadth_regime == "bearish")

    cum_vol = volume.cumsum()
    cum_pv = (close * volume).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    ema9 = close.ewm(span=EMA_FAST_SPAN, adjust=False).mean()
    ema20 = close.ewm(span=EMA_SLOW_SPAN, adjust=False).mean()
    atr = _atr(candles, ATR_PERIOD)
    vol_avg = volume.rolling(VOL_PERIOD).mean()
    rsi = _rsi(close, RSI_PERIOD)

    for i in range(MIN_BAR, min(len(candles), MAX_ENTRY_BAR + 1)):
        if (pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0 or
            pd.isna(vol_avg.iloc[i]) or vol_avg.iloc[i] <= 0 or
            pd.isna(vwap.iloc[i]) or pd.isna(ema9.iloc[i]) or
            pd.isna(ema20.iloc[i]) or pd.isna(rsi.iloc[i])):
            continue

        c = close.iloc[i]
        lo = low.iloc[i]
        hi = high.iloc[i]
        bar_range = hi - lo
        if bar_range <= 0:
            continue

        # Relative strength filter (mandatory even in bearish regime)
        if c <= vwap.iloc[i] or c <= ema20.iloc[i]:
            continue

        # Relaxed stacking in bearish regime
        if is_bearish_regime:
            if ema9.iloc[i] <= 0.995 * ema20.iloc[i]:
                continue
        else:
            if ema9.iloc[i] <= ema20.iloc[i]:
                continue

        # Pullback to EMA9 support
        if lo >= ema9.iloc[i]:
            continue
        if c <= ema9.iloc[i]:
            continue

        # Volume: absorption (spike) in bearish regime, dry-up otherwise
        vol_ratio = volume.iloc[i] / vol_avg.iloc[i]
        if is_bearish_regime:
            if vol_ratio < VOL_ABS_MULT:
                continue
        else:
            if vol_ratio > 0.85:
                continue

        # Reversal quality
        if (c - lo) / bar_range < BAR_CLOSE_MIN_PCT:
            continue
        if c < close.iloc[i - 1]:  # prefers green or neutral bar
            continue

        # RSI filter to avoid exhausted names
        if rsi.iloc[i] < 38 or rsi.iloc[i] > 68:
            continue

        a = atr.iloc[i]
        stop = round(c - STOP_ATR_MULT * a, 2)
        target = round(c + TARGET_ATR_MULT * a, 2)
        if stop >= c or target <= c:
            continue

        # Limit order posted at the EMA9 support level (buy-the-dip)
        limit_price = round(ema9.iloc[i], 2)

        signals.append({
            "entry_bar": i,
            "direction": "long",
            "order_type": "limit",
            "entry_price": limit_price,
            "target_price": target,
            "stop_price": stop,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "rs_ema9_absorption",
            "legs_json": None,
        })

    return signals