"""
Evolved Strategy — ATR-adaptive 20-bar breakout with volume confirmation,
trend filter, and volatility filter.

This file is rewritten by the research agent. The only contract is:
  scan(candles: pd.DataFrame, symbol: str) -> list[dict]

Each dict must contain:
  entry_bar, direction, order_type, entry_price, target_price,
  stop_price, max_hold_bars, setup_type, legs_json (optional)

Evolution from seed:
- ATR(14)-based stops and targets (1.5:3.0 ATR = 2:1 R:R) to give volatile names breathing room
- Start after 10:30 ET (MIN_BAR=60) to avoid morning chop/false breakouts
- Trend filter: close > SMA(50)
- Volatility filter: ATR(14)/price between 0.001 and 0.005 (realistic for 1-min bars)
- Stronger volume: > max(5-bar avg*2.5, 20-bar avg*2.5) using prior averages
- Reduced max hold to 40 bars to capture quick +0.6% moves before reversion
- Setup label updated for tracking
"""

import pandas as pd
import numpy as np


# ── Parameters ──────────────────────────────────────────────────
LOOKBACK = 20           # bars to look back for breakout high
VOLUME_MULT = 2.5       # volume multiplier (applied to both short and long avg)
ATR_PERIOD = 14         # ATR period for adaptive stops/targets
STOP_ATR_MULT = 1.5     # stop distance in ATR
TARGET_ATR_MULT = 3.0   # target distance in ATR (maintains ~2:1 R:R)
MAX_HOLD_BARS = 40      # reduced hold to capture quick winners
MIN_BAR = 60            # start after 10:30 ET (avoid open chop)
TREND_PERIOD = 50       # SMA period for trend filter
VOL_FILTER_LOW = 0.001  # minimum ATR/price (too quiet = no edge)
VOL_FILTER_HIGH = 0.005 # maximum ATR/price (too volatile = false breaks)


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Scan 1-min candles for ATR-adjusted breakout entries.

    Strategy logic:
    - Skip first 60 minutes to let open volatility settle
    - 20-bar high breakout on close with strong volume confirmation
    - Volume must exceed max(5-bar avg, 20-bar avg) * 2.5 (prior averages)
    - Trend filter: close > SMA(50)
    - Volatility filter: ATR(14)/price in [0.001, 0.005]
    - Adaptive stop = entry - 1.5 * ATR(14), target = entry + 3.0 * ATR(14)
    - Max hold reduced to 40 bars to lock in typical +0.65% winners
    """
    min_bars_needed = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 20
    if len(candles) < min_bars_needed + 1:
        return []

    signals = []

    # Pre-compute rolling metrics
    rolling_high = candles["high"].rolling(LOOKBACK).max()
    rolling_vol_avg = candles["volume"].rolling(LOOKBACK).mean().shift(1)
    rolling_vol_short = candles["volume"].rolling(5).mean().shift(1)

    # ATR calculation
    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close)
        )
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    # Trend filter
    sma_trend = candles["close"].rolling(TREND_PERIOD).mean()

    # Volatility ratio for filter
    vol_ratio = atr / candles["close"]

    start_idx = max(LOOKBACK, ATR_PERIOD, TREND_PERIOD, MIN_BAR) + 10

    for i in range(start_idx, len(candles) - 1):
        row = candles.iloc[i]

        # Safety check for NaNs
        if (pd.isna(rolling_high.iloc[i - 1]) or
            pd.isna(rolling_vol_avg.iloc[i]) or
            pd.isna(rolling_vol_short.iloc[i]) or
            pd.isna(atr.iloc[i]) or
            pd.isna(sma_trend.iloc[i]) or
            pd.isna(vol_ratio.iloc[i])):
            continue

        prev_high = rolling_high.iloc[i - 1]
        vol_avg = rolling_vol_avg.iloc[i]
        vol_short = rolling_vol_short.iloc[i]
        current_atr = atr.iloc[i]
        current_vol_ratio = vol_ratio.iloc[i]
        current_sma = sma_trend.iloc[i]

        # Core breakout conditions + filters
        volume_condition = row["volume"] > max(vol_short * VOLUME_MULT, vol_avg * VOLUME_MULT)

        if (row["close"] > prev_high and
            volume_condition and
            row["close"] > current_sma and
            VOL_FILTER_LOW <= current_vol_ratio <= VOL_FILTER_HIGH):

            entry_price = row["close"]
            stop_price = entry_price - STOP_ATR_MULT * current_atr
            target_price = entry_price + TARGET_ATR_MULT * current_atr

            # Sanity check on prices
            if stop_price >= entry_price or target_price <= entry_price:
                continue

            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "market",
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_price": stop_price,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "atr_breakout_volume_trend",
                "legs_json": None,
            })

    return signals