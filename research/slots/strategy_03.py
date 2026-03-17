"""
Slot 03 — Short Momentum Breakdown with EMA Trend Filter
(High-Vol Downtrend + Steady Momentum + Contracting Volume Regime)

Evolved for current regime: high volatility (ATR 5.55%), trend=down (68%),
momentum=steady (+0.12), volume=contracting (0.80x), breadth=bullish (+0.52).
Aligns with momentum_breakout (0.85) and vwap (0.80) environment fit.

Core logic:
- Short-only momentum breakdown: breaks below 20-bar low on elevated volume
  while below both EMA20 and EMA50 (EMA50 acts as intraday proxy for daily downtrend)
- Removed acceleration filter (current regime is steady, not accelerating)
- Simplified negative momentum requirement (5-bar close < 0)
- Volume confirmation lowered to 1.5× 20-bar average to match contracting volume
- ATR(14) scaled stops/targets adjusted per analysis: STOP_ATR_MULT=0.75 (~4.2%)
  TARGET_ATR_MULT=1.6 (~8.9%) to survive high-vol noise wicks while maintaining
  realistic R:R within observed 5.22% intraday range
- Max hold increased slightly to 15 bars to let winners develop but still exit
  before late-day chop
- Morning-to-early-afternoon window (before bar 210 ≈ 13:00 ET)
- Added EMA50 downslope filter for better regime alignment on downtrend days
- No hard symbol filter; logic naturally selects weak momentum names
  (MARA, CAVA, DUOL, PLTR, NET) via breakdown + trend conditions
- Order type changed to stop_entry for more realistic breakdown execution
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 15
MIN_BAR = 50
VOL_MULT = 1.5
EMA_SPAN = 20
EMA_LONG_SPAN = 50
ATR_PERIOD = 14
STOP_ATR_MULT = 0.75
TARGET_ATR_MULT = 1.6
MAX_ENTRY_BAR = 210
LOOKBACK_BREAK = 20
MOM_BARS = 5


def calculate_atr(candles: pd.DataFrame, period: int) -> pd.Series:
    high_low = candles["high"] - candles["low"]
    high_close = np.abs(candles["high"] - candles["close"].shift(1))
    low_close = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < 100:
        return []

    signals = []

    # EMAs for trend filter
    ema_short = candles["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    ema_long = candles["close"].ewm(span=EMA_LONG_SPAN, adjust=False).mean()

    # Volume average
    vol_avg = candles["volume"].rolling(20).mean()

    # ATR
    atr = calculate_atr(candles, ATR_PERIOD)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(ema_short.iloc[i]) or pd.isna(ema_long.iloc[i]) or
            pd.isna(vol_avg.iloc[i]) or pd.isna(atr.iloc[i]) or
            atr.iloc[i] <= 0 or vol_avg.iloc[i] <= 0):
            continue

        # Need history for breakdown level and momentum
        if i < LOOKBACK_BREAK + MOM_BARS + 5:
            continue

        price = candles.iloc[i]["close"]
        e_short = ema_short.iloc[i]
        e_long = ema_long.iloc[i]
        a = atr.iloc[i]
        vol = candles.iloc[i]["volume"]
        vol_threshold = vol_avg.iloc[i] * VOL_MULT

        # Breakdown level from prior bars (not including current bar)
        prior_low = candles["low"].iloc[i - LOOKBACK_BREAK:i].min()

        # EMA50 downslope filter (regime alignment)
        ema50_slope_down = ema_long.iloc[i] < ema_long.iloc[i - 3]

        # Momentum (simple negative, no acceleration)
        mom_now = price - candles.iloc[i - MOM_BARS]["close"]

        # Short momentum breakdown setup
        if (candles.iloc[i]["low"] < prior_low and
            vol > vol_threshold and
            price < e_short and price < e_long and
            ema50_slope_down and
            mom_now < 0):
            entry_price = prior_low  # trigger level for stop_entry
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "stop_entry",
                "entry_price": entry_price,
                "target_price": entry_price - TARGET_ATR_MULT * a,
                "stop_price": entry_price + STOP_ATR_MULT * a,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "momentum_breakdown_short",
                "legs_json": None,
            })

    return signals