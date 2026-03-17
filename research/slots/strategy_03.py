"""
Slot 03 — Short Momentum Breakdown with VWAP + EMA Filter (Regime-Aligned)
(High-Vol Downtrend + Steady Momentum + Contracting Volume)

Evolved for current regime: high volatility (ATR 5.55%), trend=down (68%),
momentum=steady (+0.12), volume=contracting (0.80x), breadth=bullish (+0.52).
Aligns with momentum_breakout (0.85) and vwap (0.80) environment fit.

Core logic:
- Short-only momentum breakdown: close breaks below 20-bar low on elevated
  volume (2.0×) while below EMA20, EMA50, and VWAP
- Close must finish in bottom 25% of its bar (avoids wick-only breaks)
- EMA50 downslope filter for downtrend alignment
- Negative 5-bar momentum (simple close delta < 0)
- ATR(14) scaled stops/targets widened for high-vol regime:
  STOP_ATR_MULT=1.1 (~6.1%), TARGET_ATR_MULT=2.0 (~11.1%) to survive
  noise while keeping realistic R:R inside 5.22% intraday range
- Max hold reduced to 10 bars to avoid late-day chop and timeout noise
- Morning-only window (before bar 210 ≈ 13:00 ET)
- Added intraday VWAP filter to enforce symbol-level down-bias and
  reduce false breakdowns on bullish tape days
- No hard symbol filter; logic naturally selects weak momentum names
  (MARA, CAVA, DUOL, PLTR, NET) via breakdown + trend + VWAP conditions
- Order type: stop_entry for realistic breakdown trigger
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 10
MIN_BAR = 60
VOL_MULT = 2.0
EMA_SPAN = 20
EMA_LONG_SPAN = 50
ATR_PERIOD = 14
STOP_ATR_MULT = 1.1
TARGET_ATR_MULT = 2.0
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


def calculate_vwap(candles: pd.DataFrame) -> pd.Series:
    typical_price = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    cum_tp_vol = (typical_price * candles["volume"]).cumsum()
    cum_vol = candles["volume"].cumsum()
    vwap = cum_tp_vol / cum_vol
    return vwap


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

    # VWAP for down-bias filter
    vwap_series = calculate_vwap(candles)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue

        if (pd.isna(ema_short.iloc[i]) or pd.isna(ema_long.iloc[i]) or
            pd.isna(vol_avg.iloc[i]) or pd.isna(atr.iloc[i]) or
            pd.isna(vwap_series.iloc[i]) or
            atr.iloc[i] <= 0 or vol_avg.iloc[i] <= 0):
            continue

        # Need history for breakdown level and momentum
        if i < LOOKBACK_BREAK + MOM_BARS + 5:
            continue

        price = candles.iloc[i]["close"]
        e_short = ema_short.iloc[i]
        e_long = ema_long.iloc[i]
        vwap_i = vwap_series.iloc[i]
        a = atr.iloc[i]
        vol = candles.iloc[i]["volume"]
        vol_threshold = vol_avg.iloc[i] * VOL_MULT

        # Breakdown level from prior bars (not including current bar)
        prior_low = candles["low"].iloc[i - LOOKBACK_BREAK:i].min()

        # EMA50 downslope filter (regime alignment)
        ema50_slope_down = ema_long.iloc[i] < ema_long.iloc[i - 3]

        # Momentum (simple negative)
        mom_now = price - candles.iloc[i - MOM_BARS]["close"]

        current_bar = candles.iloc[i]
        bar_range = current_bar["high"] - current_bar["low"]
        close_in_lower_quarter = (bar_range > 0 and
                                  price <= current_bar["low"] + 0.25 * bar_range)

        # Short momentum breakdown setup with improved filters
        if (current_bar["close"] < prior_low and
            close_in_lower_quarter and
            vol > vol_threshold and
            price < e_short and
            price < e_long and
            price < vwap_i and
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
                "setup_type": "momentum_breakdown_short_vwap",
                "legs_json": None,
            })

    return signals