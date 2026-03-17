"""
Slot 04 — Dual-Sided ORB with Short Bias, Wider Stops & Stronger Long Filter
(High-Vol Downtrend v61)

Adapted to CURRENT environment: Volatility=high (ATR 5.28%), Trend=down (68% trending down),
Breadth=bullish (+0.28 A/D), Momentum=accelerating (+0.78), Volume=normal (0.86x).

Core idea: momentum_breakout (fit=1.00) using 15-min opening range breakouts/breakdowns
with explicit short-first bias to respect the dominant downtrend while capturing selective
longs on strong names only. Restricted to current top momentum and options candidates
(MARA, CAVA, DUOL, PLTR, NET, U, PATH, IOT, HUBS, HOOD, ROKU).

Uses OR range as volatility proxy for high-vol regime. Significantly wider stops (1.3× OR)
to survive whipsaws in extreme 5.28% ATR environment. Realistic ~2.1:1 R:R with 2.75× OR
targets. Volume confirmation relaxed for shorts (1.05×) and stricter for longs (1.65×)
given downtrend bias. Requires close-through confirmation on both sides. Morning-only
entries (before bar 90). 40-bar max hold to stay in front-loaded momentum window.
5-bar momentum check + extra 1.5% up-move threshold for longs to respect trend=down
while allowing bullish-breadth outliers.

Key improvements over prior version:
- Wider stop (1.3× OR) and larger target (2.75× OR) calibrated to extreme volatility
- Stronger long filter: MIN_VOL_LONG=1.65× plus >1.5% 5-bar momentum to reduce up-day noise
- Tighter MIN_OR_RANGE_PCT=0.022 to focus on meaningful ranges in 5%+ ATR regime
- Short-first logic + close-through + momentum filter aligned with 68% downtrend
- Distinct setup_type per side for regime tracking
- One signal per day using stop_entry for live execution realism
- Looser volume on shorts, stricter on longs to reflect dominant downtrend
"""

import pandas as pd
import numpy as np

OPENING_BARS = 15
MAX_ENTRY_BAR = 90
MAX_HOLD_BARS = 40
STOP_MULT = 1.3
TARGET_MULT = 2.75
MIN_VOL_SHORT = 1.05
MIN_VOL_LONG = 1.65
MIN_OR_RANGE_PCT = 0.022
MIN_LONG_MOM_PCT = 0.015

TOP_SYMBOLS = {
    "MARA", "CAVA", "DUOL", "PLTR", "NET",
    "U", "PATH", "IOT", "HUBS", "HOOD", "ROKU"
}


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < OPENING_BARS + 30:
        return []
    if symbol not in TOP_SYMBOLS:
        return []

    signals = []
    opening = candles.iloc[:OPENING_BARS]
    or_high = opening["high"].max()
    or_low = opening["low"].min()
    or_range = or_high - or_low
    if or_range <= 0:
        return []

    or_mid = (or_high + or_low) / 2.0
    or_range_pct = or_range / or_mid
    if or_range_pct < MIN_OR_RANGE_PCT:
        return []

    vol_avg = candles["volume"].rolling(20).mean()

    for i in range(OPENING_BARS, MAX_ENTRY_BAR):
        if pd.isna(vol_avg.iloc[i]):
            continue

        row = candles.iloc[i]
        vol_ratio = row["volume"] / vol_avg.iloc[i]

        # 5-bar momentum check (accelerating in direction of break)
        mom_bars = 5
        if i - mom_bars < 0:
            continue
        prev_price = candles.iloc[i - mom_bars]["close"]
        mom_pct = (row["close"] - prev_price) / prev_price
        is_mom_down = mom_pct < 0.0
        is_mom_up = mom_pct > 0.0

        # Short side first (bias for downtrend regime)
        if (row["low"] < or_low and
            row["close"] < or_low and
            vol_ratio > MIN_VOL_SHORT and
            is_mom_down):

            entry = or_low
            signals.append({
                "entry_bar": i,
                "direction": "short",
                "order_type": "stop_entry",
                "entry_price": entry,
                "target_price": entry - or_range * TARGET_MULT,
                "stop_price": entry + or_range * STOP_MULT,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "orb_momentum_breakdown_short",
                "legs_json": None,
            })
            break  # one signal per day

        # Long side (selective, stricter filters + strong momentum gate)
        elif (row["high"] > or_high and
              row["close"] > or_high and
              vol_ratio > MIN_VOL_LONG and
              is_mom_up and
              mom_pct > MIN_LONG_MOM_PCT):

            entry = or_high
            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "stop_entry",
                "entry_price": entry,
                "target_price": entry + or_range * TARGET_MULT,
                "stop_price": entry - or_range * STOP_MULT,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "orb_momentum_breakout_long",
                "legs_json": None,
            })
            break  # one signal per day

    return signals