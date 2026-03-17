"""
Slot 04 — Dual-Sided ORB with Short Bias, Wide Stops & Momentum Filter
(High-Vol Downtrend v60)

Adapted to CURRENT environment: Volatility=high (ATR 5.28%), Trend=down (68% trending down),
Breadth=bullish (+0.28 A/D), Momentum=accelerating (+0.78), Volume=normal (0.86x).

Core idea: momentum_breakout (fit=1.00) using 15-min opening range breakouts/breakdowns
with explicit short-first bias to respect the dominant downtrend while capturing selective
longs on strong names only. Restricted to current top momentum and options candidates
(MARA, CAVA, DUOL, PLTR, NET, U, PATH, IOT, HUBS, HOOD, ROKU).

Uses OR range as volatility proxy for high-vol regime. Wider stops (0.9× OR) to survive
whipsaws in 5.28% ATR environment. Realistic ~2.2:1 R:R with 2.0× OR targets.
Volume confirmation relaxed for shorts (1.05×) and stricter for longs (1.4×) given
downtrend bias. Requires close-through confirmation on both sides to reduce false
wick breaks. Morning-only entries (before bar 90). Reduced hold (40 bars) to capture
front-loaded momentum before midday chop in downtrend regime. 5-bar momentum check
for confirmation on both sides.

Key improvements over prior versions:
- Wider stop (0.9× OR) calibrated to current extreme volatility
- Short-first logic + close-through + 5-bar momentum filter aligned with trend=down
- Selective symbol filter using current top momentum/options candidates
- MIN_OR_RANGE_PCT=0.015 to avoid noisy small ranges
- MAX_ENTRY_BAR=90 + MAX_HOLD_BARS=40 avoids afternoon noise and premature timeouts
- Distinct setup_type per side for regime tracking
- One signal per day using stop_entry for live execution realism
- Looser volume on shorts and stricter on longs to reflect 68% downtrend bias
- Momentum filter uses comparison to price 5 bars earlier (accelerating momentum label)
"""

import pandas as pd
import numpy as np

OPENING_BARS = 15
MAX_ENTRY_BAR = 90
MAX_HOLD_BARS = 40
STOP_MULT = 0.9
TARGET_MULT = 2.0
MIN_VOL_SHORT = 1.05
MIN_VOL_LONG = 1.4
MIN_OR_RANGE_PCT = 0.015

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
        is_mom_down = row["close"] < prev_price
        is_mom_up = row["close"] > prev_price

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

        # Long side (selective, stricter filters)
        elif (row["high"] > or_high and
              row["close"] > or_high and
              vol_ratio > MIN_VOL_LONG and
              is_mom_up):

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