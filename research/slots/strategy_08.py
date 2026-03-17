"""
Slot 08 — Tight Squeeze Bear Put Vertical for Accelerating Downside Momentum

Optimized for CURRENT regime: high volatility (ATR 5.30%), downtrend (67% trending down),
accelerating momentum (+0.82), bullish breadth (+0.25), normal volume (0.86x), dispersion 8.71.
- GOOD_SYMBOLS updated to current momentum/options leaders that respect breakdowns (MARA, DUOL, NET, PLTR, CAVA, U, IOT)
- Stricter squeeze detection: bandwidth < 0.85×15% quantile + 4-bar monotonic contraction
- Downside momentum filters tuned for accelerating regime: close < SMA20 and close < close[i-3]
- Added volume confirmation on the breakdown bar
- Tighter stop (0.9×ATR) and shorter max hold (8 bars) to reduce exposure to small reversals
- Wider 10-point bear put vertical (long atm+5, short atm-5) to lower gamma sensitivity in high-vol
- Morning-only entries (first ~2 hours) to capture post-squeeze resolution
- Aligns with vertical_spread (0.85) while respecting momentum_breakout top rank (1.00)
"""

import pandas as pd
import numpy as np

GOOD_SYMBOLS = {"MARA", "DUOL", "NET", "PLTR", "CAVA", "U", "IOT"}

BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
SMA_PERIOD = 20
MIN_BAR = 40
MAX_ENTRY_BAR = 125
MAX_HOLD_BARS = 8
COOLDOWN_BARS = 25
STOP_ATR_MULT = 0.9
TARGET_ATR_MULT = 2.5


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if symbol not in GOOD_SYMBOLS:
        return []
    if len(candles) < 180:
        return []

    signals = []
    closes = candles["close"]
    highs = candles["high"]
    lows = candles["low"]
    volumes = candles["volume"]

    sma = closes.rolling(SMA_PERIOD).mean()
    std = closes.rolling(BB_PERIOD).std()
    bandwidth = (std * BB_STD * 2) / sma
    bandwidth_quantile = bandwidth.rolling(80, min_periods=40).quantile(0.15)

    prev_close = closes.shift(1)
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    atr = tr.rolling(ATR_PERIOD).mean()

    last_signal_bar = -COOLDOWN_BARS * 2
    start = max(BB_PERIOD, ATR_PERIOD, SMA_PERIOD, MIN_BAR, 80)

    for i in range(start, len(candles) - 15):
        if i > MAX_ENTRY_BAR:
            break
        if pd.isna(bandwidth.iloc[i]) or pd.isna(bandwidth_quantile.iloc[i]) or pd.isna(sma.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(bandwidth.iloc[i-1]) or pd.isna(bandwidth.iloc[i-2]) or pd.isna(bandwidth.iloc[i-3]) or pd.isna(closes.iloc[i-3]):
            continue
        if i - last_signal_bar < COOLDOWN_BARS:
            continue

        # Stricter squeeze + 4-bar contraction + momentum + volume confirmation
        if (bandwidth.iloc[i] < bandwidth_quantile.iloc[i] * 0.85 and
            bandwidth.iloc[i] < bandwidth.iloc[i - 1] and
            bandwidth.iloc[i - 1] < bandwidth.iloc[i - 2] and
            bandwidth.iloc[i - 2] < bandwidth.iloc[i - 3] and
            closes.iloc[i] < sma.iloc[i] and
            closes.iloc[i] < closes.iloc[i - 3] and
            volumes.iloc[i] > volumes.iloc[i - 1]):

            entry = closes.iloc[i]
            atr_val = atr.iloc[i]
            target_price = entry - TARGET_ATR_MULT * atr_val
            stop_price = entry + STOP_ATR_MULT * atr_val

            # Wider 10-point spread for high-vol regime
            atm = round(entry / 5.0) * 5.0
            long_strike = atm + 5.0   # higher strike put (long)
            short_strike = atm - 5.0  # lower strike put (short)

            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "vertical_spread",
                "entry_price": entry,
                "target_price": target_price,
                "stop_price": stop_price,
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "tight_squeeze_bear_put_vertical",
                "legs_json": {
                    "strategy": "vertical_spread",
                    "expiration": "nearest_weekly",
                    "long_strike": long_strike,
                    "short_strike": short_strike,
                    "right": "put"
                },
            })
            last_signal_bar = i

    return signals