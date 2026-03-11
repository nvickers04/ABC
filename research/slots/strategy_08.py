"""
Slot 08 — Bear Put Vertical Spread on Volatility Squeeze + Downtrend Filter

Optimized for CURRENT regime: extreme volatility (ATR 6.1%), downtrend (53% down days),
decelerating momentum, neutral breadth, high dispersion (10.61).
- Hard filter to momentum + options names only (MARA, PATH, U, CAVA, APP, CELH, ZS, PLTR, NET)
- Dynamic squeeze detection using 20% quantile of bandwidth + contracting
- Downtrend filter (close < SMA20) to align with regime
- Morning-only bias (first ~2 hours) to avoid afternoon chop/theta
- ATR-scaled logic but using widened % stops/targets for 6%+ ATR regime
- Reduced max hold (30 bars) to limit premium decay on non-events
- Bear put vertical (5-point width) gives bearish bias vs neutral straddle
- Cooldown prevents signal clustering on same squeeze
- Wider target:stop (12%:6.5%) matches observed winner size in high-vol names
"""

import pandas as pd
import numpy as np

GOOD_SYMBOLS = {"MARA", "PATH", "U", "CAVA", "APP", "CELH", "ZS", "PLTR", "NET"}

BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
SMA_PERIOD = 20
MIN_BAR = 30
MAX_ENTRY_BAR = 125
MAX_HOLD_BARS = 30
COOLDOWN_BARS = 25
STOP_PCT = 6.5
TARGET_PCT = 12.0


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if symbol not in GOOD_SYMBOLS:
        return []
    if len(candles) < 180:
        return []

    signals = []
    closes = candles["close"]
    highs = candles["high"]
    lows = candles["low"]

    sma = closes.rolling(SMA_PERIOD).mean()
    std = closes.rolling(BB_PERIOD).std()
    bandwidth = (std * BB_STD * 2) / sma
    bandwidth_quantile = bandwidth.rolling(80, min_periods=40).quantile(0.20)

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
        if pd.isna(bandwidth.iloc[i]) or pd.isna(bandwidth_quantile.iloc[i]) or pd.isna(sma.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        if i - last_signal_bar < COOLDOWN_BARS:
            continue

        # Dynamic squeeze + contracting + downtrend filter
        if (bandwidth.iloc[i] < bandwidth_quantile.iloc[i] and
            bandwidth.iloc[i] < bandwidth.iloc[i - 1] and
            closes.iloc[i] < sma.iloc[i]):

            entry = closes.iloc[i]
            # Use strikes in $5 increments for realism
            atm = round(entry / 5.0) * 5.0
            long_strike = atm + 5.0   # higher strike put (long)
            short_strike = atm        # lower strike put (short)

            signals.append({
                "entry_bar": i,
                "direction": "long",
                "order_type": "vertical_spread",
                "entry_price": entry,
                "target_price": entry * (1 - TARGET_PCT / 100.0),
                "stop_price": entry * (1 + STOP_PCT / 100.0),
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "squeeze_bear_put_vertical",
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