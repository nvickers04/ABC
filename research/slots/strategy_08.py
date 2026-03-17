"""
Slot 08 — Tight Squeeze Bear Put Vertical for Accelerating Downside Momentum

Optimized for CURRENT regime: high volatility (ATR 5.35%), downtrend (71% trending down),
accelerating momentum (+0.54), neutral breadth, contracting volume (0.80x), dispersion 10.89.
- GOOD_SYMBOLS updated to current momentum/options leaders that respect breakdowns (CAVA, DUOL, PLTR, NET, DKNG)
- Loosened squeeze detection: bandwidth < 0.95×15% quantile + 4-bar monotonic contraction
- Strengthened downside momentum filters tuned for accelerating regime: close < SMA20, close < close[i-3], close < close[i-5]
- Volume confirmation on the breakdown bar (> prior bar)
- Looser stop (1.4×ATR) and higher target (3.0×ATR) for better R:R in high-vol whipsaw environment
- Retained 10-point bear put vertical for defined risk and lower gamma sensitivity
- Morning-only entries (first ~2 hours) to capture post-squeeze resolution
- Aligns with momentum_breakout (1.00) while using vertical_spread (0.85)
"""

import pandas as pd
import numpy as np

GOOD_SYMBOLS = {"CAVA", "DUOL", "PLTR", "NET", "DKNG"}

BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
SMA_PERIOD = 20
VOLUME_PERIOD = 20
MIN_BAR = 40
MAX_ENTRY_BAR = 125
MAX_HOLD_BARS = 10
COOLDOWN_BARS = 25
STOP_ATR_MULT = 1.4
TARGET_ATR_MULT = 3.0


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
    avg_volume = volumes.rolling(VOLUME_PERIOD).mean()

    last_signal_bar = -COOLDOWN_BARS * 2
    start = max(BB_PERIOD, ATR_PERIOD, SMA_PERIOD, MIN_BAR, VOLUME_PERIOD, 80)

    for i in range(start, len(candles) - 15):
        if i > MAX_ENTRY_BAR:
            break
        if pd.isna(bandwidth.iloc[i]) or pd.isna(bandwidth_quantile.iloc[i]) or pd.isna(sma.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(bandwidth.iloc[i-1]) or pd.isna(bandwidth.iloc[i-2]) or pd.isna(bandwidth.iloc[i-3]) or pd.isna(closes.iloc[i-3]) or pd.isna(closes.iloc[i-5]) or pd.isna(avg_volume.iloc[i]):
            continue
        if i - last_signal_bar < COOLDOWN_BARS:
            continue

        # Loosened squeeze + 4-bar contraction + stronger downside momentum + volume confirmation
        if (bandwidth.iloc[i] < bandwidth_quantile.iloc[i] * 0.95 and
            bandwidth.iloc[i] < bandwidth.iloc[i - 1] and
            bandwidth.iloc[i - 1] < bandwidth.iloc[i - 2] and
            bandwidth.iloc[i - 2] < bandwidth.iloc[i - 3] and
            closes.iloc[i] < sma.iloc[i] and
            closes.iloc[i] < closes.iloc[i - 3] and
            closes.iloc[i] < closes.iloc[i - 5] and
            volumes.iloc[i] > volumes.iloc[i - 1]):

            entry = closes.iloc[i]
            atr_val = atr.iloc[i]
            target_price = entry - TARGET_ATR_MULT * atr_val
            stop_price = entry + STOP_ATR_MULT * atr_val

            # 10-point bear put vertical for high-vol regime
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