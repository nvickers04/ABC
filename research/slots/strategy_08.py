"""
Slot 08 — Improved Bear Put Vertical Spread on Tight Squeeze + Strong Downtrend + Momentum Confirmation

Optimized for CURRENT regime: extreme volatility (ATR 5.85%), downtrend (32% up, 44% down),
decelerating momentum (shift -1.90), neutral breadth, high dispersion (11.93).
- Updated GOOD_SYMBOLS to current top options/momentum candidates (MARA, DUOL, U, CELH, IOT, PATH, NET)
- Tighter squeeze detection: bandwidth < 15% quantile + 3-bar monotonic contraction
- Stronger downtrend + momentum filter: close < SMA20 AND close < close[i-5]
- ATR-scaled risk levels (1.55×ATR stop, 2.5×ATR target) widened for extreme vol regime
- Morning-only entries (first ~2 hours), moderate hold/cooldown to control theta/gamma
- 5-point bear put vertical debit spread for directional downside capture in high-dispersion names
- Removed APP/CAVA which drove outsized losses; focused on names that respect breakdowns
"""

import pandas as pd
import numpy as np

GOOD_SYMBOLS = {"MARA", "DUOL", "U", "CELH", "IOT", "PATH", "NET"}

BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
SMA_PERIOD = 20
MIN_BAR = 40
MAX_ENTRY_BAR = 125
MAX_HOLD_BARS = 30
COOLDOWN_BARS = 25
STOP_ATR_MULT = 1.55
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
        if pd.isna(bandwidth.iloc[i]) or pd.isna(bandwidth_quantile.iloc[i]) or pd.isna(sma.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(bandwidth.iloc[i-1]) or pd.isna(bandwidth.iloc[i-2]):
            continue
        if i - last_signal_bar < COOLDOWN_BARS:
            continue

        # Tight squeeze + contracting + strong downtrend + momentum confirmation
        if (bandwidth.iloc[i] < bandwidth_quantile.iloc[i] and
            bandwidth.iloc[i] < bandwidth.iloc[i - 1] and
            bandwidth.iloc[i - 1] < bandwidth.iloc[i - 2] and
            closes.iloc[i] < sma.iloc[i] and
            closes.iloc[i] < closes.iloc[i - 5]):

            entry = closes.iloc[i]
            atr_val = atr.iloc[i]
            target_price = entry - TARGET_ATR_MULT * atr_val
            stop_price = entry + STOP_ATR_MULT * atr_val

            # Use strikes in $5 increments for realism/liquidity
            atm = round(entry / 5.0) * 5.0
            long_strike = atm + 5.0   # higher strike put (long)
            short_strike = atm        # lower strike put (short)

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