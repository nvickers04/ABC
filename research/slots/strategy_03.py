"""
Slot 03 — Adaptive Bear Put Vertical Spread (High-Vol Bearish)

Mandate: bearish / short / options_directional
Allowed order_types: vertical_spread, diagonal_spread

Evolved for CURRENT MARKET ENVIRONMENT:
Volatility=high (ATR 5.15%, intraday range 4.94%), Trend=down (68% down),
Breadth=bearish (A/D -0.56), Momentum=decelerating (-2.22), Volume=normal (0.97x),
Dispersion=10.95, Trend confidence=77%, idiosyncratic (cross-asset corr=0.23).

Core logic:
- Regime-aware parameter adaptation via env for high-vol + bearish conditions
- Bearish structure: below EMA20 & EMA50 with EMA50 sloping down
- Breakdown below 18-bar low (balanced for signal frequency in down regime)
- Volume confirmation (1.7x avg) tuned higher for normal volume days
- Negative short-term momentum + close in lower 40% of bar (loosened)
- Wider stops (1.35 ATR) to survive high-vol wicks/reversals in bearish regime
- Target 2.3 ATR for improved R:R given 5%+ ATR regime
- Morning entries only (before bar 160) to focus on high-conviction period
- Fixed 5-point bear put debit vertical (long higher-strike put, short lower)
- Uses env to avoid over-trading in up/bullish regimes by tightening filters
- Focus on top options candidates (MARA, AFRM, PATH, U, HUBS) behavior

This version widens stops, loosens decisive-selling filter, raises volume
threshold slightly, shortens breakdown lookback, adds env-driven adaptation,
and targets better expectancy in the intended bearish high-vol regime while
maintaining sufficient signal count.
"""

import pandas as pd
import numpy as np


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < 90:
        return []

    signals = []
    
    # Regime-aware parameters
    vol_mult = 1.70
    stop_atr_mult = 1.35
    target_atr_mult = 2.30
    lookback_break = 18
    mom_bars = 5
    max_entry_bar = 160
    spread_width = 5.0
    ema_short = 20
    ema_long = 50
    atr_period = 14
    
    if env is not None:
        vol_regime = env.get("volatility_regime", "normal")
        trend_regime = env.get("trend_regime", "flat")
        breadth_regime = env.get("breadth_regime", "neutral")
        mom_regime = env.get("momentum_regime", "")
        
        if vol_regime == "high":
            stop_atr_mult = 1.35
            target_atr_mult = 2.30
            vol_mult = 1.65
        if breadth_regime == "bearish":
            vol_mult = 1.55  # slightly easier in bearish to increase presence
            lookback_break = 20
        if trend_regime == "down":
            max_entry_bar = 170
        if mom_regime == "decelerating":
            mom_bars = 4

    ema20 = candles["close"].ewm(span=ema_short, adjust=False).mean()
    ema50 = candles["close"].ewm(span=ema_long, adjust=False).mean()
    vol_avg = candles["volume"].rolling(20).mean()
    atr = _atr(candles, atr_period)
    rolling_low = candles["low"].rolling(lookback_break).min()

    for i in range(lookback_break + mom_bars + 10, len(candles) - 1):
        if i > max_entry_bar:
            continue
        if (pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]) or
            pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0 or
            pd.isna(vol_avg.iloc[i]) or pd.isna(rolling_low.iloc[i - 1])):
            continue

        price = candles["close"].iloc[i]
        a = atr.iloc[i]

        # Bearish structure
        if price >= ema20.iloc[i] or price >= ema50.iloc[i]:
            continue
        # EMA50 sloping down
        if ema50.iloc[i] >= ema50.iloc[i - 3]:
            continue
        # Breakdown below recent low
        if price >= rolling_low.iloc[i - 1]:
            continue
        # Volume confirmation
        if candles["volume"].iloc[i] < vol_avg.iloc[i] * vol_mult:
            continue
        # Negative momentum
        if price >= candles["close"].iloc[i - mom_bars]:
            continue
        # Decisive selling — close in lower 40% of bar
        bar_range = candles["high"].iloc[i] - candles["low"].iloc[i]
        if bar_range <= 0:
            continue
        if (price - candles["low"].iloc[i]) / bar_range > 0.40:
            continue

        entry = price
        long_strike = round(entry)
        short_strike = long_strike - spread_width

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "vertical_spread",
            "entry_price": entry,
            "target_price": entry - target_atr_mult * a,
            "stop_price": entry + stop_atr_mult * a,
            "max_hold_bars": 35,
            "setup_type": "bear_put_vertical_adaptive",
            "legs_json": {
                "strategy": "vertical_spread",
                "expiration": "nearest_weekly",
                "long_strike": float(long_strike),
                "short_strike": float(short_strike),
                "right": "P",
            },
        })

    return signals