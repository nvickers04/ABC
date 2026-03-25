"""
Slot 11 — Adapted Neutral Iron Condor (High-Vol Downtrend)

Mandate: choppy / neutral / options_premium
Allowed order_types: iron_condor, butterfly, straddle, strangle

Evolved for CURRENT MARKET ENVIRONMENT: high volatility (ATR 5.12%), 
decisive downtrend (76% symbols trending down, -0.64 A/D, momentum shift -1.10), 
bearish breadth. 

While preserving neutral premium collection, we:
- Loosened range/chop filters that caused zero signals in prior versions
- Added slight downward skew to center the condor (respects regime without violating neutrality)
- Made wings dynamic and wider (~2×ATR) to survive 5%+ daily ranges and reduce tail losses
- Widened stop distance to 3.0×ATR for high-vol regime (avoids premature stops)
- Removed BB bandwidth filter (too regime-specific)
- Lowered MIN_BAR and SIGNAL_INTERVAL for sufficient signal count (~50-150 expected)
- Kept long theta-hold but realistic max_hold to limit overnight risk

This should improve expectancy vs prior -0.0451 by reducing large losers while still 
collecting premium on momentum symbols that often pause intraday.
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
ROC_BARS = 5
SMA_PERIOD = 30
MIN_BAR = 30
MAX_HOLD_BARS = 200
SIGNAL_INTERVAL = 20
MAX_ROC_ABS = 0.035
SMA_PROXIMITY_ATR = 2.5
BASE_WING_ATR_MULT = 2.0
DOWN_SKEW_ATR = -0.55


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < MIN_BAR + SMA_PERIOD + 20:
        return []

    signals = []
    close = candles["close"]
    sma30 = close.rolling(SMA_PERIOD).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS).fillna(0.0)

    for i in range(MIN_BAR, len(candles) - 30, SIGNAL_INTERVAL):
        if (pd.isna(sma30.iloc[i]) or pd.isna(atr.iloc[i]) or
            pd.isna(roc.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = float(close.iloc[i])
        a = float(atr.iloc[i])

        # Loosened proximity to SMA (allows trending days)
        if abs(price - sma30.iloc[i]) > SMA_PROXIMITY_ATR * a:
            continue

        # Loosened momentum filter (permits moderate downtrend)
        if abs(roc.iloc[i]) > MAX_ROC_ABS:
            continue

        # Use env when available to adapt skew for down regime
        skew = DOWN_SKEW_ATR
        if env is not None:
            if env.get("trend_regime") == "down" or env.get("breadth_regime") == "bearish":
                skew = DOWN_SKEW_ATR - 0.2  # more skew in strong downtrend
            if env.get("volatility_regime") == "high":
                # already handled via dynamic wings
                pass

        # Center condor with downward skew for regime fit
        center_price = price + skew * a
        base = round(center_price / 5.0) * 5.0

        # Dynamic wide wings for high volatility (prevents blowouts)
        wing_width = max(round(BASE_WING_ATR_MULT * a / 5.0) * 5.0, 7.5)

        put_long = base - 10.0 - wing_width
        put_short = base - 5.0
        call_short = base + 5.0
        call_long = call_short + wing_width

        entry_price = price
        stop_dist = 3.0 * a

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": entry_price,
            "target_price": entry_price,
            "stop_price": entry_price + stop_dist,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "neutral_iron_condor_adapted",
            "legs_json": {
                "strategy": "iron_condor",
                "expiration": "nearest_weekly",
                "put_long_strike": float(put_long),
                "put_short_strike": float(put_short),
                "call_short_strike": float(call_short),
                "call_long_strike": float(call_long)
            },
        })

    return signals