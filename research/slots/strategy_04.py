"""
Slot 04 — Regime-Gated Bearish-Skew Iron Condor (Tighter Upside + Dynamic ROC + More Samples)

Mandate: bearish / neutral / options_premium
Allowed order_types: butterfly, iron_condor, straddle, strangle

Adapted to CURRENT environment: Volatility=high (ATR 5.02%), Trend=down,
Breadth=bearish (A/D -0.54), Momentum=decelerating (-2.18), Volume=normal,
Dispersion=11.02. Short-premium fit 0.65.

Core logic:
- Hard regime gate for bearish breadth + down trend + high vol only
- Removed static symbol blacklist; now relies on dynamic ROC <= -0.010
- Price below SMA50 respects bearish backdrop
- ROC threshold tightened to -0.010 to filter weak momentum
- Realized-range vs ATR filter loosened (2.1×) to increase sample size
- Dynamic downside-skewed iron condor: tighter call wing (0.009-0.012)
- Strikes percentage-based then rounded to realistic increments
- Earlier entry window (bar 25+) + SIGNAL_INTERVAL=5 for more signals
- Tighter volatility-scaled stop (0.35×) and reduced max hold (25 bars)
  for high-vol regime to limit gamma risk on upside spikes
- Dispersion gate (<12.5) and volatility regime gate
- Target biased mildly lower to align with 67% down-trending symbols
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 50
ROC_BARS = 5
RANGE_BARS = 25
MIN_BAR = 25
MAX_HOLD_BARS = 25
SIGNAL_INTERVAL = 5


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _round_strike(price: float) -> float:
    """Realistic option strike rounding based on price level."""
    if price < 25:
        return round(price / 0.5) * 0.5
    elif price < 60:
        return round(price / 1.0) * 1.0
    else:
        return round(price / 5.0) * 5.0


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < MIN_BAR + ATR_PERIOD + SMA_PERIOD + 30:
        return []

    # Hard regime gate - only trade in the proven high-expectancy bucket
    if env is not None:
        breadth_regime = env.get("breadth_regime", "")
        trend_regime = env.get("trend_regime", "")
        volatility_regime = env.get("volatility_regime", "")
        if breadth_regime != "bearish" or trend_regime != "down":
            return []
        if volatility_regime != "high":
            return []
        
        # Avoid high-chaos days
        dispersion = env.get("dispersion", 11.0)
        if dispersion > 12.5:
            return []

    signals = []
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    sma50 = close.rolling(SMA_PERIOD).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)

    for i in range(MIN_BAR, len(candles) - 35, SIGNAL_INTERVAL):
        if (pd.isna(sma50.iloc[i]) or pd.isna(atr.iloc[i]) or 
            pd.isna(roc.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = float(close.iloc[i])
        a = float(atr.iloc[i])

        # Bearish context
        if price >= sma50.iloc[i]:
            continue

        # Dynamic momentum filter for decelerating bearish regime
        if roc.iloc[i] > -0.010:
            continue

        # Realized range filter — prefer periods not excessively stretched
        start = max(0, i - RANGE_BARS)
        recent_range = float(high.iloc[start:i+1].max() - low.iloc[start:i+1].min())
        expected_range = a * np.sqrt(RANGE_BARS) * 2.1
        if recent_range > expected_range:
            continue

        # Dynamic skew based on realized volatility (tighter call wing in high vol)
        vol_pct = a / price if price > 0 else 0
        if vol_pct > 0.045:
            call_mult = 0.009
            put_mult = 0.060
        else:
            call_mult = 0.012
            put_mult = 0.055
        wing_width = 0.021 * price

        short_call = _round_strike(price + call_mult * price)
        long_call = _round_strike(short_call + wing_width)
        short_put = _round_strike(price - put_mult * price)
        long_put = _round_strike(short_put - wing_width)

        # Avoid degenerate or too-narrow strikes
        if (short_call <= price or short_put >= price or 
            long_call <= short_call or long_put >= short_put or
            (long_call - short_call) < 0.5 or (short_put - long_put) < 0.5):
            continue

        # Tighter volatility-scaled exits for high-ATR regime
        vol_move = 1.0 * a * np.sqrt(18)
        target_price = price - 0.55 * vol_move
        stop_price = price + 0.35 * vol_move

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": price,
            "target_price": float(target_price),
            "stop_price": float(stop_price),
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "regime_gated_bearish_skew_iron_condor_v2",
            "legs_json": {
                "strategy": "iron_condor",
                "expiration": "nearest_weekly",
                "put_long_strike": float(long_put),
                "put_short_strike": float(short_put),
                "call_short_strike": float(short_call),
                "call_long_strike": float(long_call)
            },
        })

    return signals