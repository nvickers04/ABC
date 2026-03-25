"""
Slot 08 — Regime-Adaptive Bullish-Skew Iron Condor

Mandate: bullish / neutral / options_premium
Allowed order_types: iron_condor, butterfly, straddle, strangle

Bullish-leaning premium collection via iron condor with upside skew.
Wider call wing (more room on upside) while keeping tighter put wing.
Fully adapted to current high-vol, down-trend, bearish-breadth regime:
- Uses env dict to detect volatility_regime and trend_regime
- Shorter SMA20 for local structure instead of SMA50
- Mild bullish filter: relaxed to allow price modestly below SMA in down regime
- ROC threshold scaled by volatility regime (more tolerance in high vol)
- Bandwidth squeeze only enforced in normal/low vol regimes
- ATR-scaled strike distances with consistent upside skew
- Wider stops/targets in high-vol regime
- Earlier entry (bar 45+), moderate signal spacing, long hold for theta
- Correct legs_json key names per schema
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
ROC_BARS = 5
MIN_BAR_BASE = 45
MAX_HOLD_BARS = 180
SIGNAL_INTERVAL = 15


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < 120:
        return []

    signals = []
    close = candles["close"]
    
    # Regime awareness
    vol_regime = env.get("volatility_regime", "normal") if env and isinstance(env, dict) else "normal"
    trend_regime = env.get("trend_regime", "flat") if env and isinstance(env, dict) else "flat"
    is_high_vol = vol_regime == "high"
    is_down_trend = trend_regime == "down"
    
    sma_period = 20
    sma = close.rolling(sma_period).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)
    
    # Bollinger bandwidth for optional squeeze detection
    sma_bb = close.rolling(BB_PERIOD).mean()
    std_bb = close.rolling(BB_PERIOD).std(ddof=0)
    bandwidth = (std_bb * BB_STD * 2) / sma_bb
    bw_q50 = bandwidth.rolling(50, min_periods=25).quantile(0.50)
    
    min_bar = 30 if is_high_vol else MIN_BAR_BASE
    max_roc_abs = 0.035 if is_high_vol else 0.018
    max_hold = 160 if is_high_vol else MAX_HOLD_BARS
    wing_scale = 7.5 if is_high_vol else 5.0
    
    for i in range(min_bar, len(candles) - 25, SIGNAL_INTERVAL):
        if any(pd.isna(x) for x in (
            sma.iloc[i], atr.iloc[i], roc.iloc[i], 
            bandwidth.iloc[i], bw_q50.iloc[i]
        )) or atr.iloc[i] <= 0:
            continue

        price = float(close.iloc[i])
        sma_val = float(sma.iloc[i])
        a = float(atr.iloc[i])
        
        # Mild bullish-leaning filter, relaxed in downtrend regime
        if is_down_trend:
            if price < sma_val * 0.965:  # allow modest drawdown in bearish regime
                continue
        else:
            if price < sma_val * 0.992:
                continue
        
        # Range-bound filter with volatility-adjusted tolerance
        if abs(roc.iloc[i]) > max_roc_abs:
            continue
        
        # Bandwidth compression only required outside high-vol regime
        if not is_high_vol and bandwidth.iloc[i] > bw_q50.iloc[i] * 1.15:
            continue

        # Base strike rounded to nearest 5
        base = round(price / 5.0) * 5.0
        
        # Upside skew: tighter put wing, wider call wing (bullish-leaning)
        short_put = base - wing_scale
        long_put = short_put - wing_scale
        short_call = base + int(wing_scale * 1.8)
        long_call = short_call + int(wing_scale * 1.6)

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": price,
            "target_price": price + 0.4 * a,
            "stop_price": price - 2.2 * a,
            "max_hold_bars": max_hold,
            "setup_type": "regime_adapted_bullish_skew_ic",
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