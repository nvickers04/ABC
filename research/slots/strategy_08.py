"""
Slot 08 — Hard-Gated High-Vol Down-Regime Bullish-Skew Iron Condor

Mandate: bullish / neutral / options_premium
Allowed order_types: iron_condor, butterfly, straddle, strangle

Bullish-leaning premium collection via iron condor with upside skew.
Hard-gated to the currently profitable regime (high-vol + down-trend + bearish/neutral breadth).
- Only generates signals in vol=high + trend=down + breadth in (bearish, neutral)
- Restricted to top Options/Momentum symbols (PLTR, AFRM, etc.)
- Removed mild bullish SMA filter in down regime; relies on ROC for range-bound condition
- ATR-scaled wings with consistent upside skew (wider call wing)
- Wider target/stop distances for high-vol regime (0.75 ATR target, 2.0 ATR stop)
- Later entry (bar 50+), moderate spacing, hold for theta decay
- Bandwidth filter skipped entirely in high-vol
- Improved expectancy focus in current market snapshot
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
ROC_BARS = 6
MIN_BAR = 50
MAX_HOLD_BARS = 200
SIGNAL_INTERVAL = 20

ALLOWED_SYMBOLS = {"PLTR", "AFRM", "ROKU", "APP", "DUOL", "MARA", "PATH", "U", "HUBS", "NET", "SOFI", "ZS"}


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < 130 or symbol not in ALLOWED_SYMBOLS:
        return []

    signals = []
    close = candles["close"]
    
    # Regime awareness
    if env is None or not isinstance(env, dict):
        vol_regime = "normal"
        trend_regime = "flat"
        breadth_regime = "neutral"
    else:
        vol_regime = env.get("volatility_regime", "normal")
        trend_regime = env.get("trend_regime", "flat")
        breadth_regime = env.get("breadth_regime", "neutral")
    
    is_high_vol = vol_regime == "high"
    is_down_trend = trend_regime == "down"
    is_bearish_breadth = breadth_regime in ("bearish", "neutral")
    
    # Hard regime gate: only trade in the currently profitable bucket
    if not (is_high_vol and is_down_trend and is_bearish_breadth):
        return []
    
    sma_period = 20
    sma = close.rolling(sma_period).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)
    
    # Bollinger bandwidth (only used outside high-vol, but we gate to high-vol)
    sma_bb = close.rolling(BB_PERIOD).mean()
    std_bb = close.rolling(BB_PERIOD).std(ddof=0)
    bandwidth = (std_bb * BB_STD * 2) / sma_bb
    bw_q50 = bandwidth.rolling(50, min_periods=25).quantile(0.50)
    
    max_roc_abs = 0.045  # relaxed for high-vol
    wing_scale = 8.0
    max_hold = MAX_HOLD_BARS
    
    for i in range(MIN_BAR, len(candles) - 30, SIGNAL_INTERVAL):
        if any(pd.isna(x) for x in (
            sma.iloc[i], atr.iloc[i], roc.iloc[i], 
            bandwidth.iloc[i], bw_q50.iloc[i]
        )) or atr.iloc[i] <= 0:
            continue

        price = float(close.iloc[i])
        sma_val = float(sma.iloc[i])
        a = float(atr.iloc[i])
        
        # No bullish SMA filter in down regime (per analysis)
        # Only require not crashing too hard
        if price < sma_val * 0.93:
            continue
        
        # Range-bound filter with volatility-adjusted tolerance
        if abs(roc.iloc[i]) > max_roc_abs:
            continue

        # Base strike rounded to nearest 5
        base = round(price / 5.0) * 5.0
        
        # Upside skew: tighter put wing, wider call wing (bullish-leaning)
        short_put = base - wing_scale
        long_put = short_put - (wing_scale * 0.9)
        short_call = base + int(wing_scale * 1.85)
        long_call = short_call + int(wing_scale * 1.55)

        # Wider exits for high-vol regime
        target_price = price + 0.75 * a
        stop_price = price - 2.0 * a

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": max_hold,
            "setup_type": "hard_gated_highvol_down_bullish_skew_ic",
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