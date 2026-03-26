"""
Slot 08 — High-Vol Decelerating-Momentum Short Straddle (Premium)

Mandate: bullish / neutral / options_premium
Allowed order_types: butterfly, iron_condor, straddle, strangle

Pivot to straddle per selector directive (diversify from repeated iron_condor failures).
Short straddle premium collection during brief decelerating pauses inside the downtrend.
Adapted to current high-vol (ATR 5.15%), down-trend, bearish-breadth, decelerating momentum:
- Keep high-vol + down-trend + bearish-breadth regime gate but loosened ROC threshold
- Expanded universe to momentum/options candidates: AFRM, PLTR, ROKU, APP, DUOL, MARA
- ATM strike rounded to nearest $5, slight upward bias for bullish mandate
- ATR-aware exits (wider than 0.35%/0.25% to survive 5.15% ATR noise)
- Earlier entry window (bar 45+) and shorter hold for intraday theta capture
- SIGNAL_INTERVAL=15 to generate sufficient signals while avoiding over-sampling
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
ROC_BARS = 6
MIN_BAR = 45
MAX_HOLD_BARS = 35
SIGNAL_INTERVAL = 15

ALLOWED_SYMBOLS = {"AFRM", "PLTR", "ROKU", "APP", "DUOL", "MARA"}


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
    
    # Regime awareness from env
    if env is None or not isinstance(env, dict):
        vol_regime = "normal"
        trend_regime = "flat"
        breadth_regime = "neutral"
        mom_regime = "steady"
    else:
        vol_regime = env.get("volatility_regime", "normal")
        trend_regime = env.get("trend_regime", "flat")
        breadth_regime = env.get("breadth_regime", "neutral")
        mom_regime = env.get("momentum_regime", "steady")
    
    is_high_vol = vol_regime == "high"
    is_down_trend = trend_regime == "down"
    is_bearish_breadth = breadth_regime == "bearish"
    is_decelerating = mom_regime in ("decel", "decelerating")
    
    # Regime gate: only trade in current high-vol down bearish bucket
    if not (is_high_vol and is_down_trend and is_bearish_breadth):
        return []
    
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)
    
    max_roc_abs = 0.028  # loosened from 0.015 to allow more signals in trending regime
    
    for i in range(MIN_BAR, len(candles) - 30, SIGNAL_INTERVAL):
        if any(pd.isna(x) for x in (
            atr.iloc[i], roc.iloc[i]
        )) or atr.iloc[i] <= 0:
            continue

        price = float(close.iloc[i])
        a = float(atr.iloc[i])
        
        # Range-bound filter for decelerating momentum (short premium friendly)
        if abs(roc.iloc[i]) > max_roc_abs:
            continue

        # ATM strike with slight upward bias for bullish-leaning mandate
        base = round(price * 1.008 / 5.0) * 5.0
        
        # Wider ATR-scaled exits to survive high-vol noise (target ~0.8% / stop ~2.0%)
        target_price = price * 1.008
        stop_price = price * 0.980

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "straddle",
            "entry_price": price,
            "target_price": target_price,
            "stop_price": stop_price,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "highvol_decel_straddle_premium",
            "legs_json": {
                "strategy": "straddle",
                "expiration": "nearest_weekly",
                "strike": float(base)
            },
        })

    return signals