"""
Slot 08 — Bullish-Skew Iron Condor (Bullish regime)

Mandate: bullish / neutral / options_premium
Allowed order_types: iron_condor, butterfly, strangle, straddle

Premium collection via iron condor with upside skew in bullish regime.
Wider call wing, tighter put wing to lean bullish while collecting theta.

Core logic:
- Price above SMA50 (bullish backdrop)
- Low directional momentum (abs 5-bar ROC < 1.5%) — range-bound intraday
- Bollinger bandwidth compression (squeeze condition)
- Iron condor: sell OTM put & call, buy further OTM wings
  Put spread narrower (5pt), call spread wider (10pt) for bullish skew
- Entry after opening volatility settles (bar 60+)
- Long max_hold — premium strategies need time to decay
- 30-bar signal spacing
"""

import pandas as pd
import numpy as np

ATR_PERIOD = 14
SMA_PERIOD = 50
BB_PERIOD = 20
BB_STD = 2.0
ROC_BARS = 5
MIN_BAR = 60
MAX_HOLD_BARS = 200
MAX_ROC_ABS = 0.015
SIGNAL_INTERVAL = 30


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < MIN_BAR + SMA_PERIOD + BB_PERIOD + 10:
        return []

    signals = []
    close = candles["close"]
    sma50 = close.rolling(SMA_PERIOD).mean()
    atr = _atr(candles, ATR_PERIOD)
    roc = close.pct_change(ROC_BARS)

    # Bollinger bandwidth for squeeze detection
    sma_bb = close.rolling(BB_PERIOD).mean()
    std_bb = close.rolling(BB_PERIOD).std()
    bandwidth = (std_bb * BB_STD * 2) / sma_bb
    bw_q50 = bandwidth.rolling(60, min_periods=30).quantile(0.50)

    for i in range(MIN_BAR, len(candles) - 20, SIGNAL_INTERVAL):
        if (pd.isna(sma50.iloc[i]) or pd.isna(atr.iloc[i]) or
            pd.isna(roc.iloc[i]) or pd.isna(bandwidth.iloc[i]) or
            pd.isna(bw_q50.iloc[i]) or atr.iloc[i] <= 0):
            continue

        price = close.iloc[i]

        # Bullish context
        if price <= sma50.iloc[i]:
            continue
        # Range-bound — no strong directional move
        if abs(roc.iloc[i]) > MAX_ROC_ABS:
            continue
        # Bandwidth compression (squeeze-ish)
        if bandwidth.iloc[i] > bw_q50.iloc[i]:
            continue

        entry = float(price)
        a = float(atr.iloc[i])

        # Iron condor strikes — bullish skew (tighter put side)
        base = round(entry / 5.0) * 5.0
        short_put = base - 5.0
        long_put = short_put - 5.0        # 5-pt put wing (tighter)
        short_call = base + 10.0
        long_call = short_call + 10.0      # 10-pt call wing (wider)

        signals.append({
            "entry_bar": i,
            "direction": "neutral",
            "order_type": "iron_condor",
            "entry_price": entry,
            "target_price": entry + 0.5 * a,
            "stop_price": entry - 1.5 * a,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "bullish_skew_iron_condor",
            "legs_json": {
                "strategy": "iron_condor",
                "expiration": "nearest_weekly",
                "long_put": float(long_put),
                "short_put": float(short_put),
                "short_call": float(short_call),
                "long_call": float(long_call),
                "right": "both",
            },
        })

    return signals