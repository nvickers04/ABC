"""
Slot 12 — Momentum Breakdown Short (VWAP + EMA Filter)

Short momentum continuation in accelerating downtrend regime.
Triggers on break of recent lows while price < VWAP and < EMA20,
with volume confirmation. Designed for high-vol (ATR~5.5%), 
68% down-trending environment. Bracket order with realistic 
ATR multiples (2.1 stop, 4.0 target) for 5%+ daily ranges.
Focuses exclusively on strong momentum names that trend.
Reduced max hold to avoid chop; avoids first ~45min and late day.
"""

import pandas as pd
import numpy as np

ALLOWED_SYMBOLS = {"MARA", "CAVA", "DUOL", "PLTR", "NET", "U", "PATH"}

RSI_PERIOD = 14
EMA_PERIOD = 20
ATR_PERIOD = 14
RECENT_LOW_BARS = 10
VOL_PERIOD = 20
VOL_MULT = 1.25
STOP_ATR_MULT = 2.1
TARGET_ATR_MULT = 4.0
MAX_HOLD_BARS = 14
MIN_BAR = 45
MAX_ENTRY_BAR = 300


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if symbol not in ALLOWED_SYMBOLS:
        return []
    
    min_bars = max(RSI_PERIOD, ATR_PERIOD, EMA_PERIOD, VOL_PERIOD, MIN_BAR) + 20
    if len(candles) < min_bars + 1:
        return []

    signals = []
    
    rsi = _rsi(candles["close"], RSI_PERIOD)
    ema = _ema(candles["close"], EMA_PERIOD)
    
    cum_vol = candles["volume"].cumsum()
    cum_vp = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_vp / cum_vol.replace(0, 1e-10)
    
    prev_close = candles["close"].shift(1)
    tr = np.maximum(
        candles["high"] - candles["low"],
        np.maximum(
            np.abs(candles["high"] - prev_close),
            np.abs(candles["low"] - prev_close),
        ),
    )
    atr = tr.rolling(ATR_PERIOD).mean()
    avg_vol = candles["volume"].rolling(VOL_PERIOD).mean()

    start = max(RSI_PERIOD, ATR_PERIOD, EMA_PERIOD, VOL_PERIOD, MIN_BAR) + 5
    for i in range(start, min(len(candles) - 1, MAX_ENTRY_BAR)):
        row = candles.iloc[i]
        if (pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i]) or 
            pd.isna(vwap.iloc[i]) or pd.isna(ema.iloc[i]) or 
            pd.isna(avg_vol.iloc[i])):
            continue

        price = row["close"]
        cur_atr = atr.iloc[i]
        if cur_atr <= 0:
            continue

        # Momentum breakdown short: below both VWAP and EMA,
        # breaks recent low with volume expansion.
        # Avoids RSI-oversold traps in downtrend.
        recent_low = candles["low"].iloc[i - RECENT_LOW_BARS:i].min()
        
        if (
            price < vwap.iloc[i]
            and price < ema.iloc[i]
            and rsi.iloc[i] > 45.0
            and row["low"] < recent_low
            and row["volume"] > avg_vol.iloc[i] * VOL_MULT
        ):
            entry = candles.iloc[i + 1]["open"]
            stop = entry + cur_atr * STOP_ATR_MULT
            target = entry - cur_atr * TARGET_ATR_MULT

            signals.append({
                "entry_bar": i + 1,
                "direction": "short",
                "order_type": "bracket",
                "entry_price": round(entry, 2),
                "target_price": round(target, 2),
                "stop_price": round(stop, 2),
                "max_hold_bars": MAX_HOLD_BARS,
                "setup_type": "momentum_breakdown_vwap",
                "legs_json": None,
            })

    return signals