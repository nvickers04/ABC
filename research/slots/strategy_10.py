"""
Slot 10 — Mean Reversion Short (Choppy regime)

Mandate: choppy / short / stock_mean_reversion
Allowed order_types: limit, midprice, moc

Overbought fade entries at resistance levels with limit orders.
Shorts when RSI spikes above overbought threshold near upper Bollinger
Band in range-bound (choppy) markets.

Core logic:
- RSI(14) above 70 (overbought)
- Price touches or exceeds upper Bollinger Band (2 std)
- Price below SMA100 (not in a major uptrend — choppy/capped)
- Volume spike on blow-off bar (>1.3x avg)
- Close in lower 40% of bar range (sellers stepping in)
- Limit order at close (fading the overbought spike)
- Tight risk: 0.7 ATR stop, 1.4 ATR target (2:1 R:R)
- Entry window: bars 40–250
"""

import pandas as pd
import numpy as np

MAX_HOLD_BARS = 30
MIN_BAR = 40
MAX_ENTRY_BAR = 250
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70.0
BB_PERIOD = 20
BB_STD = 2.0
SMA_LONG = 100
ATR_PERIOD = 14
VOL_MULT = 1.3
STOP_ATR_MULT = 0.70
TARGET_ATR_MULT = 1.40
BAR_CLOSE_MAX_PCT = 0.40


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    hl = candles["high"] - candles["low"]
    hc = np.abs(candles["high"] - candles["close"].shift(1))
    lc = np.abs(candles["low"] - candles["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def scan(candles: pd.DataFrame, symbol: str) -> list[dict]:
    if len(candles) < SMA_LONG + 20:
        return []

    signals = []
    close = candles["close"]
    rsi = _rsi(close, RSI_PERIOD)
    sma100 = close.rolling(SMA_LONG).mean()
    sma_bb = close.rolling(BB_PERIOD).mean()
    std_bb = close.rolling(BB_PERIOD).std()
    upper_band = sma_bb + BB_STD * std_bb
    atr = _atr(candles, ATR_PERIOD)
    vol_avg = candles["volume"].rolling(20).mean()

    for i in range(MIN_BAR, min(len(candles), MAX_ENTRY_BAR + 1)):
        if (pd.isna(rsi.iloc[i]) or pd.isna(sma100.iloc[i]) or
            pd.isna(upper_band.iloc[i]) or pd.isna(atr.iloc[i]) or
            atr.iloc[i] <= 0 or pd.isna(vol_avg.iloc[i])):
            continue

        price = close.iloc[i]
        lo = candles["low"].iloc[i]
        hi = candles["high"].iloc[i]
        bar_range = hi - lo
        if bar_range <= 0:
            continue

        # Overbought
        if rsi.iloc[i] <= RSI_OVERBOUGHT:
            continue
        # Touching upper Bollinger Band
        if hi < upper_band.iloc[i]:
            continue
        # Not in major uptrend (below SMA100 = capped/choppy)
        if price > sma100.iloc[i]:
            continue
        # Volume blow-off
        if candles["volume"].iloc[i] < vol_avg.iloc[i] * VOL_MULT:
            continue
        # Bar close shows sellers (lower portion)
        if (price - lo) / bar_range > BAR_CLOSE_MAX_PCT:
            continue

        a = atr.iloc[i]
        stop = round(price + STOP_ATR_MULT * a, 2)
        target = round(price - TARGET_ATR_MULT * a, 2)
        if stop <= price or target >= price:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "limit",
            "entry_price": round(price, 2),
            "target_price": target,
            "stop_price": stop,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "overbought_fade_short",
            "legs_json": None,
        })

    return signals