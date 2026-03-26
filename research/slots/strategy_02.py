"""
Slot 02 — Short Stock Bracket (Bearish regime)

Mandate: bearish / short / stock_bracket
Allowed order_types: bracket, limit, oca_exit, trailing_stop_exit

Adapted for CURRENT MARKET ENVIRONMENT: high volatility (ATR 5.15%), 
down-trend (68% trending down, 77% trend confidence), bearish breadth 
(A/D = -0.56), decelerating momentum (-2.22), normal volume (0.97x).

Leans into momentum_breakout (0.85 fit) + vwap (0.80 fit) while 
staying strictly inside the short-bracket mandate.

Core logic (addressing prior failures: catastrophic R:R, oversized stops 
in high-vol regime, timeout bleed on decelerating momentum, weak mom 
filter, swing buffer making stops too wide):
- Hard regime gate using env (only trade on down + bearish breadth days)
- Strong bearish structure: below VWAP + EMA50 with declining EMA50 slope
- Momentum breakdown: red bar + new 10-bar close low + strong negative 5-bar mom (>=0.3 ATR)
- Volume expansion (1.5× 20-bar avg, tuned for normal-volume regime)
- RSI(14) < 40 confirms weakness
- Bracket tuned for high-vol regime: stop 1.15×ATR above entry (tightened), 
  target 2.0×ATR below (improved R:R for quick scalps that follow through)
- Removed wide swing-high buffer (primary cause of -0.9% losers)
- Tighter entry window (bars 40-130) + reduced max hold (12 bars) to kill bleed
- Setup reflects vwap_momentum_breakdown_short for interpretability
"""

import pandas as pd
import numpy as np


MAX_HOLD_BARS = 12
MIN_BAR = 40
VOL_MULT = 1.5
EMA_SHORT = 21
EMA_LONG = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
STOP_ATR_MULT = 1.15
TARGET_ATR_MULT = 2.0
MAX_ENTRY_BAR = 130
RSI_MAX = 40.0
LOW_LOOKBACK = 10
MOM_PERIOD = 5
MOM_ATR_THRESHOLD = 0.3


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


def scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]:
    if len(candles) < MIN_BAR + 30:
        return []

    # Hard regime gate - only trade in the current bearish environment
    if env is not None:
        if env.get("trend_regime") != "down" or env.get("breadth_regime") != "bearish":
            return []
        if env.get("volatility_regime") not in (None, "high"):
            return []

    signals = []
    ema21 = candles["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    ema50 = candles["close"].ewm(span=EMA_LONG, adjust=False).mean()
    rsi = _rsi(candles["close"], RSI_PERIOD)
    vol_avg = candles["volume"].rolling(20).mean()
    atr = _atr(candles, ATR_PERIOD)
    cum_vol = candles["volume"].cumsum()
    cum_pv = (candles["close"] * candles["volume"]).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)

    for i in range(MIN_BAR, len(candles) - 1):
        if i > MAX_ENTRY_BAR:
            continue
        if pd.isna(ema21.iloc[i]) or pd.isna(ema50.iloc[i]) or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        if pd.isna(rsi.iloc[i]) or pd.isna(vol_avg.iloc[i]) or pd.isna(vwap.iloc[i]):
            continue

        price = candles["close"].iloc[i]
        a = atr.iloc[i]

        # Bearish structure
        if price >= vwap.iloc[i] or price >= ema50.iloc[i]:
            continue

        # Declining EMA50 slope (bearish regime confirmation)
        if i > 5 and ema50.iloc[i] >= ema50.iloc[i - 5]:
            continue

        # Momentum breakdown: new 10-bar close low
        prev_min_close = candles["close"].iloc[max(0, i - LOW_LOOKBACK):i].min()
        if price >= prev_min_close:
            continue

        # Red bar (bearish candle)
        if candles["close"].iloc[i] >= candles["open"].iloc[i]:
            continue

        # Stronger short-term negative momentum (addressing decelerating regime)
        if i > MOM_PERIOD:
            mom_delta = candles["close"].iloc[i - MOM_PERIOD] - price
            if mom_delta < MOM_ATR_THRESHOLD * a:
                continue

        # RSI weakness
        if rsi.iloc[i] >= RSI_MAX:
            continue

        # Volume confirmation
        if candles["volume"].iloc[i] < vol_avg.iloc[i] * VOL_MULT:
            continue

        # Bracket levels: tighter stop, wider target for better R:R
        # Removed swing-high buffer that was making stops excessively wide
        stop = price + STOP_ATR_MULT * a
        target = price - TARGET_ATR_MULT * a

        if stop <= price or target >= price:
            continue

        signals.append({
            "entry_bar": i,
            "direction": "short",
            "order_type": "bracket",
            "entry_price": price,
            "target_price": target,
            "stop_price": stop,
            "max_hold_bars": MAX_HOLD_BARS,
            "setup_type": "vwap_momentum_breakdown_short",
            "legs_json": None,
        })

    return signals