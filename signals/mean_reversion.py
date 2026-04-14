"""Signal 2: Mean reversion — RSI, Bollinger Band position, SMA distance."""

import numpy as np
from signals.base import Signal, SignalResult


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI from close prices."""
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) else 0
    avg_loss = np.mean(losses) if len(losses) else 1e-9
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class MeanReversionSignal(Signal):
    name = "mean_reversion"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 30:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)

        # RSI(14) distance from 50
        rsi = _rsi(close, 14)
        rsi_score = (50 - rsi) / 50  # Oversold > 0, overbought < 0
        rsi_score = np.clip(rsi_score, -1, 1)

        # Bollinger Band position
        sma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        if std20 > 0:
            bb_position = (close[-1] - sma20) / (2 * std20)
            bb_score = -np.clip(bb_position, -1, 1)  # Below band = bullish
        else:
            bb_score = 0.0

        # Distance from 20-day SMA in ATR units
        highs = np.array(candles.high, dtype=float)
        lows = np.array(candles.low, dtype=float)
        prev_close = close[-21:-1] if len(close) > 20 else close[-20:]
        h20 = highs[-20:]
        l20 = lows[-20:]
        n = min(len(prev_close), len(h20), len(l20))
        tr = np.maximum(
            h20[-n:] - l20[-n:],
            np.maximum(
                np.abs(h20[-n:] - prev_close[-n:]),
                np.abs(l20[-n:] - prev_close[-n:]),
            ),
        )
        atr = np.mean(tr) if len(tr) > 0 else 1e-9
        if atr > 0:
            sma_dist = -(close[-1] - sma20) / atr
            sma_dist_score = np.clip(sma_dist / 3, -1, 1)  # 3 ATR = max
        else:
            sma_dist_score = 0.0

        score = rsi_score * 0.4 + bb_score * 0.35 + sma_dist_score * 0.25
        # Higher confidence at extremes
        confidence = min(1.0, max(abs(rsi - 50) / 30, abs(bb_score)) * 1.2)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "rsi": float(rsi),
                "rsi_score": float(rsi_score),
                "bb_score": float(bb_score),
                "sma_dist_score": float(sma_dist_score),
            },
        )
