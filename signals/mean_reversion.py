"""Signal 2: Mean reversion — RSI, Bollinger Band position, SMA distance."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength, rolling_atr
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
        rsi_score = bounded_tanh((50.0 - rsi) / 25.0, scale=1.0)

        # Bollinger Band position
        sma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        if std20 > 0:
            bb_position = (close[-1] - sma20) / (2 * std20)
            bb_score = -bounded_tanh(bb_position, scale=1.2)  # Below band = bullish
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
        atr = rolling_atr(highs, lows, close, period=14)
        if atr > 0:
            sma_dist = -(close[-1] - sma20) / atr
            sma_dist_score = bounded_tanh(sma_dist, scale=0.6)
        else:
            sma_dist_score = 0.0

        score = rsi_score * 0.4 + bb_score * 0.35 + sma_dist_score * 0.25
        # Higher confidence at extremes and with adequate history.
        data_quality = min(1.0, len(close) / 90.0)
        confidence = confidence_from_strength(abs(score), data_quality=data_quality)

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
