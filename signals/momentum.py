"""Signal 1: Trend/momentum — EMA crosses, ROC, MACD histogram."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength, safe_pct_change
from signals.base import Signal, SignalResult


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


class MomentumSignal(Signal):
    name = "momentum"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 50:
            return SignalResult(0.0, 0.0, {"error": "insufficient candle data"})

        close = np.array(candles.close, dtype=float)

        # EMA(10) vs EMA(50) cross direction and distance
        ema10 = _ema(close, 10)
        ema50 = _ema(close, 50)
        ema_diff = safe_pct_change(ema10[-1], ema50[-1])
        ema_score = bounded_tanh(ema_diff, scale=18.0)

        # 5-bar and 10-bar ROC
        roc5 = safe_pct_change(close[-1], close[-6]) if len(close) > 5 else 0.0
        roc10 = safe_pct_change(close[-1], close[-11]) if len(close) > 10 else 0.0
        roc_score = bounded_tanh((roc5 * 0.6) + (roc10 * 0.4), scale=12.0)

        # MACD histogram sign and slope
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        histogram = macd_line - signal_line
        macd_ratio = histogram[-1] / max(abs(close[-1]), 1e-9)
        macd_score = bounded_tanh(macd_ratio, scale=120.0)

        score = (ema_score * 0.4 + roc_score * 0.3 + macd_score * 0.3)
        data_quality = min(1.0, len(close) / 80.0)
        confidence = confidence_from_strength(abs(score), data_quality=data_quality)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "ema_score": float(ema_score),
                "roc_score": float(roc_score),
                "macd_score": float(macd_score),
            },
        )
