"""Signal 10: Price acceleration — second derivative of price (ROC of ROC)."""

import numpy as np
from signals.base import Signal, SignalResult


class PriceAccelerationSignal(Signal):
    name = "price_acceleration"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 20:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)

        # First derivative: ROC at different lookbacks
        roc_5 = (close[-1] - close[-6]) / close[-6] if len(close) > 5 else 0
        roc_5_prev = (close[-6] - close[-11]) / close[-11] if len(close) > 10 else 0

        roc_10 = (close[-1] - close[-11]) / close[-11] if len(close) > 10 else 0
        roc_10_prev = (close[-11] - close[-21]) / close[-21] if len(close) > 20 else 0

        # Second derivative: change in ROC
        accel_5 = roc_5 - roc_5_prev
        accel_10 = roc_10 - roc_10_prev

        # Weighted combination
        accel = accel_5 * 0.6 + accel_10 * 0.4

        score = np.clip(accel * 30, -1, 1)
        confidence = min(1.0, abs(accel) * 40)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "accel_5": float(accel_5),
                "accel_10": float(accel_10),
                "roc_5": float(roc_5),
                "roc_10": float(roc_10),
            },
        )
