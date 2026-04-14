"""Signal 8: Volume-weighted momentum — price change weighted by relative volume."""

import numpy as np
from signals.base import Signal, SignalResult


class VolumeWeightedMomentumSignal(Signal):
    name = "volume_weighted_momentum"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 25:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)
        volume = np.array(candles.volume, dtype=float)

        avg_vol_20 = np.mean(volume[-20:])
        if avg_vol_20 <= 0:
            return SignalResult(0.0, 0.0, {"error": "zero volume"})

        # Weighted momentum over different lookbacks
        scores = []
        for lookback in [5, 10, 20]:
            if len(close) <= lookback:
                continue
            price_change = (close[-1] - close[-lookback - 1]) / close[-lookback - 1]
            rel_vol = np.mean(volume[-lookback:]) / avg_vol_20
            weighted = price_change * rel_vol
            scores.append(weighted)

        if not scores:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        # Average across lookbacks
        avg_score = np.mean(scores)
        score = np.clip(avg_score * 15, -1, 1)

        # Confidence from volume magnitude
        rel_vol_now = volume[-1] / avg_vol_20
        confidence = min(1.0, rel_vol_now / 3.0 * abs(score))

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "avg_weighted_momentum": float(avg_score),
                "rel_volume_now": float(rel_vol_now),
            },
        )
