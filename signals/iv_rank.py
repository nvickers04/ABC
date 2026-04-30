"""Signal 15: IV rank — IV percentile position (sell premium when high)."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength
from signals.base import Signal, SignalResult


class IVRankSignal(Signal):
    name = "iv_rank"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2
    # IV percentile changes meaningfully over days, not hours.
    return_resolution = "D"
    return_horizon = 3
    return_lookback_days = 60

    def compute(self, symbol: str, data: dict) -> SignalResult:
        iv_info = data.get("iv_info")

        if iv_info is None or iv_info.iv_rank is None:
            return SignalResult(0.0, 0.0, {"error": "no IV rank data"})

        iv_rank = iv_info.iv_rank  # 0-100

        # High IV rank = premium-selling opportunity (+1)
        # Low IV rank = cheap options (-1)
        centered = (iv_rank - 50.0) / 50.0
        score = bounded_tanh(centered, scale=1.4)

        # Confidence higher at extremes
        confidence = confidence_from_strength(abs(score), data_quality=1.0)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "iv_rank": float(iv_rank),
                "iv_current": float(iv_info.iv_current) if iv_info.iv_current else 0,
            },
        )
