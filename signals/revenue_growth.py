"""Signal 30: Revenue growth."""

import numpy as np
from signals.base import Signal, SignalResult


class RevenueGrowthSignal(Signal):
    name = "revenue_growth"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        fundamentals = data.get("fundamentals")

        if fundamentals is None:
            return SignalResult(0.0, 0.0, {"error": "no fundamentals"})

        growth = getattr(fundamentals, "revenue_growth", None)
        if growth is None:
            return SignalResult(0.0, 0.0, {"error": "no revenue growth data"})

        # Revenue growth: +30% = strong, -10% = declining
        score = np.clip(float(growth) * 3, -1, 1)
        confidence = min(1.0, abs(growth) * 4)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={"revenue_growth": float(growth)},
        )
