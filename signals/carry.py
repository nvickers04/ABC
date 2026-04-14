"""Signal 32: Dividend carry — yield as value/income signal."""

import numpy as np
from signals.base import Signal, SignalResult


class CarrySignal(Signal):
    name = "dividend_carry"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        fundamentals = data.get("fundamentals")

        if fundamentals is None:
            return SignalResult(0.0, 0.0, {"error": "no fundamentals"})

        div_yield = getattr(fundamentals, "dividend_yield", None)

        if div_yield is None or div_yield == 0:
            # No dividend — neutral for growth stocks
            return SignalResult(0.0, 0.2, {"dividend_yield": 0, "note": "no dividend"})

        # Higher yield = income + value signal
        score = np.clip(float(div_yield) * 15, -1, 1)  # ~7% yield = max
        confidence = min(1.0, abs(div_yield) * 10)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={"dividend_yield": float(div_yield)},
        )
