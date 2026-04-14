"""Signal 33: Size factor — market cap relative to universe median (Fama-French SMB)."""

import numpy as np
from signals.base import Signal, SignalResult


class SizeFactorSignal(Signal):
    name = "size_factor"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        basic_fund = data.get("basic_fundamentals")

        market_cap = None
        if basic_fund:
            market_cap = getattr(basic_fund, "market_cap", None)

        if market_cap is None or market_cap <= 0:
            return SignalResult(0.0, 0.0, {"error": "no market cap"})

        # Universe median is approximately $5B-8B for medium-cap universe
        median_cap = 6e9

        # Smaller relative to universe = positive (small-cap premium)
        log_ratio = np.log(median_cap / market_cap)
        score = np.clip(log_ratio / 2.0, -1, 1)

        confidence = 0.4  # Size factor is weak but consistent

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "market_cap": float(market_cap),
                "log_ratio": float(log_ratio),
            },
        )
