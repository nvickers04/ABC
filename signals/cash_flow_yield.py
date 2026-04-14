"""Signal 31: Cash flow yield — FCF / market cap."""

import numpy as np
from signals.base import Signal, SignalResult


class CashFlowYieldSignal(Signal):
    name = "cash_flow_yield"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        fundamentals = data.get("fundamentals")

        if fundamentals is None:
            return SignalResult(0.0, 0.0, {"error": "no fundamentals"})

        fcf = getattr(fundamentals, "free_cash_flow", None)

        # Need market cap from basic fundamentals  
        basic_fund = data.get("basic_fundamentals")
        market_cap = None
        if basic_fund:
            market_cap = getattr(basic_fund, "market_cap", None)

        if fcf is None:
            return SignalResult(0.0, 0.0, {"error": "no FCF data"})

        if market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap
            score = np.clip(fcf_yield * 20, -1, 1)  # 5% yield = strong
            components = {"fcf_yield": float(fcf_yield), "fcf": float(fcf)}
        else:
            # Simple positive/negative signal
            score = 0.5 if fcf > 0 else -0.5
            components = {"fcf": float(fcf), "fcf_yield": None}

        confidence = min(1.0, abs(score) * 1.2)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
