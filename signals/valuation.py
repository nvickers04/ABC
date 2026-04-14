"""Signal 25: Relative valuation — P/E, P/S, PEG vs sector median."""

import numpy as np
from signals.base import Signal, SignalResult


class ValuationSignal(Signal):
    name = "relative_valuation"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        fundamentals = data.get("fundamentals")

        if fundamentals is None:
            return SignalResult(0.0, 0.0, {"error": "no fundamentals"})

        scores = []
        components = {}

        # P/E ratio: lower = undervalued
        pe = getattr(fundamentals, "pe_ratio", None) if hasattr(fundamentals, "pe_ratio") else None
        if pe is None:
            pe = getattr(fundamentals, "ev_to_ebitda", None)

        if pe and pe > 0:
            # Compare to universe median (~25 for growth stocks)
            pe_score = np.clip((25 - pe) / 30, -1, 1)
            scores.append(pe_score)
            components["pe_ratio"] = float(pe)
            components["pe_score"] = float(pe_score)

        # P/S ratio
        ps = getattr(fundamentals, "price_to_sales", None)
        if ps and ps > 0:
            ps_score = np.clip((8 - ps) / 12, -1, 1)
            scores.append(ps_score)
            components["price_to_sales"] = float(ps)

        # PEG ratio
        peg = getattr(fundamentals, "peg_ratio", None)
        if peg and peg > 0:
            peg_score = np.clip((1.5 - peg) / 2, -1, 1)
            scores.append(peg_score * 1.2)  # Weight PEG slightly higher
            components["peg_ratio"] = float(peg)

        if not scores:
            return SignalResult(0.0, 0.0, {"error": "no valuation metrics"})

        score = np.clip(np.mean(scores), -1, 1)
        confidence = min(1.0, len(scores) / 3.0 * 0.7)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
