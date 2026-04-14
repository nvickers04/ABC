"""Signal 34: Debt health — leverage and liquidity metrics."""

import numpy as np
from signals.base import Signal, SignalResult


class DebtHealthSignal(Signal):
    name = "debt_health"
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

        # Debt-to-equity (lower is better)
        dte = getattr(fundamentals, "debt_to_equity", None)
        if dte is not None:
            dte_score = np.clip((100 - float(dte)) / 150, -1, 1)
            scores.append(dte_score)
            components["debt_to_equity"] = float(dte)

        # Current ratio (higher is better, >1.5 = healthy)
        cr = getattr(fundamentals, "current_ratio", None)
        if cr is not None:
            cr_score = np.clip((float(cr) - 1.0) / 1.5, -1, 1)
            scores.append(cr_score)
            components["current_ratio"] = float(cr)

        # Quick ratio
        qr = getattr(fundamentals, "quick_ratio", None)
        if qr is not None:
            qr_score = np.clip((float(qr) - 0.8) / 1.2, -1, 1)
            scores.append(qr_score)
            components["quick_ratio"] = float(qr)

        if not scores:
            return SignalResult(0.0, 0.0, {"error": "no debt metrics"})

        score = np.clip(np.mean(scores), -1, 1)
        confidence = min(1.0, len(scores) / 3.0 * 0.6)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
