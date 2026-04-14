"""Signal 26: Company quality — ROE, debt-to-equity, margins, FCF."""

import numpy as np
from signals.base import Signal, SignalResult


class QualitySignal(Signal):
    name = "company_quality"
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

        # ROE
        roe = getattr(fundamentals, "return_on_equity", None)
        if roe is not None:
            roe_score = np.clip(roe * 3, -1, 1)  # 33% ROE = max
            scores.append(roe_score)
            components["roe"] = float(roe)

        # Debt-to-equity (lower is better)
        dte = getattr(fundamentals, "debt_to_equity", None)
        if dte is not None:
            dte_score = np.clip((100 - dte) / 150, -1, 1)  # D/E < 100 = positive
            scores.append(dte_score)
            components["debt_to_equity"] = float(dte)

        # Profit margin
        margin = getattr(fundamentals, "profit_margin", None)
        if margin is not None:
            margin_score = np.clip(margin * 4, -1, 1)  # 25% margin = max
            scores.append(margin_score)
            components["profit_margin"] = float(margin)

        # Free cash flow (positive = good, scaled by market cap if available)
        fcf = getattr(fundamentals, "free_cash_flow", None)
        if fcf is not None:
            fcf_score = 0.5 if fcf > 0 else -0.5
            scores.append(fcf_score)
            components["free_cash_flow"] = float(fcf)

        if not scores:
            return SignalResult(0.0, 0.0, {"error": "no quality metrics"})

        score = np.clip(np.mean(scores), -1, 1)
        confidence = min(1.0, len(scores) / 4.0 * 0.7)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
