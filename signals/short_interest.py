"""Signal 27: Short interest pressure — short % float, days to cover."""

import numpy as np
from signals.base import Signal, SignalResult


class ShortInterestSignal(Signal):
    name = "short_interest"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        fundamentals = data.get("fundamentals")

        if fundamentals is None:
            return SignalResult(0.0, 0.0, {"error": "no fundamentals"})

        short_pct = getattr(fundamentals, "short_percent_float", None)
        short_ratio = getattr(fundamentals, "short_ratio", None)

        if short_pct is None and short_ratio is None:
            return SignalResult(0.0, 0.0, {"error": "no short interest data"})

        components = {}
        scores = []

        # High short % = squeeze potential (contrarian bullish)
        if short_pct is not None:
            short_pct_val = float(short_pct) * 100 if short_pct < 1 else float(short_pct)
            pct_score = np.clip((short_pct_val - 5) / 20, -0.5, 1)  # >5% = above avg
            scores.append(pct_score)
            components["short_pct_float"] = short_pct_val

        # Days to cover > 5 = significant squeeze risk
        if short_ratio is not None:
            ratio_score = np.clip((float(short_ratio) - 2) / 6, -0.5, 1)
            scores.append(ratio_score)
            components["short_ratio_days"] = float(short_ratio)

        score = np.clip(np.mean(scores), -1, 1)
        confidence = min(1.0, abs(score) * 1.2)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
