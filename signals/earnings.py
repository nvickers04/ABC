"""Signal 24: Earnings momentum — EPS surprise + proximity risk."""

import numpy as np
from signals.base import Signal, SignalResult


class EarningsSignal(Signal):
    name = "earnings_momentum"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1
    # Post-earnings drift literature: 1–3 weeks horizon.
    return_resolution = "D"
    return_horizon = 10
    return_lookback_days = 120

    def compute(self, symbol: str, data: dict) -> SignalResult:
        earnings_info = data.get("earnings_info")
        earnings_history = data.get("earnings_history")

        if not earnings_history:
            return SignalResult(0.0, 0.0, {"error": "no earnings history"})

        # Last earnings surprise
        surprises = []
        for rec in earnings_history:
            if isinstance(rec, dict):
                surprise = rec.get("surprise_eps_pct") or rec.get("surprisePercent")
            else:
                surprise = getattr(rec, "surprise_eps_pct", None) or getattr(rec, "surprisePercent", None)
            if surprise is not None:
                surprises.append(float(surprise))

        if not surprises:
            return SignalResult(0.0, 0.0, {"error": "no surprise data"})

        last_surprise = surprises[0]  # Most recent
        avg_surprise = np.mean(surprises[:4])  # Last 4 quarters

        # Score from surprise magnitude
        surprise_score = np.clip(last_surprise / 20, -1, 1)  # ±20% = max

        # Proximity damping: reduce conviction near next earnings
        proximity_factor = 1.0
        if earnings_info and earnings_info.days_until_earnings is not None:
            days = earnings_info.days_until_earnings
            if days <= 3:
                proximity_factor = 0.2  # Very close to earnings - high uncertainty
            elif days <= 7:
                proximity_factor = 0.5
            elif days <= 14:
                proximity_factor = 0.8

        score = surprise_score * proximity_factor
        confidence = min(1.0, abs(last_surprise) / 15 * proximity_factor)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "last_surprise_pct": float(last_surprise),
                "avg_surprise_pct": float(avg_surprise),
                "days_to_earnings": earnings_info.days_until_earnings if earnings_info else None,
                "proximity_factor": float(proximity_factor),
            },
        )
