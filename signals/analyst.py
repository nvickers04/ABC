"""Signal 29: Analyst consensus — recommendations, target upside."""

import numpy as np
from signals.base import Signal, SignalResult


class AnalystSignal(Signal):
    name = "analyst_consensus"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        analyst = data.get("analyst")

        if analyst is None:
            return SignalResult(0.0, 0.0, {"error": "no analyst data"})

        components = {}
        scores = []

        # Mean recommendation (1=strong buy, 5=sell)
        rec_mean = getattr(analyst, "recommendation_mean", None)
        if rec_mean is not None:
            rec_score = np.clip((3 - rec_mean) / 2, -1, 1)  # 1=+1, 3=0, 5=-1
            scores.append(rec_score * 0.4)
            components["rec_mean"] = float(rec_mean)

        # Target upside %
        upside = getattr(analyst, "upside_pct", None)
        if upside is not None:
            upside_score = np.clip(upside / 30, -1, 1)  # 30% upside = max
            scores.append(upside_score * 0.4)
            components["upside_pct"] = float(upside)

        # Upgrades vs downgrades
        upgrades = getattr(analyst, "recent_upgrades", 0)
        downgrades = getattr(analyst, "recent_downgrades", 0)
        if upgrades + downgrades > 0:
            change_ratio = (upgrades - downgrades) / (upgrades + downgrades)
            scores.append(change_ratio * 0.2)
            components["upgrades"] = upgrades
            components["downgrades"] = downgrades

        if not scores:
            return SignalResult(0.0, 0.0, {"error": "no analyst metrics"})

        score = np.clip(sum(scores), -1, 1)
        num_analysts = getattr(analyst, "num_analysts", 0) or 0
        confidence = min(1.0, num_analysts / 10 * 0.5 + abs(score) * 0.5)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
