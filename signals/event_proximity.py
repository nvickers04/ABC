"""Signal 39: Event proximity — distance to FOMC, CPI, NFP events."""

import numpy as np
from signals.base import Signal, SignalResult


class EventProximitySignal(Signal):
    name = "event_proximity"
    category = "macro"
    data_source = "environment"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        env = data.get("environment")

        if env is None:
            return SignalResult(0.0, 0.0, {"error": "no environment data"})

        components = {}
        urgency_scores = []

        # Days to next macro events from environment
        events = {
            "days_to_fomc": env.get("days_to_fomc"),
            "days_to_cpi": env.get("days_to_cpi"),
            "days_to_nfp": env.get("days_to_nfp"),
        }

        for key, days in events.items():
            if days is not None:
                days = float(days)
                components[key] = days
                # Closer events = more caution (negative)
                # 0-1 days = very cautious, 5+ days = negligible
                if days <= 1:
                    urgency_scores.append(-0.8)
                elif days <= 3:
                    urgency_scores.append(-0.4)
                elif days <= 5:
                    urgency_scores.append(-0.1)
                else:
                    urgency_scores.append(0.0)

        # Earnings proximity for this specific symbol
        earnings = data.get("earnings")
        if earnings is not None:
            days_to_earnings = getattr(earnings, "days_until_earnings", None)
            if days_to_earnings is not None:
                days_to_earnings = float(days_to_earnings)
                components["days_to_earnings"] = days_to_earnings
                if days_to_earnings <= 2:
                    urgency_scores.append(-0.6)
                elif days_to_earnings <= 5:
                    urgency_scores.append(-0.2)
                else:
                    urgency_scores.append(0.0)

        if not urgency_scores:
            return SignalResult(0.0, 0.0, components)

        # Score is cautionary: negative near events, zero otherwise
        score = np.clip(np.min(urgency_scores), -1, 1)
        # Confidence high near events (we're certain about being cautious)
        confidence = min(1.0, abs(score) * 1.5) if score < -0.1 else 0.3

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
