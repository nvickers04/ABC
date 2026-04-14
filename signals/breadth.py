"""Signal 36: Market breadth — advance/decline ratio from environment."""

import numpy as np
from signals.base import Signal, SignalResult


class BreadthSignal(Signal):
    name = "market_breadth"
    category = "macro"
    data_source = "environment"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        env = data.get("environment")

        if env is None:
            return SignalResult(0.0, 0.0, {"error": "no environment data"})

        # Advance/decline ratio from environment snapshot
        ad_ratio = env.get("advance_decline_ratio")
        pct_up = env.get("pct_trending_up", 0.5)
        pct_down = env.get("pct_trending_down", 0.5)
        breadth_regime = env.get("breadth_regime", "neutral")

        components = {}
        scores = []

        if ad_ratio is not None:
            # A/D > 1 = broad bullish, < 1 = broad bearish
            ad_score = np.clip((ad_ratio - 1.0) / 2.0, -1, 1)
            scores.append(ad_score)
            components["ad_ratio"] = float(ad_ratio)

        if pct_up is not None and pct_down is not None:
            net_breadth = float(pct_up) - float(pct_down)
            breadth_score = np.clip(net_breadth * 2, -1, 1)
            scores.append(breadth_score)
            components["pct_up"] = float(pct_up)
            components["pct_down"] = float(pct_down)

        # Regime as additional input
        regime_map = {"bullish": 0.3, "neutral": 0.0, "bearish": -0.3}
        regime_adj = regime_map.get(breadth_regime, 0.0)
        components["breadth_regime"] = breadth_regime

        if not scores:
            return SignalResult(regime_adj, 0.3, components)

        score = np.clip(np.mean(scores) + regime_adj * 0.3, -1, 1)
        confidence = min(1.0, abs(score) * 1.5)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
