"""Signal 35: Institutional ownership — level + accumulation/distribution."""

import numpy as np
from signals.base import Signal, SignalResult


class InstitutionalSignal(Signal):
    name = "institutional_ownership"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "10min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        institutional = data.get("institutional")

        if institutional is None:
            return SignalResult(0.0, 0.0, {"error": "no institutional data"})

        inst_pct = getattr(institutional, "institutional_pct", None)

        if inst_pct is None:
            return SignalResult(0.0, 0.0, {"error": "no institutional %"})

        # High institutional ownership with accumulation = bullish
        # Typical growth stock: 50-80% institutional
        inst_level_score = np.clip((float(inst_pct) - 0.5) / 0.4, -1, 1)

        # Use number of top holders as breadth indicator
        holders = getattr(institutional, "top_holders", []) or []
        breadth_score = np.clip(len(holders) / 10 - 0.5, -0.5, 0.5)

        score = inst_level_score * 0.7 + breadth_score * 0.3
        score = np.clip(score, -1, 1)
        confidence = 0.4  # Institutional data is slow-moving, moderate confidence

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "institutional_pct": float(inst_pct),
                "n_holders": len(holders),
            },
        )
