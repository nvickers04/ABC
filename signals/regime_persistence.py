"""Signal 44: Regime persistence — regime duration and mean-reversion risk."""

import numpy as np
from signals.base import Signal, SignalResult


class RegimePersistenceSignal(Signal):
    name = "regime_persistence"
    category = "macro"
    data_source = "environment"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        env = data.get("environment")

        if env is None:
            return SignalResult(0.0, 0.0, {"error": "no environment data"})

        components = {}
        scores = []

        # Volatility regime duration
        vol_regime = env.get("volatility_regime", "normal")
        vol_regime_duration = env.get("vol_regime_duration")
        components["vol_regime"] = vol_regime

        if vol_regime_duration is not None:
            vol_regime_duration = float(vol_regime_duration)
            components["vol_regime_duration"] = vol_regime_duration

            # Fresh regime (< 3 periods): trade with it
            # Extended regime (> 10 periods): mean-reversion risk increases
            if vol_regime_duration <= 3:
                persistence_score = 0.5  # Fresh regime, likely to persist
            elif vol_regime_duration <= 10:
                persistence_score = 0.2  # Moderate
            else:
                persistence_score = -0.3  # Extended, reversal risk

            # Adjust sign based on regime direction
            if vol_regime in ("low", "low_vol"):
                scores.append(persistence_score)  # Low vol persisting is calm
            elif vol_regime in ("high", "high_vol"):
                scores.append(-persistence_score)  # High vol persisting = eventual relief
            else:
                scores.append(persistence_score * 0.3)

        # Trend regime duration
        trend_regime = env.get("trend_regime", "neutral")
        trend_duration = env.get("trend_regime_duration")
        components["trend_regime"] = trend_regime

        if trend_duration is not None:
            trend_duration = float(trend_duration)
            components["trend_regime_duration"] = trend_duration

            if trend_duration <= 3:
                t_score = 0.5  # Fresh
            elif trend_duration <= 10:
                t_score = 0.2
            else:
                t_score = -0.3  # Overextended

            if trend_regime in ("up", "bullish"):
                scores.append(t_score)
            elif trend_regime in ("down", "bearish"):
                scores.append(-t_score)
            else:
                scores.append(0.0)

        if not scores:
            return SignalResult(0.0, 0.2, components)

        score = np.clip(np.mean(scores), -1, 1)
        confidence = min(1.0, abs(score) * 1.5 + 0.2)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
