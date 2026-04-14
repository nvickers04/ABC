"""Signal 38: Correlation regime — cross-asset correlation from environment."""

import numpy as np
from signals.base import Signal, SignalResult


class CorrelationRegimeSignal(Signal):
    name = "correlation_regime"
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

        # Cross-asset correlation level
        cross_corr = env.get("cross_asset_correlation")
        if cross_corr is not None:
            # High correlation = risk-off herding, low = stock-picking regime
            # During high corr, signal is bearish (risk-off), low corr is neutral-to-bullish
            corr_score = np.clip(-(cross_corr - 0.5) * 2, -1, 1)
            scores.append(corr_score)
            components["cross_asset_correlation"] = float(cross_corr)

        # Dispersion — inverse of correlation
        dispersion = env.get("dispersion") or env.get("return_dispersion")
        if dispersion is not None:
            # High dispersion = good for stock-picking
            disp_score = np.clip((dispersion - 0.02) / 0.03, -1, 1)
            scores.append(disp_score * 0.5)
            components["return_dispersion"] = float(dispersion)

        # Correlation regime label from environment
        corr_regime = env.get("correlation_regime", "normal")
        regime_map = {
            "high_correlation": -0.5,
            "normal": 0.0,
            "low_correlation": 0.3,
            "divergent": 0.4,
        }
        regime_adj = regime_map.get(corr_regime, 0.0)
        components["correlation_regime_label"] = corr_regime

        if not scores:
            return SignalResult(
                float(np.clip(regime_adj, -1, 1)),
                0.3,
                components,
            )

        score = np.clip(np.mean(scores) + regime_adj * 0.2, -1, 1)
        confidence = min(1.0, abs(score) * 1.5)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
