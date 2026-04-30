"""Signal 19: Realized vol percentile — current 20-day RV vs historical RV distribution."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength
from signals.base import Signal, SignalResult


class RealizedVolConeSignal(Signal):
    name = "realized_vol_cone"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles_daily = data.get("candles_daily")

        if candles_daily is None or len(candles_daily) < 60:
            return SignalResult(0.0, 0.0, {"error": "insufficient daily data"})

        daily_close = np.array(candles_daily.close, dtype=float)
        log_returns = np.diff(np.log(daily_close))

        # Current 20-day realized vol
        if len(log_returns) < 20:
            return SignalResult(0.0, 0.0, {"error": "insufficient returns for RV"})

        rv_20d = np.std(log_returns[-20:]) * np.sqrt(252) * 100

        # Historical RV distribution: rolling 20-day windows
        rv_history = []
        for i in range(20, len(log_returns)):
            window = log_returns[i - 20 : i]
            rv = np.std(window) * np.sqrt(252) * 100
            rv_history.append(rv)

        if len(rv_history) < 10:
            return SignalResult(0.0, 0.0, {"error": "insufficient RV history"})

        # Percentile rank
        rv_percentile = sum(1 for r in rv_history if r <= rv_20d) / len(rv_history) * 100

        # Low RV = calm, options cheap for buyers (+1)
        # High RV = volatile, risky (-1)
        score = bounded_tanh(-(rv_percentile - 50) / 50.0, scale=1.4)
        history_quality = min(1.0, len(rv_history) / 120.0)
        confidence = confidence_from_strength(abs(score), data_quality=history_quality)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "rv_20d": float(rv_20d),
                "rv_percentile": float(rv_percentile),
                "rv_history_len": len(rv_history),
            },
        )
