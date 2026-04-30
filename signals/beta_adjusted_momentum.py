"""Signal 6: Beta-adjusted momentum — idiosyncratic alpha vs SPY."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength, safe_pct_change
from signals.base import Signal, SignalResult


class BetaAdjustedMomentumSignal(Signal):
    name = "beta_adjusted_momentum"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        spy_candles = data.get("spy_candles")
        fundamentals = data.get("fundamentals")

        if candles is None or len(candles) < 25:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)

        # Get beta from fundamentals; default to 1.0
        beta = 1.0
        if fundamentals and hasattr(fundamentals, "beta") and fundamentals.beta:
            beta = fundamentals.beta

        # SPY returns
        spy_returns_5 = 0.0
        spy_returns_10 = 0.0
        spy_returns_20 = 0.0
        if spy_candles is not None and len(spy_candles) >= 25:
            spy_close = np.array(spy_candles.close, dtype=float)
            spy_returns_5 = safe_pct_change(spy_close[-1], spy_close[-6])
            spy_returns_10 = safe_pct_change(spy_close[-1], spy_close[-11])
            spy_returns_20 = safe_pct_change(spy_close[-1], spy_close[-21]) if len(spy_close) > 20 else spy_returns_10

        # Symbol returns
        returns_5 = safe_pct_change(close[-1], close[-6]) if len(close) > 5 else 0.0
        returns_10 = safe_pct_change(close[-1], close[-11]) if len(close) > 10 else 0.0
        returns_20 = safe_pct_change(close[-1], close[-21]) if len(close) > 20 else returns_10

        # Alpha = symbol return minus (beta x SPY return)
        alpha_5 = returns_5 - beta * spy_returns_5
        alpha_10 = returns_10 - beta * spy_returns_10
        alpha_20 = returns_20 - beta * spy_returns_20

        # Weighted average alpha
        avg_alpha = alpha_5 * 0.5 + alpha_10 * 0.3 + alpha_20 * 0.2
        score = bounded_tanh(avg_alpha, scale=18.0)
        data_quality = min(1.0, len(close) / 90.0) * (1.0 if spy_candles is not None else 0.6)
        confidence = confidence_from_strength(abs(score), data_quality=data_quality)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "alpha_5": float(alpha_5),
                "alpha_10": float(alpha_10),
                "alpha_20": float(alpha_20),
                "beta": float(beta),
            },
        )
