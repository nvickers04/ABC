"""Signal 48: Quote stability — bid/ask mid-price variance measure."""

import numpy as np
from signals.base import Signal, SignalResult


class QuoteStabilitySignal(Signal):
    name = "quote_stability"
    category = "microstructure"
    data_source = "mda_quotes"
    refresh_rate = "every_round"
    tier = 1
    # True tick-level: 1-min bars, 30 bars (=30 min) ahead, 2 days history.
    return_resolution = "1min"
    return_horizon = 30
    return_lookback_days = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        quote = data.get("quote")
        candles = data.get("candles")

        if candles is None or len(candles) < 5:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        components = {}

        try:
            highs = np.array([c.high for c in candles[-20:] if c.high], dtype=float)
            lows = np.array([c.low for c in candles[-20:] if c.low], dtype=float)
            closes = np.array([c.close for c in candles[-20:] if c.close], dtype=float)
        except (AttributeError, TypeError):
            return SignalResult(0.0, 0.0, {"error": "invalid candle data"})

        if len(closes) < 5:
            return SignalResult(0.0, 0.0, {"error": "insufficient closes"})

        # Intrabar range as proxy for quote instability
        ranges = (highs[:len(lows)] - lows[:len(highs)]) / np.maximum(closes[:min(len(highs), len(lows))], 1e-9)
        avg_range = np.mean(ranges) if len(ranges) > 0 else 0

        # Recent vs trailing range comparison
        if len(ranges) >= 10:
            recent_range = np.mean(ranges[-5:])
            trailing_range = np.mean(ranges[-10:-5])
            range_change = (recent_range / max(trailing_range, 1e-9)) - 1.0
        else:
            recent_range = avg_range
            range_change = 0.0

        components["avg_range_pct"] = float(round(avg_range * 100, 3))
        components["range_change"] = float(round(range_change * 100, 2))

        # Close-to-close returns volatility
        returns = np.diff(closes) / closes[:-1]
        if len(returns) > 0:
            ret_std = np.std(returns)
            components["return_std"] = float(round(ret_std * 100, 3))
        else:
            ret_std = 0

        # Stable = low range + narrowing ranges = positive
        # Unstable = high range + widening ranges = negative
        stability_score = 0.0

        # Range level scoring
        if avg_range < 0.01:  # < 1% avg range = very stable
            stability_score += 0.5
        elif avg_range < 0.02:
            stability_score += 0.2
        elif avg_range < 0.04:
            stability_score -= 0.2
        else:
            stability_score -= 0.5

        # Range trend scoring
        if range_change < -0.1:  # Ranges narrowing
            stability_score += 0.3
        elif range_change > 0.2:  # Ranges expanding
            stability_score -= 0.3

        score = np.clip(stability_score, -1, 1)
        confidence = min(1.0, len(ranges) / 15.0 * 0.7)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
