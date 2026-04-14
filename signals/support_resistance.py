"""Signal 12: Support/resistance proximity — distance to swing high/low."""

import numpy as np
from signals.base import Signal, SignalResult


def _find_swing_highs(high: np.ndarray, lookback: int = 5) -> list[float]:
    """Find local maxima with minimum lookback window on each side."""
    swings = []
    for i in range(lookback, len(high) - lookback):
        if high[i] == max(high[i - lookback : i + lookback + 1]):
            swings.append(float(high[i]))
    return swings


def _find_swing_lows(low: np.ndarray, lookback: int = 5) -> list[float]:
    """Find local minima with minimum lookback window on each side."""
    swings = []
    for i in range(lookback, len(low) - lookback):
        if low[i] == min(low[i - lookback : i + lookback + 1]):
            swings.append(float(low[i]))
    return swings


class SupportResistanceSignal(Signal):
    name = "support_resistance"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 30:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        high = np.array(candles.high, dtype=float)
        low = np.array(candles.low, dtype=float)
        close = np.array(candles.close, dtype=float)

        price = close[-1]

        # ATR for distance normalization
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 1e-9

        # Find nearest support and resistance
        swing_highs = _find_swing_highs(high[-30:])
        swing_lows = _find_swing_lows(low[-30:])

        # Nearest resistance above current price
        resistances = [s for s in swing_highs if s > price]
        nearest_resistance = min(resistances) if resistances else price + atr * 3

        # Nearest support below current price
        supports = [s for s in swing_lows if s < price]
        nearest_support = max(supports) if supports else price - atr * 3

        # Distance in ATR units
        dist_to_resistance = (nearest_resistance - price) / atr
        dist_to_support = (price - nearest_support) / atr

        # Score: near support = bullish (bounce), near resistance = bearish
        if dist_to_support < dist_to_resistance:
            # Closer to support = bullish
            score = np.clip(1.0 - dist_to_support / 2.0, 0, 1)
        else:
            # Closer to resistance = bearish
            score = np.clip(-(1.0 - dist_to_resistance / 2.0), -1, 0)

        min_dist = min(dist_to_support, dist_to_resistance)
        confidence = min(1.0, max(0.0, 1.0 - min_dist / 3.0))

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "dist_to_resistance_atr": float(dist_to_resistance),
                "dist_to_support_atr": float(dist_to_support),
                "nearest_resistance": float(nearest_resistance),
                "nearest_support": float(nearest_support),
                "n_swing_highs": len(swing_highs),
                "n_swing_lows": len(swing_lows),
            },
        )
