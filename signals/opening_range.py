"""Signal 13: Opening range position — price vs first 30-min high/low."""

import numpy as np
from signals.base import Signal, SignalResult


class OpeningRangeSignal(Signal):
    name = "opening_range"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 35:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        high = np.array(candles.high, dtype=float)
        low = np.array(candles.low, dtype=float)
        close = np.array(candles.close, dtype=float)

        # Opening range = first 30 bars (at 1-min resolution = 30 min)
        # Adjust if resolution is different
        or_bars = min(30, len(high) // 3)
        if or_bars < 5:
            return SignalResult(0.0, 0.0, {"error": "not enough bars for opening range"})

        or_high = np.max(high[:or_bars])
        or_low = np.min(low[:or_bars])
        or_range = or_high - or_low

        if or_range < 1e-9:
            return SignalResult(0.0, 0.0, {"error": "zero opening range"})

        price = close[-1]

        if price > or_high:
            # Above opening range — bullish breakout
            dist = (price - or_high) / or_range
            score = min(1.0, dist * 0.5 + 0.5)
        elif price < or_low:
            # Below opening range — bearish breakdown
            dist = (or_low - price) / or_range
            score = max(-1.0, -(dist * 0.5 + 0.5))
        else:
            # Inside opening range
            pos = (price - or_low) / or_range  # 0 = bottom, 1 = top
            score = (pos - 0.5) * 0.6  # Mild directional bias

        # Confidence: higher when outside range with distance
        if price > or_high or price < or_low:
            dist = abs(price - (or_high if price > or_high else or_low)) / or_range
            confidence = min(1.0, 0.5 + dist * 0.3)
        else:
            confidence = 0.3  # Low confidence inside range

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "or_high": float(or_high),
                "or_low": float(or_low),
                "or_range": float(or_range),
                "price": float(price),
                "position": "above" if price > or_high else "below" if price < or_low else "inside",
            },
        )
