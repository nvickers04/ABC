"""Signal 3: Breakout — distance from highs/lows, volume confirmation, range expansion."""

import numpy as np
from signals.base import Signal, SignalResult


class BreakoutSignal(Signal):
    name = "breakout"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 25:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)
        high = np.array(candles.high, dtype=float)
        low = np.array(candles.low, dtype=float)
        volume = np.array(candles.volume, dtype=float)

        # Distance from 20-bar high/low
        high_20 = np.max(high[-20:])
        low_20 = np.min(low[-20:])
        rng = high_20 - low_20
        if rng < 1e-9:
            return SignalResult(0.0, 0.0, {"error": "no range"})

        pos_in_range = (close[-1] - low_20) / rng  # 0 = at low, 1 = at high
        # Near high = bullish breakout, near low = bearish breakdown
        direction_score = (pos_in_range - 0.5) * 2  # -1 to +1

        # Volume confirmation
        avg_vol_20 = np.mean(volume[-20:])
        rel_vol = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0
        vol_confirm = np.clip((rel_vol - 1.0) / 2.0, 0, 1)  # 0 at avg, 1 at 3x

        # Range expansion (today's range vs 5-day avg range)
        daily_ranges = high[-5:] - low[-5:]
        avg_range = np.mean(daily_ranges[:-1]) if len(daily_ranges) > 1 else daily_ranges[0]
        current_range = high[-1] - low[-1]
        range_expansion = (current_range / avg_range - 1.0) if avg_range > 0 else 0
        range_score = np.clip(range_expansion, 0, 1)

        # Combine: direction + volume + range expansion
        score = direction_score * (0.5 + vol_confirm * 0.25 + range_score * 0.25)
        confidence = min(1.0, vol_confirm * 0.5 + range_score * 0.5)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "pos_in_range": float(pos_in_range),
                "rel_volume": float(rel_vol),
                "range_expansion": float(range_expansion),
            },
        )
