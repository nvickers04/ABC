"""Signal 11: Trend strength — DMI (+DI vs -DI) and ADX-style measure."""

import numpy as np
from signals.base import Signal, SignalResult


class TrendStrengthSignal(Signal):
    name = "trend_strength"
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
        period = 14

        # True range
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # +DM and -DM
        plus_dm = np.zeros(len(close))
        minus_dm = np.zeros(len(close))
        for i in range(1, len(close)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth with Wilder's method (same as EMA with alpha=1/period)
        alpha = 1.0 / period
        atr = np.zeros(len(close))
        plus_di_raw = np.zeros(len(close))
        minus_di_raw = np.zeros(len(close))

        atr[period] = np.mean(tr[1:period + 1])
        plus_di_raw[period] = np.mean(plus_dm[1:period + 1])
        minus_di_raw[period] = np.mean(minus_dm[1:period + 1])

        for i in range(period + 1, len(close)):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
            plus_di_raw[i] = plus_di_raw[i - 1] * (1 - alpha) + plus_dm[i] * alpha
            minus_di_raw[i] = minus_di_raw[i - 1] * (1 - alpha) + minus_dm[i] * alpha

        # +DI and -DI as percentages
        atr_val = atr[-1]
        if atr_val <= 0:
            return SignalResult(0.0, 0.0, {"error": "zero ATR"})

        plus_di = (plus_di_raw[-1] / atr_val) * 100
        minus_di = (minus_di_raw[-1] / atr_val) * 100

        # ADX approximation
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = abs(plus_di - minus_di) / di_sum * 100
        else:
            dx = 0

        # Direction from +DI vs -DI
        direction = 1.0 if plus_di > minus_di else -1.0

        # Score: direction * strength
        strength = min(dx / 40.0, 1.0)  # ADX 40+ = max strength
        score = direction * strength

        # Confidence from ADX magnitude
        confidence = min(1.0, dx / 50.0)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "plus_di": float(plus_di),
                "minus_di": float(minus_di),
                "dx": float(dx),
                "direction": float(direction),
            },
        )
