"""Signal 4: VWAP deviation — price distance from VWAP in ATR units."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength, rolling_atr
from signals.base import Signal, SignalResult


class VWAPSignal(Signal):
    name = "vwap_deviation"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 20:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)
        high = np.array(candles.high, dtype=float)
        low = np.array(candles.low, dtype=float)
        volume = np.array(candles.volume, dtype=float)

        # Compute VWAP (typical price * volume cumulative)
        typical = (high + low + close) / 3.0
        cum_vol = np.cumsum(volume)
        cum_tp_vol = np.cumsum(typical * volume)
        vwap = cum_tp_vol / np.where(cum_vol > 0, cum_vol, 1)

        # ATR for normalization
        atr = rolling_atr(high, low, close, period=14)

        # Distance from VWAP in ATR units
        vwap_dist = (close[-1] - vwap[-1]) / atr if atr > 0 else 0
        score = bounded_tanh(vwap_dist, scale=0.7)

        # VWAP slope direction (momentum)
        if len(vwap) >= 5:
            vwap_slope = (vwap[-1] - vwap[-5]) / atr if atr > 0 else 0
            slope_component = bounded_tanh(vwap_slope, scale=0.9)
        else:
            slope_component = 0.0

        score = bounded_tanh(score * 0.7 + slope_component * 0.3, scale=1.1)
        data_quality = min(1.0, len(close) / 80.0)
        confidence = confidence_from_strength(abs(score), data_quality=data_quality)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "vwap_dist_atr": float(vwap_dist),
                "vwap_slope": float(slope_component),
                "vwap": float(vwap[-1]),
                "price": float(close[-1]),
            },
        )
