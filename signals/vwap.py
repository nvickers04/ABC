"""Signal 4: VWAP deviation — price distance from VWAP in ATR units."""

import numpy as np
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
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 1e-9

        # Distance from VWAP in ATR units
        vwap_dist = (close[-1] - vwap[-1]) / atr if atr > 0 else 0
        score = np.clip(vwap_dist / 3.0, -1, 1)  # 3 ATR = max

        # VWAP slope direction (momentum)
        if len(vwap) >= 5:
            vwap_slope = (vwap[-1] - vwap[-5]) / atr if atr > 0 else 0
            slope_component = np.clip(vwap_slope, -1, 1)
        else:
            slope_component = 0.0

        score = score * 0.7 + slope_component * 0.3
        confidence = min(1.0, abs(score) * 1.5)

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
