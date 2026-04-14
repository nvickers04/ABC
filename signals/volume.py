"""Signal 5: Volume profile — relative volume, volume trend, OBV direction."""

import numpy as np
from signals.base import Signal, SignalResult


class VolumeSignal(Signal):
    name = "volume_profile"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        if candles is None or len(candles) < 25:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)
        volume = np.array(candles.volume, dtype=float)

        # Relative volume (current bar vs 20-bar avg)
        avg_vol_20 = np.mean(volume[-20:])
        rel_vol = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # Volume trend (expanding vs contracting over 5 bars)
        if len(volume) >= 10:
            recent_avg = np.mean(volume[-5:])
            prior_avg = np.mean(volume[-10:-5])
            vol_trend = (recent_avg - prior_avg) / prior_avg if prior_avg > 0 else 0
        else:
            vol_trend = 0

        # On-balance volume direction
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # OBV slope (last 10 bars)
        if len(obv) >= 10:
            obv_slope = (obv[-1] - obv[-10]) / (np.std(obv[-10:]) + 1e-9)
            obv_score = np.clip(obv_slope / 3, -1, 1)
        else:
            obv_score = 0.0

        # Combine: direction from OBV, magnitude from relative volume
        vol_magnitude = rel_vol / 2.0  # Scale 2x avg = 1.0
        vol_trend_score = np.clip(vol_trend, -1, 1)

        # Accumulation (+) vs distribution (-)
        score = obv_score * 0.5 + vol_trend_score * 0.3 + np.clip(vol_magnitude - 1, -1, 1) * 0.2 * np.sign(obv_score)
        confidence = min(1.0, max(abs(obv_score), vol_magnitude / 3.0))

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "rel_volume": float(rel_vol),
                "vol_trend": float(vol_trend),
                "obv_score": float(obv_score),
            },
        )
