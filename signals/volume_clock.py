"""Signal 49: Volume clock — intraday volume acceleration vs expected profile."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength
from signals.base import Signal, SignalResult


class VolumeClockSignal(Signal):
    name = "volume_clock"
    category = "microstructure"
    data_source = "mda_quotes"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        quote = data.get("quote")
        candles = data.get("candles")

        if quote is None or candles is None:
            return SignalResult(0.0, 0.0, {"error": "missing data"})

        components = {}

        # Current session volume from quote
        current_vol = getattr(quote, "volume", None)
        if current_vol is None or current_vol <= 0:
            return SignalResult(0.0, 0.0, {"error": "no current volume"})

        # Historical average daily volume from candles
        try:
            hist_vols = [
                c.volume for c in candles[-20:] if c.volume and c.volume > 0
            ]
        except (AttributeError, TypeError):
            return SignalResult(0.0, 0.0, {"error": "invalid candle data"})

        if len(hist_vols) < 5:
            return SignalResult(0.0, 0.0, {"error": "insufficient volume history"})

        avg_daily_vol = np.mean(hist_vols)
        components["current_volume"] = int(current_vol)
        components["avg_daily_volume"] = int(avg_daily_vol)
        components["mode"] = "proxy_intraday_participation"

        # Volume ratio: current vs average full-day volume
        vol_ratio = current_vol / max(avg_daily_vol, 1)
        components["volume_ratio"] = float(round(vol_ratio, 2))

        # When current volume already exceeds daily average, there's conviction
        # When volume is lagging, it's a quiet/uninterested day
        score = bounded_tanh(vol_ratio - 0.8, scale=1.2)

        # Compare recent volume trend
        if len(hist_vols) >= 10:
            recent_avg = np.mean(hist_vols[-5:])
            older_avg = np.mean(hist_vols[-10:-5])
            vol_trend = (recent_avg / max(older_avg, 1)) - 1.0
            components["vol_trend_5d"] = float(round(vol_trend * 100, 1))
            # Rising volume trend is bullish confirmation
            if vol_trend > 0.2:
                score = min(1.0, score + 0.15)
            elif vol_trend < -0.2:
                score = max(-1.0, score - 0.15)

        confidence = confidence_from_strength(
            abs(score),
            data_quality=min(1.0, len(hist_vols) / 12.0),
        )

        return SignalResult(
            score=float(np.clip(score, -1, 1)),
            confidence=float(confidence),
            components=components,
        )
