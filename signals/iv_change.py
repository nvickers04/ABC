"""Signal 18: IV rank momentum — change in IV rank over trailing period."""

import numpy as np
from signals.base import Signal, SignalResult


class IVChangeSignal(Signal):
    name = "iv_rank_momentum"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        iv_info = data.get("iv_info")

        if iv_info is None or iv_info.iv_current is None:
            return SignalResult(0.0, 0.0, {"error": "no IV data"})

        # Use iv_high and iv_low to estimate IV regime change
        # Without historical IV snapshots, we approximate direction from
        # where IV sits relative to its range
        iv_current = iv_info.iv_current
        iv_high = iv_info.iv_high
        iv_low = iv_info.iv_low
        iv_rank = iv_info.iv_rank

        if iv_rank is None or iv_high is None or iv_low is None:
            return SignalResult(0.0, 0.0, {"error": "incomplete IV data"})

        # IV falling = fear dissipating (+1), IV rising = fear building (-1)
        # Use position in range as proxy: closer to high = recently risen
        # We infer momentum from where rank sits and the range spread
        range_pct = (iv_high - iv_low) / iv_high * 100 if iv_high > 0 else 0

        # Higher rank with wider range = more room to fall back = sell premium
        # Lower rank = IV compressed = may expand
        if iv_rank > 70:
            score = 0.5  # High IV, likely to mean-revert down
        elif iv_rank < 30:
            score = -0.5  # Low IV, may expand
        else:
            score = 0.0

        # Adjust for range width
        if range_pct > 50:
            score *= 1.3  # Wide range = more conviction

        score = np.clip(score, -1, 1)
        confidence = min(1.0, abs(score) * 0.8 + range_pct / 100 * 0.3)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "iv_rank": float(iv_rank),
                "iv_current": float(iv_current),
                "range_pct": float(range_pct),
            },
        )
