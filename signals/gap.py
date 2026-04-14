"""Signal 7: Overnight gap — gap direction + partial fill assessment."""

import numpy as np
from signals.base import Signal, SignalResult


class GapSignal(Signal):
    name = "overnight_gap"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        candles_daily = data.get("candles_daily")

        if candles is None or len(candles) < 10:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)
        open_ = np.array(candles.open, dtype=float)
        high = np.array(candles.high, dtype=float)
        low = np.array(candles.low, dtype=float)

        # Gap from daily candles (prev close vs today open)
        if candles_daily is not None and len(candles_daily) >= 2:
            prev_close = np.array(candles_daily.close, dtype=float)[-2]
            today_open = np.array(candles_daily.open, dtype=float)[-1]
            gap_pct = (today_open - prev_close) / prev_close
        else:
            # Fallback: use intraday open vs first bar
            gap_pct = 0.0
            return SignalResult(0.0, 0.0, {"error": "no daily data for gap calc"})

        # Gap size for scoring
        if abs(gap_pct) < 0.001:  # Less than 0.1% gap
            return SignalResult(0.0, 0.1, {"gap_pct": float(gap_pct), "note": "negligible gap"})

        # Partial fill assessment: how much of the gap has been filled intraday?
        current_price = close[-1]
        if gap_pct > 0:  # Gap up
            fill_pct = max(0, (today_open - current_price) / (today_open - prev_close))
        else:  # Gap down
            fill_pct = max(0, (current_price - today_open) / (prev_close - today_open))
        fill_pct = min(fill_pct, 1.0)

        # Holding gap (not filled) = continuation; filled gap = reversal
        if fill_pct < 0.3:
            # Gap holding = bullish for gap ups, bearish for gap downs
            score = np.clip(gap_pct * 20, -1, 1)
        elif fill_pct > 0.7:
            # Gap filled = fading, slight reversal signal
            score = np.clip(-gap_pct * 10, -1, 1)
        else:
            # Partial fill, uncertain
            score = np.clip(gap_pct * 5, -1, 1)

        confidence = min(1.0, abs(gap_pct) * 30 * (1 - fill_pct * 0.5))

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "gap_pct": float(gap_pct),
                "fill_pct": float(fill_pct),
                "prev_close": float(prev_close),
                "today_open": float(today_open),
            },
        )
