"""Signal 14: IV vs realized vol spread — options overpriced/underpriced."""

import numpy as np
from signals.base import Signal, SignalResult


class IVRVSpreadSignal(Signal):
    name = "iv_rv_spread"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        iv_info = data.get("iv_info")
        candles_daily = data.get("candles_daily")

        if iv_info is None or iv_info.iv_current is None:
            return SignalResult(0.0, 0.0, {"error": "no IV data"})

        iv_current = iv_info.iv_current

        # Compute realized vol from daily candle returns (20-day)
        if candles_daily is not None and len(candles_daily) >= 21:
            daily_close = np.array(candles_daily.close, dtype=float)
            returns = np.diff(np.log(daily_close[-21:]))
            rv_20d = np.std(returns) * np.sqrt(252) * 100  # Annualized %
        else:
            return SignalResult(0.0, 0.0, {"error": "insufficient daily data for RV"})

        # Spread = IV - RV
        spread = iv_current - rv_20d

        # Positive spread = options overpriced (sell premium edge)
        # Negative spread = options underpriced (buy options edge)
        score = np.clip(spread / 20, -1, 1)  # 20% spread = max

        confidence = min(1.0, abs(spread) / 15.0)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "iv_current": float(iv_current),
                "rv_20d": float(rv_20d),
                "spread": float(spread),
            },
        )
