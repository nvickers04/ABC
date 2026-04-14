"""Signal 42: Relative strength vs market — symbol returns vs SPY/QQQ."""

import numpy as np
from signals.base import Signal, SignalResult


class RelativeStrengthMarketSignal(Signal):
    name = "relative_strength_market"
    category = "macro"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")
        spy_candles = data.get("spy_candles")

        if candles is None or spy_candles is None:
            return SignalResult(0.0, 0.0, {"error": "missing candles"})

        components = {}
        scores = []

        try:
            sym_close = np.array([c.close for c in candles if c.close], dtype=float)
            spy_close = np.array([c.close for c in spy_candles if c.close], dtype=float)
        except (AttributeError, TypeError):
            return SignalResult(0.0, 0.0, {"error": "invalid candle data"})

        min_len = min(len(sym_close), len(spy_close))
        if min_len < 6:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        sym_close = sym_close[-min_len:]
        spy_close = spy_close[-min_len:]

        # Relative strength over multiple horizons
        for period, weight in [(5, 0.5), (10, 0.3), (20, 0.2)]:
            if min_len <= period:
                continue
            sym_ret = (sym_close[-1] / sym_close[-period - 1]) - 1.0
            spy_ret = (spy_close[-1] / spy_close[-period - 1]) - 1.0
            excess = sym_ret - spy_ret
            rs_score = np.clip(excess / 0.05, -1, 1)  # ±5% excess = ±1
            scores.append(rs_score * weight)
            components[f"excess_{period}d"] = float(round(excess * 100, 2))

        if not scores:
            return SignalResult(0.0, 0.0, components)

        total_weight = sum(
            w for p, w in [(5, 0.5), (10, 0.3), (20, 0.2)] if min_len > p
        )
        score = np.clip(sum(scores) / max(total_weight, 1e-9), -1, 1)
        confidence = min(1.0, len(scores) / 3.0 * 0.8)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
