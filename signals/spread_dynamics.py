"""Signal 46: Bid-ask spread dynamics — spread vs trailing average."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength
from signals.base import Signal, SignalResult


class SpreadDynamicsSignal(Signal):
    name = "spread_dynamics"
    category = "microstructure"
    data_source = "mda_quotes"
    refresh_rate = "every_round"
    tier = 1
    # True tick-level: 1-min bars, 30 bars (=30 min) ahead, 2 days history.
    return_resolution = "1min"
    return_horizon = 30
    return_lookback_days = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        quote = data.get("quote")

        if quote is None:
            return SignalResult(0.0, 0.0, {"error": "no quote data"})

        components = {}

        bid = getattr(quote, "bid", None)
        ask = getattr(quote, "ask", None)
        last = getattr(quote, "last", None) or getattr(quote, "mid", None)

        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return SignalResult(0.0, 0.0, {"error": "no bid/ask"})

        mid = (bid + ask) / 2.0
        if mid <= 0:
            return SignalResult(0.0, 0.0, {"error": "invalid mid"})

        spread = ask - bid
        spread_pct = spread / mid * 100

        components["bid"] = float(bid)
        components["ask"] = float(ask)
        components["mode"] = "proxy_quoted_spread"
        components["spread"] = float(round(spread, 4))
        components["spread_pct"] = float(round(spread_pct, 4))

        # Compare to typical spread for this price level
        # Rough heuristic: stocks > $50 typically have < 0.05% spread
        # Penny stocks have wider spreads
        if last and last > 0:
            price = float(last)
        else:
            price = mid

        # Expected spread decreases with price level
        if price > 100:
            expected_spread_pct = 0.03
        elif price > 50:
            expected_spread_pct = 0.05
        elif price > 20:
            expected_spread_pct = 0.10
        elif price > 5:
            expected_spread_pct = 0.20
        else:
            expected_spread_pct = 0.50

        spread_ratio = spread_pct / expected_spread_pct if expected_spread_pct > 0 else 1.0
        components["spread_ratio_vs_expected"] = float(round(spread_ratio, 2))

        # Tightening spread (< expected) = safe, healthy market
        # Widening spread (> expected) = informed trading, caution
        score = bounded_tanh(-(spread_ratio - 1.0), scale=1.6)
        confidence = confidence_from_strength(
            abs(score),
            data_quality=1.0 if (bid > 0 and ask > 0) else 0.0,
        )

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
