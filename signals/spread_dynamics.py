"""Signal 46: Bid-ask spread dynamics — spread vs trailing average."""

import numpy as np
from signals.base import Signal, SignalResult


class SpreadDynamicsSignal(Signal):
    name = "spread_dynamics"
    category = "microstructure"
    data_source = "mda_quotes"
    refresh_rate = "every_round"
    tier = 1

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
        if spread_ratio <= 0.5:
            score = 0.8  # Very tight, healthy
        elif spread_ratio <= 1.0:
            score = 0.3  # Normal
        elif spread_ratio <= 2.0:
            score = -0.3  # Wider than usual
        else:
            score = -0.8  # Abnormally wide

        confidence = min(1.0, 0.6 + abs(score) * 0.3)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
