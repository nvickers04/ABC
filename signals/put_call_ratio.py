"""Signal 22: Put/call ratio — contrarian indicator from OI ratio."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength
from signals.base import Signal, SignalResult


class PutCallRatioSignal(Signal):
    name = "put_call_ratio"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        option_chain = data.get("option_chain")

        if option_chain is None or not option_chain.contracts:
            return SignalResult(0.0, 0.0, {"error": "no option chain"})

        total_put_oi = sum(c.open_interest for c in option_chain.puts() if c.open_interest)
        total_call_oi = sum(c.open_interest for c in option_chain.calls() if c.open_interest)

        if total_call_oi == 0:
            return SignalResult(0.0, 0.0, {"error": "zero call OI"})

        pc_ratio = total_put_oi / total_call_oi

        # Contrarian: extreme put/call = oversold (bullish), extreme call/put = overbought
        # Normal range: 0.7-1.3. Below 0.5 = extreme call = overbought. Above 1.5 = extreme put = oversold
        # Contrarian mapping around neutral ratio=1.
        score = bounded_tanh((pc_ratio - 1.0), scale=1.6)
        depth = min((total_put_oi + total_call_oi) / 50_000.0, 1.0)
        confidence = confidence_from_strength(abs(score), data_quality=depth)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "put_oi": int(total_put_oi),
                "call_oi": int(total_call_oi),
                "pc_ratio": float(pc_ratio),
            },
        )
