"""Signal 22: Put/call ratio — contrarian indicator from OI ratio."""

import numpy as np
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
        if pc_ratio > 1.5:
            score = min(1.0, (pc_ratio - 1.0) / 1.0)  # Contrarian bullish
        elif pc_ratio < 0.5:
            score = max(-1.0, -(1.0 - pc_ratio) / 0.5)  # Contrarian bearish
        else:
            score = (pc_ratio - 1.0) * 0.5  # Mild signal in normal range

        confidence = min(1.0, abs(pc_ratio - 1.0) / 0.8)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "put_oi": int(total_put_oi),
                "call_oi": int(total_call_oi),
                "pc_ratio": float(pc_ratio),
            },
        )
