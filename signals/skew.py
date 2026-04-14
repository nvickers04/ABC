"""Signal 16: Volatility skew — put IV vs call IV from option chain."""

import numpy as np
from signals.base import Signal, SignalResult


class SkewSignal(Signal):
    name = "volatility_skew"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        option_chain = data.get("option_chain")

        if option_chain is None or not option_chain.contracts:
            return SignalResult(0.0, 0.0, {"error": "no option chain data"})

        # Filter ATM options (delta ~0.40-0.60)
        atm_calls = [
            c for c in option_chain.calls()
            if c.delta and 0.35 <= abs(c.delta) <= 0.65 and c.iv and c.iv > 0
        ]
        atm_puts = [
            c for c in option_chain.puts()
            if c.delta and 0.35 <= abs(c.delta) <= 0.65 and c.iv and c.iv > 0
        ]

        if not atm_calls or not atm_puts:
            return SignalResult(0.0, 0.0, {"error": "no ATM options for skew"})

        avg_call_iv = np.mean([c.iv for c in atm_calls])
        avg_put_iv = np.mean([c.iv for c in atm_puts])

        # Skew = put IV - call IV
        # Positive skew = put IV > call IV = bearish hedging
        skew = avg_put_iv - avg_call_iv

        # Call skew = bullish, put skew = bearish
        score = np.clip(-skew / 10, -1, 1)  # 10% skew = max

        confidence = min(1.0, abs(skew) / 8.0)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "avg_call_iv": float(avg_call_iv),
                "avg_put_iv": float(avg_put_iv),
                "skew": float(skew),
                "n_atm_calls": len(atm_calls),
                "n_atm_puts": len(atm_puts),
            },
        )
