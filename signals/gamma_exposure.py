"""Signal 23: Gamma exposure estimate — aggregate gamma × OI across strikes."""

import numpy as np
from signals.base import Signal, SignalResult


class GammaExposureSignal(Signal):
    name = "gamma_exposure"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2
    # Option-chain refresh dictates daily cadence; gamma effects play out over days.
    return_resolution = "D"
    return_horizon = 3
    return_lookback_days = 60

    def compute(self, symbol: str, data: dict) -> SignalResult:
        option_chain = data.get("option_chain")
        quote = data.get("quote")

        if option_chain is None or not option_chain.contracts:
            return SignalResult(0.0, 0.0, {"error": "no option chain"})

        underlying = None
        if quote and quote.last:
            underlying = quote.last
        elif option_chain.contracts[0].underlying_price:
            underlying = option_chain.contracts[0].underlying_price

        if not underlying or underlying <= 0:
            return SignalResult(0.0, 0.0, {"error": "no underlying price"})

        # Compute net gamma exposure across all strikes
        total_gamma = 0.0
        for c in option_chain.contracts:
            if c.gamma is None or c.open_interest is None:
                continue
            # Calls: positive gamma for market makers when short; Puts: negative
            sign = 1.0 if c.side == 'call' else -1.0
            total_gamma += sign * c.gamma * c.open_interest * 100

        # Normalize by underlying price
        normalized_gamma = total_gamma / underlying if underlying > 0 else 0

        # Positive gamma = MM long gamma = dampens moves = stability
        # Negative gamma = MM short gamma = amplifies moves = instability
        # For directional bias: use the skew of gamma positioning
        call_gamma = sum(
            c.gamma * c.open_interest * 100
            for c in option_chain.calls()
            if c.gamma and c.open_interest
        )
        put_gamma = sum(
            c.gamma * c.open_interest * 100
            for c in option_chain.puts()
            if c.gamma and c.open_interest
        )

        # Directional bias from gamma skew
        total_abs_gamma = abs(call_gamma) + abs(put_gamma)
        if total_abs_gamma > 0:
            gamma_skew = (call_gamma - put_gamma) / total_abs_gamma
        else:
            gamma_skew = 0.0

        score = np.clip(gamma_skew, -1, 1)
        confidence = min(1.0, total_abs_gamma / (underlying * 10000))

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "net_gamma": float(normalized_gamma),
                "call_gamma": float(call_gamma),
                "put_gamma": float(put_gamma),
                "gamma_skew": float(gamma_skew),
            },
        )
