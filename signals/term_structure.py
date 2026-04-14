"""Signal 17: IV term structure — front-month vs second-month IV."""

import numpy as np
from signals.base import Signal, SignalResult


class TermStructureSignal(Signal):
    name = "term_structure"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        option_chain = data.get("option_chain")

        if option_chain is None or not option_chain.contracts:
            return SignalResult(0.0, 0.0, {"error": "no option chain data"})

        # Group ATM options by expiration
        atm_options = [
            c for c in option_chain.contracts
            if c.delta and 0.35 <= abs(c.delta) <= 0.65 and c.iv and c.iv > 0 and c.dte
        ]

        if len(atm_options) < 2:
            return SignalResult(0.0, 0.0, {"error": "insufficient expirations"})

        # Group by expiration and compute avg IV per expiration
        exp_iv: dict[str, list[float]] = {}
        exp_dte: dict[str, int] = {}
        for c in atm_options:
            exp = c.expiration
            if exp not in exp_iv:
                exp_iv[exp] = []
                exp_dte[exp] = c.dte
            exp_iv[exp].append(c.iv)

        # Sort by DTE
        sorted_exps = sorted(exp_iv.keys(), key=lambda e: exp_dte[e])
        if len(sorted_exps) < 2:
            return SignalResult(0.0, 0.0, {"error": "need at least 2 expirations"})

        front_iv = np.mean(exp_iv[sorted_exps[0]])
        back_iv = np.mean(exp_iv[sorted_exps[1]])

        # Contango: back > front (normal) = calm
        # Backwardation: front > back = stress/event
        spread = back_iv - front_iv

        # Contango = calm (+1), backwardation = stress (-1)
        score = np.clip(spread / 10, -1, 1)

        confidence = min(1.0, abs(spread) / 8.0)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "front_iv": float(front_iv),
                "back_iv": float(back_iv),
                "spread": float(spread),
                "front_dte": exp_dte[sorted_exps[0]],
                "back_dte": exp_dte[sorted_exps[1]],
            },
        )
