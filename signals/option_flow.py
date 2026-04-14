"""Signal 47: Option OI flow — call vs put OI direction."""

import numpy as np
from signals.base import Signal, SignalResult


class OptionFlowSignal(Signal):
    name = "option_flow"
    category = "microstructure"
    data_source = "mda_options"
    refresh_rate = "5min"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        chain = data.get("option_chain")

        if not chain:
            return SignalResult(0.0, 0.0, {"error": "no option chain"})

        components = {}
        total_call_oi = 0
        total_put_oi = 0
        total_call_vol = 0
        total_put_vol = 0

        contracts = getattr(chain, "contracts", []) or []
        for contract in contracts:
            oi = getattr(contract, "open_interest", 0) or 0
            vol = getattr(contract, "volume", 0) or 0
            opt_type = getattr(contract, "side", "").lower()

            if opt_type == "call":
                total_call_oi += oi
                total_call_vol += vol
            elif opt_type == "put":
                total_put_oi += oi
                total_put_vol += vol

        total_oi = total_call_oi + total_put_oi
        total_vol = total_call_vol + total_put_vol

        components["call_oi"] = total_call_oi
        components["put_oi"] = total_put_oi
        components["call_vol"] = total_call_vol
        components["put_vol"] = total_put_vol

        if total_oi == 0 and total_vol == 0:
            return SignalResult(0.0, 0.0, components)

        scores = []

        # OI skew: more call OI = bullish, more put OI = bearish
        if total_oi > 0:
            oi_ratio = total_call_oi / total_oi
            oi_score = np.clip((oi_ratio - 0.5) * 4, -1, 1)
            scores.append(oi_score * 0.5)
            components["call_oi_pct"] = float(round(oi_ratio * 100, 1))

        # Volume skew: call volume building = bullish flow
        if total_vol > 0:
            vol_ratio = total_call_vol / total_vol
            vol_score = np.clip((vol_ratio - 0.5) * 4, -1, 1)
            scores.append(vol_score * 0.5)
            components["call_vol_pct"] = float(round(vol_ratio * 100, 1))

        if not scores:
            return SignalResult(0.0, 0.0, components)

        score = np.clip(sum(scores), -1, 1)
        confidence = min(1.0, (total_oi + total_vol) / 10000 * 0.5 + 0.3)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
