"""Signal 21: Option volume surge — today's option volume vs 20-day average."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength
from signals.base import Signal, SignalResult


class OptionVolumeRatioSignal(Signal):
    name = "option_volume_surge"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        option_chain = data.get("option_chain")

        if option_chain is None or not option_chain.contracts:
            return SignalResult(0.0, 0.0, {"error": "no option chain"})

        # Total call and put volume
        call_volume = sum(c.volume for c in option_chain.calls() if c.volume)
        put_volume = sum(c.volume for c in option_chain.puts() if c.volume)
        total_volume = call_volume + put_volume

        if total_volume == 0:
            return SignalResult(0.0, 0.0, {"error": "zero option volume"})

        # Direction bias from volume split
        if call_volume + put_volume > 0:
            call_ratio = call_volume / (call_volume + put_volume)
            direction = (call_ratio - 0.5) * 2  # -1 (all puts) to +1 (all calls)
        else:
            direction = 0.0

        # Volume magnitude (use OI as baseline since we don't have historical vol)
        total_oi = sum(c.open_interest for c in option_chain.contracts if c.open_interest)
        if total_oi > 0:
            vol_to_oi = total_volume / total_oi
            magnitude = np.clip(vol_to_oi / 0.3 - 1.0, 0, 1)  # 30% turnover = baseline
        else:
            magnitude = 0.5

        score = bounded_tanh(direction * (0.5 + magnitude * 0.5), scale=1.3)
        depth = min((call_volume + put_volume) / 20_000.0, 1.0)
        confidence = confidence_from_strength(
            abs(score),
            data_quality=min(1.0, 0.5 * magnitude + 0.5 * depth),
        )

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "call_volume": int(call_volume),
                "put_volume": int(put_volume),
                "call_ratio": float(call_ratio) if (call_volume + put_volume) > 0 else 0,
                "vol_to_oi": float(vol_to_oi) if total_oi > 0 else 0,
            },
        )
