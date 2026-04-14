"""Signal 50: Institutional flow — combined institutional positioning signal."""

import numpy as np
from signals.base import Signal, SignalResult


class InstitutionalFlowSignal(Signal):
    name = "institutional_flow"
    category = "microstructure"
    data_source = "mda_options"
    refresh_rate = "10min"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        components = {}
        scores = []

        # Institutional ownership from fundamentals
        fundamentals = data.get("fundamentals")
        if fundamentals:
            inst_pct = getattr(fundamentals, "institutional_pct", None)
            if inst_pct is not None:
                inst_pct = float(inst_pct)
                # High institutional ownership = stability/accumulation
                if inst_pct > 80:
                    inst_score = 0.4
                elif inst_pct > 60:
                    inst_score = 0.2
                elif inst_pct > 40:
                    inst_score = 0.0
                else:
                    inst_score = -0.2  # Low institutional interest
                scores.append(inst_score)
                components["institutional_pct"] = inst_pct

        # Option OI skew as institutional flow proxy
        chain = data.get("option_chain")
        if chain:
            total_call_oi = 0
            total_put_oi = 0
            large_call_oi = 0
            large_put_oi = 0

            contracts = getattr(chain, "contracts", []) or []
            for contract in contracts:
                oi = getattr(contract, "open_interest", 0) or 0
                opt_type = getattr(contract, "side", "").lower()

                if opt_type == "call":
                    total_call_oi += oi
                    if oi > 1000:  # Large position proxy
                        large_call_oi += oi
                elif opt_type == "put":
                    total_put_oi += oi
                    if oi > 1000:
                        large_put_oi += oi

            total_oi = total_call_oi + total_put_oi
            if total_oi > 0:
                call_pct = total_call_oi / total_oi
                oi_score = np.clip((call_pct - 0.5) * 3, -1, 1)
                scores.append(oi_score * 0.3)
                components["call_oi_pct"] = float(round(call_pct * 100, 1))

            large_total = large_call_oi + large_put_oi
            if large_total > 0:
                large_call_pct = large_call_oi / large_total
                large_score = np.clip((large_call_pct - 0.5) * 3, -1, 1)
                scores.append(large_score * 0.3)
                components["large_call_oi_pct"] = float(round(large_call_pct * 100, 1))

        # Volume surge from quote data as block-trade proxy
        quote = data.get("quote")
        candles = data.get("candles")
        if quote and candles:
            current_vol = getattr(quote, "volume", None)
            if current_vol and current_vol > 0:
                try:
                    hist_vols = [
                        c.volume for c in candles[-10:] if c.volume and c.volume > 0
                    ]
                    if len(hist_vols) >= 3:
                        avg_vol = np.mean(hist_vols)
                        vol_ratio = current_vol / max(avg_vol, 1)
                        # Abnormally high volume = institutional activity
                        if vol_ratio > 3.0:
                            scores.append(0.3)
                            components["volume_surge"] = float(round(vol_ratio, 1))
                        elif vol_ratio > 2.0:
                            scores.append(0.15)
                            components["volume_surge"] = float(round(vol_ratio, 1))
                except (AttributeError, TypeError):
                    pass

        if not scores:
            return SignalResult(0.0, 0.0, components)

        score = np.clip(sum(scores) / max(len(scores), 1), -1, 1)
        confidence = min(1.0, len(scores) / 4.0 * 0.8)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
