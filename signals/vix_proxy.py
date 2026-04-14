"""Signal 41: VIX proxy — fear gauge via UVXY / SPY IV / SPY ATR fallback chain."""

import numpy as np
from signals.base import Signal, SignalResult


class VixProxySignal(Signal):
    name = "vix_proxy"
    category = "macro"
    data_source = "mda_quotes"
    refresh_rate = "5min"
    tier = 1

    # Historical VIX ranges for scoring
    _VIX_LOW = 12.0
    _VIX_MID = 20.0
    _VIX_HIGH = 30.0

    def compute(self, symbol: str, data: dict) -> SignalResult:
        dp = data.get("data_provider")
        if dp is None:
            return SignalResult(0.0, 0.0, {"error": "no data_provider"})

        vix_value = None
        source = None
        confidence_penalty = 0.0
        components = {}

        # Fallback 1: UVXY quote as VIX proxy
        try:
            uvxy = dp.get_quote("UVXY")
            if uvxy and uvxy.last and uvxy.last > 0:
                vix_value = float(uvxy.last)
                source = "UVXY"
                # UVXY is leveraged, normalize roughly
                # UVXY ~12-20 in calm, ~30+ in fear — use as-is as rough VIX proxy
                components["uvxy_price"] = vix_value
        except Exception:
            pass

        # Fallback 2: SPY IV rank
        if vix_value is None:
            try:
                iv_info = dp.get_iv_info("SPY", dte_min=20, dte_max=40)
                if iv_info and getattr(iv_info, "iv_current", None):
                    vix_value = float(iv_info.iv_current)
                    source = "SPY_IV"
                    components["spy_iv"] = vix_value
            except Exception:
                pass

        # Fallback 3: SPY ATR proxy
        if vix_value is None:
            try:
                atr = dp.get_atr("SPY", 14)
                quote = dp.get_quote("SPY")
                if atr and quote and quote.last and quote.last > 0:
                    atr_pct = atr.value / quote.last
                    vix_value = float(round(atr_pct * (252 ** 0.5) * 100, 1))
                    source = "SPY_ATR"
                    confidence_penalty = 0.3  # ATR proxy is least reliable
                    components["spy_atr"] = float(atr.value)
                    components["spy_price"] = float(quote.last)
            except Exception:
                pass

        if vix_value is None:
            return SignalResult(0.0, 0.0, {"error": "all VIX sources failed"})

        components["vix_estimate"] = vix_value
        components["source"] = source

        # Score: low VIX = calm = bullish (+1), high VIX = fear = bearish (-1)
        if vix_value <= self._VIX_LOW:
            score = 1.0
        elif vix_value >= self._VIX_HIGH:
            score = -1.0
        else:
            # Linear interpolation between low and high
            score = 1.0 - 2.0 * (vix_value - self._VIX_LOW) / (self._VIX_HIGH - self._VIX_LOW)

        score = np.clip(score, -1, 1)
        confidence = max(0.0, min(1.0, 0.9 - confidence_penalty))

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
