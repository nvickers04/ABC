"""Signal 37: Sector momentum — sector relative strength via peer comparison."""

import numpy as np
from signals.base import Signal, SignalResult


class SectorMomentumSignal(Signal):
    name = "sector_momentum"
    category = "macro"
    data_source = "yfinance"
    refresh_rate = "10min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        peer = data.get("peer_comparison")

        if peer is None:
            return SignalResult(0.0, 0.0, {"error": "no peer data"})

        components = {}
        scores = []

        # PeerComparison is a dataclass with: vs_sector, outperforming_sector,
        # symbol_return_20d, sector_return_20d, sector, sector_etf
        vs_sector = getattr(peer, "vs_sector", None)
        if vs_sector is not None:
            perf_score = np.clip(vs_sector / 5.0, -1, 1)
            scores.append(perf_score)
            components["perf_vs_sector"] = float(vs_sector)

        sym_return = getattr(peer, "symbol_return_20d", None)
        if sym_return is not None:
            components["symbol_return_20d"] = float(sym_return)

        sector_return = getattr(peer, "sector_return_20d", None)
        if sector_return is not None:
            components["sector_return_20d"] = float(sector_return)

        sector = getattr(peer, "sector", None)
        if sector:
            components["sector"] = sector

        if not scores:
            return SignalResult(0.0, 0.0, components)

        score = np.clip(sum(scores) / max(1, len(scores)), -1, 1)
        confidence = min(1.0, len(scores) / 3.0 * 0.8)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
