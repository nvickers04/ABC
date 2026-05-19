"""
Signal base class — interface for all 50 signals.

Every signal subclass must define:
  - name, category, data_source, refresh_rate
  - score(symbol, data) -> {score, confidence, components}
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Global signal registry — populated by Signal.__init_subclass__
SIGNAL_REGISTRY: dict[str, "Signal"] = {}


@dataclass
class SignalResult:
    """Result of a signal scoring call."""
    score: float       # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    components: dict   # Breakdown of sub-scores for transparency

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "components": self.components,
        }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── Per-category forward-return defaults ─────────────────────────────
# (resolution, horizon_in_bars, lookback_days).  These define the
# "natural cadence" at which a category of signal is expected to
# manifest its edge in realised forward returns:
#
#   microstructure: 5-min bars, 6 bars ahead = 30 min, 5 days history
#                   → flow/quote signals decay within minutes, IC accumulates fast
#   price:          1-hour bars, 4 bars ahead = 4 hours, 30 days history
#                   → intraday momentum/breakout resolve within a session
#   volatility:     1-hour bars, 24 bars ahead = ~1 day, 30 days history
#                   → IV mean-reverts over a session+
#   macro:          daily bars, 1 day ahead, 60 days history
#                   → regime/breadth shifts unfold day-over-day
#   fundamental:    daily bars, 5 days ahead, 90 days history
#                   → earnings drift / re-rating takes a week+
#
# Individual signals may override any of these on the subclass.
CATEGORY_FORWARD_DEFAULTS: dict[str, dict[str, Any]] = {
    "microstructure": {"return_resolution": "5min", "return_horizon": 6,  "return_lookback_days": 5},
    "price":          {"return_resolution": "1h",   "return_horizon": 4,  "return_lookback_days": 30},
    "volatility":     {"return_resolution": "1h",   "return_horizon": 24, "return_lookback_days": 30},
    "macro":          {"return_resolution": "D",    "return_horizon": 1,  "return_lookback_days": 60},
    "fundamental":    {"return_resolution": "D",    "return_horizon": 5,  "return_lookback_days": 90},
}


class Signal(ABC):
    """Base class for all scoring signals."""

    name: str = ""
    category: str = ""         # "price", "volatility", "fundamental", "macro", "microstructure"
    data_source: str = ""      # "mda_candles", "mda_options", "yfinance", "environment", "mda_quotes"
    refresh_rate: str = "every_round"  # "every_round", "5min", "10min", "daily"

    # Tier: 1 = cheap (no option chains), 2 = expensive (needs option chains)
    tier: int = 1

    # If True, this signal needs a live IBKR connection to produce
    # meaningful output (e.g. NYSE auction-imbalance generic ticks).
    # The research host (which has IBKR quotes disabled) skips these
    # signals entirely; they only run inside the trader process.
    requires_ibkr: bool = False

    # ── Forward-return measurement attributes ─────────────────────────
    # Each signal declares the candle resolution and horizon (in bars of
    # that resolution) used to compute its realised forward return for
    # IC attribution.  Signals with naturally short horizons (microstructure,
    # short-term price) sample fast and produce many independent
    # observations per day; longer-horizon signals (fundamentals, macro)
    # produce fewer observations.  This per-signal cadence is the basis
    # of the IC pipeline — it MUST match the timeframe at which the
    # signal's edge would manifest.  Defaults below are the per-category
    # baselines from CATEGORY_FORWARD_DEFAULTS; subclasses may override.
    return_resolution: str = "D"
    return_horizon: int = 5
    return_lookback_days: int = 60

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Apply per-category forward-return defaults if subclass didn't
        # explicitly override them.  Looks at the subclass's __dict__ —
        # if return_resolution / return_horizon / return_lookback_days
        # aren't set on the subclass itself, inherit from category.
        cat_defaults = CATEGORY_FORWARD_DEFAULTS.get(cls.category)
        if cat_defaults:
            for attr, default_val in cat_defaults.items():
                if attr not in cls.__dict__:
                    setattr(cls, attr, default_val)
        if cls.name:
            SIGNAL_REGISTRY[cls.name] = cls()

    @abstractmethod
    def compute(self, symbol: str, data: dict) -> SignalResult:
        """
        Compute the signal score for a symbol.

        Args:
            symbol: Stock ticker
            data: Dict containing pre-fetched data relevant to this signal.
                  Keys depend on signal category:
                  - "candles": Candles dataclass (price signals)
                  - "candles_daily": Candles dataclass with daily resolution
                  - "quote": Quote dataclass (microstructure signals)
                  - "option_chain": OptionChain dataclass (volatility signals)
                  - "iv_info": IVInfo dataclass (volatility signals)
                  - "fundamentals": ExtendedFundamentals (fundamental signals)
                  - "analyst": AnalystData (fundamental signals)
                  - "insider": InsiderData (fundamental signals)
                  - "institutional": InstitutionalData (fundamental signals)
                  - "earnings_info": EarningsInfo (fundamental signals)
                  - "earnings_history": list (fundamental signals)
                  - "news": NewsData (macro signals)
                  - "environment": dict (macro signals)
                  - "peer": PeerComparison (macro signals)
                  - "spy_candles": Candles (macro signals)
                  - "qqq_candles": Candles (macro signals)

        Returns:
            SignalResult with score, confidence, and component breakdown.
        """
        ...

    def score(self, symbol: str, data: dict) -> dict:
        """Safe wrapper that catches errors and clamps output."""
        try:
            result = self.compute(symbol, data)
            result.score = _clamp(result.score, -1.0, 1.0)
            result.confidence = _clamp(result.confidence, 0.0, 1.0)
            return result.to_dict()
        except Exception as e:
            logger.warning(f"Signal {self.name} failed for {symbol}: {e}")
            return {"score": 0.0, "confidence": 0.0, "components": {"error": str(e)}}
