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


class Signal(ABC):
    """Base class for all scoring signals."""

    name: str = ""
    category: str = ""         # "price", "volatility", "fundamental", "macro", "microstructure"
    data_source: str = ""      # "mda_candles", "mda_options", "yfinance", "environment", "mda_quotes"
    refresh_rate: str = "every_round"  # "every_round", "5min", "10min", "daily"

    # Tier: 1 = cheap (no option chains), 2 = expensive (needs option chains)
    tier: int = 1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
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
