"""
Signals Module — 50 independent signals from 5 data source categories.

Categories:
  Price (13)          — MDA candles (OHLCV multi-resolution)
  Volatility (10)     — MDA option chains + IV endpoints
  Fundamental (12)    — yfinance fundamentals/insider
  Macro/Sentiment (10)— environment + calendar + news
  Microstructure (5)  — MDA quotes + option flow

Each signal produces {score: float [-1,1], confidence: float [0,1], components: dict}
"""

from signals.base import Signal, SIGNAL_REGISTRY

__all__ = ["Signal", "SIGNAL_REGISTRY"]
