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


def _autoload_signal_modules() -> None:
    """Import every signal module in this package so their Signal subclasses
    register themselves in SIGNAL_REGISTRY on package import.

    Infrastructure modules (base/combiner/scorer/templates/…) are skipped
    because they contain no Signal subclasses and several of them import
    heavy runtime dependencies we don't want to pull in transitively.
    """
    import importlib
    import logging
    import pkgutil

    logger = logging.getLogger(__name__)

    _SKIP = {
        "base",
        "combiner",
        "scorer",
        "templates",
        "template_evolution",
        "briefing",
        # ── Regime / market-wide signals ──────────────────────────
        # These emit the SAME value for every symbol in the universe
        # each round, so they contribute zero cross-sectional info to
        # per-symbol composites (a constant added to every composite
        # cancels out under any monotone ranking).  They belong in a
        # regime-conditioning layer, not in the per-symbol scorer.
        # Disabled here pending that refactor; can be re-enabled by
        # removing from this set.
        "market_momentum",
        "vix_proxy",
        "breadth",
        "correlation_regime",
        "carry",
    }

    for _finder, modname, _ispkg in pkgutil.iter_modules(__path__):
        if modname.startswith("_") or modname in _SKIP:
            continue
        try:
            importlib.import_module(f"{__name__}.{modname}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to import signal module %s: %s", modname, e)


_autoload_signal_modules()

__all__ = ["Signal", "SIGNAL_REGISTRY"]
