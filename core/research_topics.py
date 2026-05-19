"""Topic categorization for the agent's research cache TTL policy.

TTLs come from master ProfitConfig (:func:`core.central_profit_config.get_research_settings`)
via :class:`core.memory_config.MemoryConfig` — same values as the trader's research() cache.
"""

from __future__ import annotations

_TICKER_SYMBOLS: set[str] = set()


def invalidate_research_topic_caches() -> None:
    """Clear lazy universe cache after ProfitConfig profile reload."""
    global _TICKER_SYMBOLS
    _TICKER_SYMBOLS = set()


def _get_ticker_symbols() -> set[str]:
    """Lazily load RESEARCH_UNIVERSE symbols for topic categorization."""
    global _TICKER_SYMBOLS
    if not _TICKER_SYMBOLS:
        try:
            from research.config import RESEARCH_UNIVERSE
            _TICKER_SYMBOLS = {s.upper() for s in RESEARCH_UNIVERSE}
        except Exception:
            pass
    return _TICKER_SYMBOLS


def _categorize_query(query: str) -> tuple[str, int]:
    """Categorize a research query and return (category, ttl_seconds).

    Categories:
      macro  - market-wide events, Fed, economic -> TTL=session boundary
      sector - industry, sector rotation         -> TTL=4h
      ticker - specific symbol mentions          -> TTL=30min
    """
    from core.central_profit_config import get_research_settings

    mem = get_research_settings()
    q = query.upper()
    tickers = _get_ticker_symbols()
    for sym in tickers:
        if sym in q.split():
            return "ticker", mem.research_ttl_ticker_seconds
    sector_kw = {"SECTOR", "INDUSTRY", "ROTATION", "CYCLICAL", "DEFENSIVE", "GROWTH", "VALUE"}
    if any(kw in q for kw in sector_kw):
        return "sector", mem.research_ttl_sector_seconds
    return "macro", mem.research_ttl_macro_seconds


__all__ = [
    "_TICKER_SYMBOLS",
    "_get_ticker_symbols",
    "_categorize_query",
    "invalidate_research_topic_caches",
]
