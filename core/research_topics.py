"""Topic categorization for the agent's research cache TTL policy.

Research cache TTL categorization for agent research queries.
"""

from __future__ import annotations

_TICKER_SYMBOLS: set[str] = set()


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
    q = query.upper()
    tickers = _get_ticker_symbols()
    # Check if query mentions a known ticker
    for sym in tickers:
        if sym in q.split():
            return "ticker", 1800  # 30 min
    # Sector keywords
    sector_kw = {"SECTOR", "INDUSTRY", "ROTATION", "CYCLICAL", "DEFENSIVE", "GROWTH", "VALUE"}
    if any(kw in q for kw in sector_kw):
        return "sector", 14400  # 4 hours
    return "macro", 0  # 0 means "until session change"


__all__ = [
    "_TICKER_SYMBOLS",
    "_get_ticker_symbols",
    "_categorize_query",
]
