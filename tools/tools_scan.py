"""Market scan tool — bulk quote snapshot for top liquid names."""

import logging
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)

# Broad liquid watchlist: ETFs + mega-cap + sectors + momentum names (40+)
SCAN_SYMBOLS = [
    # Index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XBI", "ARKK", "XLP", "XLI",
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
    # Semiconductors
    "AMD", "SMCI", "AVGO", "MU", "MRVL",
    # Financials
    "JPM", "GS", "BAC",
    # Energy / commodities
    "XOM", "CVX", "SLV",
    # Volatility / hedging
    "UVXY", "GLD", "TLT",
    # Other liquid movers
    "NFLX", "CRM", "COIN", "PLTR", "SNAP", "BABA", "UBER", "SQ", "SHOP",
]


async def handle_market_scan(executor, params: dict) -> Any:
    """
    Scan top liquid names — returns sorted by absolute change%.

    Returns list of {symbol, last, change_pct, volume, bid, ask} sorted
    by biggest movers first, so the agent spots opportunities fast.
    """
    symbols = params.get("symbols", SCAN_SYMBOLS)
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]

    try:
        quotes = executor.data_provider.get_quotes_bulk(symbols)
        if not quotes:
            return {"error": "No quotes returned", "symbols_tried": len(symbols)}

        results = []
        for sym, q in quotes.items():
            try:
                d = asdict(q)
                results.append({
                    "symbol": d.get("symbol", sym),
                    "last": d.get("last"),
                    "change_pct": round((d.get("change_pct") or 0) * 100, 2),
                    "volume": d.get("volume", 0),
                    "bid": d.get("bid"),
                    "ask": d.get("ask"),
                })
            except Exception:
                logger.debug(f"Skipping bad quote for {sym}")
                continue

        # Sort by absolute change — biggest movers first
        results.sort(key=lambda r: abs(r.get("change_pct", 0)), reverse=True)

        return {
            "scan_count": len(results),
            "movers": results,
        }
    except Exception as e:
        logger.warning(f"market_scan failed: {e}")
        return {"error": str(e)}


HANDLERS = {
    "market_scan": handle_market_scan,
}
