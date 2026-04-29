"""
IBKRQuoteSource — Real-time stock NBBO from IBKR with line-budget enforcement.

DESIGN
------
- Single owner of IBKR streaming subscriptions (the trading process).
- Hard cap on concurrent streams ("market data lines") so we never blow
  IBKR's per-user limit.  Default 90 of 100 lines, leaving 10 headroom
  for transient snapshot calls and bracket-order monitoring.
- Three fetch modes (priority order):
    1. Streaming: persistent reqMktData subscription, pushed updates,
       reads return latest tick instantly. Used for the active universe.
    2. Snapshot:  reqMktData(snapshot=True), one-shot, releases the
       line in ~11s.  Used for one-off lookups outside the universe.
    3. (NOT IMPLEMENTED YET) regulatorySnapshot for cold-start when
       no Network A/B/C subscription exists.
- Imbalance feed piggybacks on stock streams via genericTickList='588'
  (NYSE / ARCA / MKT order imbalances).  Costs zero additional lines.
- LRU eviction: when the streaming budget is full, promote() boots
  the least-recently-accessed stream to make room.
- FAIL HONEST: if IBKR is disconnected, contract qualification fails,
  or no tick has arrived yet, get_quote() returns None.  Callers must
  abstain.  We never silently fall back to delayed data.
- latest_quotes table: every successful tick is written so the future
  research daemon can read what the trader saw without holding its
  own IBKR connection.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------- Tunables (also exposed via core.config) ----------
DEFAULT_LINE_BUDGET = 90              # of IBKR's 100-line allocation
DEFAULT_SNAPSHOT_TIMEOUT_SECS = 11.0  # IBKR releases snapshot lines ~11s after request
# genericTickList '225' = Auction values (open/close cross).  Delivers tick IDs:
#   34 = auctionVolume      (paired shares)
#   35 = auctionPrice       (indicative cross price)
#   36 = auctionImbalance   (signed shares; +buy / -sell)
#   61 = regulatoryImbalance (NYSE only, signed shares)
# Free piggyback on the existing stock subscription.
DEFAULT_GENERIC_TICK_LIST = "225"


@dataclass
class IBKRQuote:
    """Quote payload extracted from an IBKR ticker."""
    symbol: str
    last: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    volume: int
    high: Optional[float]
    low: Optional[float]
    ts: float  # unix seconds when read
    # Auction-cross fields (None outside the open/close imbalance windows,
    # or when generic tick list 225 isn't subscribed).  Imbalance values
    # are SIGNED shares (positive = buy-side, negative = sell-side).
    auction_imbalance: Optional[float] = None
    auction_volume: Optional[int] = None
    auction_price: Optional[float] = None
    regulatory_imbalance: Optional[float] = None


def _ticker_value(value: Any) -> Optional[float]:
    """Convert ib_insync ticker fields (float or NaN sentinel) to float|None.

    ib_insync uses NaN to mean "no value yet" instead of None.  We normalize.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # NaN check without numpy dep
    if v != v:
        return None
    if v <= 0:
        # IBKR uses -1 / 0 as "unset" for some fields too
        return None
    return v


def _signed_ticker_value(value: Any) -> Optional[float]:
    """Like ``_ticker_value`` but preserves negatives.

    Auction imbalance and regulatory imbalance are SIGNED (negative means
    sell-side imbalance).  Only None and NaN are treated as "no data".
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _read_ticker(symbol: str, ticker: Any) -> Optional[IBKRQuote]:
    """Convert an ib_insync Ticker into an IBKRQuote, or None if no usable data."""
    if ticker is None:
        return None
    last = _ticker_value(getattr(ticker, "last", None))
    bid = _ticker_value(getattr(ticker, "bid", None))
    ask = _ticker_value(getattr(ticker, "ask", None))
    # Need at least one of (last) or (bid AND ask) to be useful
    if last is None and (bid is None or ask is None):
        return None
    vol_raw = getattr(ticker, "volume", 0)
    try:
        volume = int(vol_raw) if vol_raw and vol_raw == vol_raw and vol_raw >= 0 else 0
    except (TypeError, ValueError):
        volume = 0
    # Auction-cross fields (only present during open/close imbalance windows
    # when genericTickList='225' is subscribed).  Imbalance values are signed.
    auction_imbalance = _signed_ticker_value(getattr(ticker, "auctionImbalance", None))
    auction_price = _ticker_value(getattr(ticker, "auctionPrice", None))
    regulatory_imbalance = _signed_ticker_value(getattr(ticker, "regulatoryImbalance", None))
    auction_volume_raw = getattr(ticker, "auctionVolume", None)
    auction_volume: Optional[int]
    try:
        if auction_volume_raw is None or auction_volume_raw != auction_volume_raw:  # None / NaN
            auction_volume = None
        else:
            av = int(auction_volume_raw)
            auction_volume = av if av > 0 else None
    except (TypeError, ValueError):
        auction_volume = None

    return IBKRQuote(
        symbol=symbol,
        last=last,
        bid=bid,
        ask=ask,
        volume=volume,
        high=_ticker_value(getattr(ticker, "high", None)),
        low=_ticker_value(getattr(ticker, "low", None)),
        ts=time.time(),
        auction_imbalance=auction_imbalance,
        auction_volume=auction_volume,
        auction_price=auction_price,
        regulatory_imbalance=regulatory_imbalance,
    )


class IBKRQuoteSource:
    """Manages streaming + snapshot quote subscriptions to IBKR with a line cap."""

    def __init__(
        self,
        connector: Any,
        *,
        line_budget: int = DEFAULT_LINE_BUDGET,
        generic_tick_list: str = DEFAULT_GENERIC_TICK_LIST,
        snapshot_timeout_secs: float = DEFAULT_SNAPSHOT_TIMEOUT_SECS,
        latest_quote_writer: Optional[Any] = None,
    ):
        """
        Args:
            connector:  IBKRConnector singleton (must expose `.ib`, `.is_connected()`,
                        and have a running event loop).
            line_budget: max concurrent streaming subscriptions.  Default 90.
            generic_tick_list: IBKR genericTickList passed to reqMktData.
                        '588' enables NYSE/ARCA/MKT order imbalance ticks at no
                        extra line cost (piggybacks on the existing stock sub).
            snapshot_timeout_secs: how long to await a snapshot tick.
            latest_quote_writer: optional callable(symbol, IBKRQuote) invoked on
                        every successful read so an external store (latest_quotes
                        table) can mirror what the trader sees.
        """
        self._connector = connector
        self._line_budget = max(1, int(line_budget))
        self._generic_tick_list = generic_tick_list
        self._snapshot_timeout_secs = float(snapshot_timeout_secs)
        self._writer = latest_quote_writer

        # Streaming state. OrderedDict gives us LRU ordering for free:
        # move_to_end on access, popitem(last=False) to evict the oldest.
        self._streams: "OrderedDict[str, Any]" = OrderedDict()  # symbol -> ticker
        self._lock = Lock()

    # ---------- introspection ----------

    @property
    def line_budget(self) -> int:
        return self._line_budget

    @property
    def lines_in_use(self) -> int:
        with self._lock:
            return len(self._streams)

    def is_streaming(self, symbol: str) -> bool:
        with self._lock:
            return symbol.upper() in self._streams

    def streamed_symbols(self) -> list[str]:
        with self._lock:
            return list(self._streams.keys())

    # ---------- streaming subscription management ----------

    async def promote(self, symbol: str) -> bool:
        """Ensure `symbol` has a streaming subscription.  Evicts LRU on overflow.

        Returns True if a stream is in place after the call, False on failure.
        """
        sym = symbol.upper()
        if not self._connector_ready():
            return False

        # Already streaming? Bump LRU and return.
        with self._lock:
            if sym in self._streams:
                self._streams.move_to_end(sym)
                return True

        # Need to subscribe.  Make room if at budget.
        await self._evict_until_room()

        ticker = await self._subscribe(sym)
        if ticker is None:
            return False

        with self._lock:
            # Re-check under lock in case another coroutine subscribed concurrently
            if sym not in self._streams:
                self._streams[sym] = ticker
            else:
                # Two coroutines raced; cancel the loser's ticker
                try:
                    self._connector.ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
                self._streams.move_to_end(sym)
        return True

    async def demote(self, symbol: str) -> bool:
        """Cancel the streaming subscription for `symbol`.  Returns True if removed."""
        sym = symbol.upper()
        with self._lock:
            ticker = self._streams.pop(sym, None)
        if ticker is None:
            return False
        try:
            self._connector.ib.cancelMktData(ticker.contract)
            return True
        except Exception as e:
            logger.warning("demote(%s): cancelMktData failed: %s", sym, e)
            return False

    async def _evict_until_room(self) -> None:
        """Pop LRU streams until len(_streams) < budget."""
        while True:
            with self._lock:
                if len(self._streams) < self._line_budget:
                    return
                victim_sym, victim_ticker = self._streams.popitem(last=False)
            try:
                self._connector.ib.cancelMktData(victim_ticker.contract)
                logger.info(
                    "IBKRQuoteSource evicted LRU stream %s (lines now %d/%d)",
                    victim_sym, len(self._streams), self._line_budget,
                )
            except Exception as e:
                logger.warning("Eviction cancelMktData(%s) failed: %s", victim_sym, e)

    async def _subscribe(self, symbol: str) -> Optional[Any]:
        """Open a new streaming subscription for `symbol`. Returns ticker or None."""
        from ib_insync.contract import Stock
        ib = self._connector.ib
        try:
            # Force live data type — paper accounts have free real-time stocks
            # if the entitlement is in place.  No silent delayed fallback.
            ib.reqMarketDataType(1)
            contract = Stock(symbol, "SMART", "USD")
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.warning("IBKRQuoteSource: contract %s did not qualify", symbol)
                return None
            ticker = ib.reqMktData(
                contract,
                self._generic_tick_list,
                False,  # snapshot
                False,  # regulatorySnapshot
            )
            logger.info(
                "IBKRQuoteSource subscribed %s (lines now %d/%d, ticks=%s)",
                symbol, self.lines_in_use + 1, self._line_budget,
                self._generic_tick_list or "default",
            )
            return ticker
        except Exception as e:
            logger.warning("IBKRQuoteSource subscribe(%s) failed: %s", symbol, e)
            return None

    # ---------- read paths ----------

    async def get_quote(self, symbol: str, *, allow_promote: bool = True) -> Optional[IBKRQuote]:
        """Get current quote for `symbol`.

        If already streaming, read the latest tick.  Otherwise, if
        allow_promote=True, promote to streaming (evicting LRU if needed).
        If allow_promote=False, fall back to a one-shot snapshot (does
        not consume a long-lived line).

        Returns None on failure — caller MUST abstain rather than trade.
        """
        sym = symbol.upper()
        if not self._connector_ready():
            return None

        # Fast path: already streaming
        ticker = self._get_ticker(sym)
        if ticker is not None:
            quote = _read_ticker(sym, ticker)
            if quote is not None:
                self._touch_lru(sym)
                self._mirror(quote)
                return quote
            # Stream exists but no usable data yet.  Wait briefly for a tick.
            quote = await self._await_tick(ticker, sym)
            if quote is not None:
                self._touch_lru(sym)
                self._mirror(quote)
            return quote

        if allow_promote:
            ok = await self.promote(sym)
            if not ok:
                return None
            ticker = self._get_ticker(sym)
            quote = _read_ticker(sym, ticker) if ticker is not None else None
            if quote is None and ticker is not None:
                quote = await self._await_tick(ticker, sym)
            if quote is not None:
                self._mirror(quote)
            return quote

        # Snapshot fallback — does NOT hold a line beyond ~11s
        return await self._get_snapshot(sym)

    async def _await_tick(self, ticker: Any, symbol: str) -> Optional[IBKRQuote]:
        """Wait up to snapshot_timeout for the streaming ticker to populate."""
        import asyncio
        deadline = time.time() + self._snapshot_timeout_secs
        while time.time() < deadline:
            await asyncio.sleep(0.1)
            quote = _read_ticker(symbol, ticker)
            if quote is not None:
                return quote
        return None

    async def _get_snapshot(self, symbol: str) -> Optional[IBKRQuote]:
        """One-shot snapshot request (does not hold a long-lived line)."""
        from ib_insync.contract import Stock
        import asyncio
        ib = self._connector.ib
        try:
            ib.reqMarketDataType(1)
            contract = Stock(symbol, "SMART", "USD")
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                return None
            ticker = ib.reqMktData(contract, "", True, False)  # snapshot=True
            try:
                deadline = time.time() + self._snapshot_timeout_secs
                while time.time() < deadline:
                    await asyncio.sleep(0.1)
                    quote = _read_ticker(symbol, ticker)
                    if quote is not None:
                        self._mirror(quote)
                        return quote
                return None
            finally:
                # Snapshot lines auto-release, but be explicit.
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("IBKRQuoteSource snapshot(%s) failed: %s", symbol, e)
            return None

    # ---------- helpers ----------

    def _connector_ready(self) -> bool:
        try:
            return bool(self._connector and self._connector.is_connected())
        except Exception:
            return False

    def _get_ticker(self, symbol: str) -> Optional[Any]:
        with self._lock:
            return self._streams.get(symbol)

    def _touch_lru(self, symbol: str) -> None:
        with self._lock:
            if symbol in self._streams:
                self._streams.move_to_end(symbol)

    def _mirror(self, quote: IBKRQuote) -> None:
        if self._writer is None:
            return
        try:
            self._writer(quote)
        except Exception as e:
            logger.debug("latest_quotes writer failed for %s: %s", quote.symbol, e)


# ---------- Module-level singleton plumbing ----------

_singleton: Optional[IBKRQuoteSource] = None
_singleton_lock = Lock()


def get_ibkr_quote_source() -> Optional[IBKRQuoteSource]:
    """Return the process-wide IBKRQuoteSource, constructing on first call.

    Returns None if IBKR_QUOTES_ENABLED is False or the connector cannot
    be obtained.  Construction does NOT trigger a connection — that happens
    lazily on the first promote()/get_quote() call.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        try:
            from core import config as _cfg
            if not getattr(_cfg, "IBKR_QUOTES_ENABLED", False):
                return None
            line_budget = int(getattr(_cfg, "IBKR_QUOTE_LINE_BUDGET", DEFAULT_LINE_BUDGET))
        except Exception:
            return None
        try:
            from execution.ibkr_core import get_ibkr_connector
            connector = get_ibkr_connector()
        except Exception as e:
            logger.warning("get_ibkr_quote_source: connector unavailable: %s", e)
            return None
        # Wire the latest_quotes writer
        try:
            from memory import write_latest_quote
            writer = write_latest_quote
        except Exception:
            writer = None
        _singleton = IBKRQuoteSource(
            connector,
            line_budget=line_budget,
            latest_quote_writer=writer,
        )
        logger.info(
            "IBKRQuoteSource initialized (line_budget=%d, generic_ticks=%s)",
            line_budget, DEFAULT_GENERIC_TICK_LIST,
        )
        return _singleton


def reset_ibkr_quote_source_for_tests() -> None:
    """Test helper: clear the singleton so each test gets a fresh instance."""
    global _singleton
    with _singleton_lock:
        _singleton = None
