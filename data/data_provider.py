"""
DataProvider - Unified interface for all market data access.

LAYER 2: Data (Stable Interface)
Wraps MarketDataClient (official SDK) with source tracking.

Usage:
    from data.data_provider import get_data_provider

    provider = get_data_provider()
    quote = provider.get_quote('AAPL')  # Returns Quote with source='marketdata' or 'yfinance'
    atr = provider.get_atr('AAPL')      # Returns ATRResult with value and source
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES (with source tracking)
# ============================================================

@dataclass
class Quote:
    """Real-time quote with source tracking."""
    symbol: str
    last: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    volume: int
    change_pct: Optional[float]
    source: str  # 'marketdata', 'yfinance', 'ibkr'
    timestamp: Optional[datetime] = None
    source_updated: Optional[int] = None  # Unix timestamp from data source
    # Auction-cross fields (populated only by the IBKR backend during the
    # open/close imbalance windows; None otherwise).  Imbalance values are
    # SIGNED shares (positive = buy-side, negative = sell-side).
    auction_imbalance: Optional[float] = None
    auction_volume: Optional[int] = None
    auction_price: Optional[float] = None
    regulatory_imbalance: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def data_age_seconds(self) -> Optional[float]:
        """How old the source data is, in seconds."""
        if self.source_updated:
            return datetime.now().timestamp() - self.source_updated
        return None

    @property
    def is_stale(self) -> bool:
        """True if data is more than 30 seconds old."""
        age = self.data_age_seconds
        return age is not None and age > 30


@dataclass
class Candles:
    """Historical OHLCV data with source tracking.

    Supports iteration and slicing to yield SimpleNamespace bar objects
    with .open, .high, .low, .close, .volume, .timestamp attributes.
    """
    symbol: str
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[int]
    timestamps: List[int]  # Unix timestamps
    source: str

    def __len__(self) -> int:
        return len(self.close)

    def _bar(self, i: int):
        from types import SimpleNamespace
        return SimpleNamespace(
            open=self.open[i] if i < len(self.open) else None,
            high=self.high[i] if i < len(self.high) else None,
            low=self.low[i] if i < len(self.low) else None,
            close=self.close[i] if i < len(self.close) else None,
            volume=self.volume[i] if i < len(self.volume) else None,
            timestamp=self.timestamps[i] if i < len(self.timestamps) else None,
        )

    def __getitem__(self, key):
        n = len(self.close)
        if isinstance(key, slice):
            return [self._bar(i) for i in range(*key.indices(n))]
        if isinstance(key, int):
            if key < 0:
                key += n
            if 0 <= key < n:
                return self._bar(key)
            raise IndexError(key)
        raise TypeError(key)

    def __iter__(self):
        for i in range(len(self.close)):
            yield self._bar(i)

    @property
    def latest_close(self) -> Optional[float]:
        """Get most recent close price."""
        return self.close[-1] if self.close else None


@dataclass
class ATRResult:
    """ATR calculation result with source tracking.
    
    TODO: Wire as tool - useful for position sizing and stop placement.
    """
    symbol: str
    value: float
    period: int
    source: str

    @property
    def as_percent(self) -> float:
        """ATR as percentage (requires price context)."""
        return self.value  # Caller should divide by price if needed


@dataclass
class Fundamentals:
    """TODO: Wire as tool - agent needs fundamental data for research.Basic fundamental data."""
    symbol: str
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    earnings_date: Optional[datetime]
    source: str


@dataclass
class EarningsInfo:
    """Earnings calendar information."""
    symbol: str
    next_earnings_date: Optional[datetime]
    days_until_earnings: Optional[int]
    source: str



@dataclass
class OptionContract:
    """Option contract with Greeks and pricing."""
    option_symbol: str
    underlying: str
    strike: float
    side: str  # 'call' or 'put'
    expiration: str
    dte: Optional[int]
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    last: Optional[float]
    volume: int
    open_interest: int
    # Greeks
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    iv: Optional[float]
    source: str
    # Historical metadata (set when fetched for a specific historical date)
    as_of_date: Optional[str] = None
    underlying_price: Optional[float] = None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Spread as percentage of mid."""
        if self.mid and self.mid > 0 and self.spread:
            return (self.spread / self.mid) * 100
        return None


@dataclass
class OptionChain:
    """Options chain for an underlying."""
    symbol: str
    contracts: List[OptionContract]
    source: str
    is_historical: bool = False
    as_of_date: Optional[str] = None

    def calls(self) -> List[OptionContract]:
        """Get call contracts only."""
        return [c for c in self.contracts if c.side == 'call']

    def puts(self) -> List[OptionContract]:
        """Get put contracts only."""
        return [c for c in self.contracts if c.side == 'put']

    def filter_by_dte(self, min_dte: int, max_dte: int) -> List[OptionContract]:
        """Filter contracts by DTE range."""
        return [c for c in self.contracts if c.dte and min_dte <= c.dte <= max_dte]

    def filter_by_delta(self, min_delta: float, max_delta: float) -> List[OptionContract]:
        """Filter contracts by delta range (absolute value)."""
        return [
            c for c in self.contracts
            if c.delta and min_delta <= abs(c.delta) <= max_delta
        ]


@dataclass
class IVInfo:
    """Implied volatility information."""
    symbol: str
    iv_current: float  # Current IV as percentage
    iv_rank: Optional[float]  # IV percentile (0-100)
    iv_high: Optional[float]  # 52-week high
    iv_low: Optional[float]  # 52-week low
    source: str


@dataclass
class ExtendedFundamentals:
    """Extended fundamental data from yfinance."""
    symbol: str
    # Valuation
    enterprise_value: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    peg_ratio: Optional[float] = None
    # Profitability
    profit_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    # Financial health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    # Short interest
    short_ratio: Optional[float] = None
    short_percent_float: Optional[float] = None
    # Dividend
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    source: str = 'yfinance'


@dataclass
class AnalystData:
    """Analyst ratings and price targets."""
    symbol: str
    recommendation: Optional[str] = None  # e.g., 'buy', 'hold', 'sell'
    recommendation_mean: Optional[float] = None  # 1=strong buy, 5=sell
    num_analysts: Optional[int] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    upside_pct: Optional[float] = None  # % upside to mean target
    recent_upgrades: int = 0
    recent_downgrades: int = 0
    source: str = 'yfinance'


@dataclass
class InstitutionalHolding:
    """Institutional holder record."""
    holder: str
    shares: int
    value: Optional[float] = None
    pct_held: Optional[float] = None


@dataclass
class InstitutionalData:
    """Institutional ownership data."""
    symbol: str
    insider_pct: Optional[float] = None
    institutional_pct: Optional[float] = None
    top_holders: List[InstitutionalHolding] = None
    source: str = 'yfinance'

    def __post_init__(self):
        if self.top_holders is None:
            self.top_holders = []


@dataclass
class InsiderTransaction:
    """Single insider transaction."""
    insider: str
    relation: Optional[str] = None
    transaction_type: str = ''  # 'Buy' or 'Sale'
    shares: int = 0
    value: Optional[float] = None
    date: Optional[str] = None


@dataclass
class InsiderData:
    """Insider transaction summary."""
    symbol: str
    recent_buys: int = 0
    recent_sells: int = 0
    net_sentiment: str = 'neutral'  # 'bullish', 'bearish', 'neutral'
    total_buy_value: float = 0
    total_sell_value: float = 0
    transactions: List[InsiderTransaction] = None
    source: str = 'yfinance'

    def __post_init__(self):
        if self.transactions is None:
            self.transactions = []


@dataclass
class NewsItem:
    """Single news item."""
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    timestamp: Optional[int] = None


@dataclass
class NewsData:
    """News and sentiment data."""
    symbol: str
    items: List[NewsItem] = None
    sentiment: str = 'neutral'  # 'positive', 'negative', 'neutral'
    source: str = 'yfinance'

    def __post_init__(self):
        if self.items is None:
            self.items = []


@dataclass
class PeerComparison:
    """Peer/sector comparison data."""
    symbol: str
    sector: Optional[str] = None
    sector_etf: Optional[str] = None
    symbol_return_20d: Optional[float] = None
    sector_return_20d: Optional[float] = None
    vs_sector: Optional[float] = None  # Relative performance
    outperforming_sector: bool = False
    source: str = 'yfinance'


# ============================================================
# PROTOCOL (Interface Definition)
# ============================================================

@runtime_checkable
class DataProviderProtocol(Protocol):
    """
    Protocol for market data providers.

    Implementations must return typed results with source tracking.
    All methods have both sync and async variants.
    """

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote for a symbol."""
        ...

    def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 30
    ) -> Optional[Candles]:
        """Get historical OHLCV candles."""
        ...

    def get_atr(self, symbol: str, period: int = 14) -> Optional[ATRResult]:
        """Get Average True Range."""
        ...

    def get_quotes_bulk(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        ...


# ============================================================
# IMPLEMENTATION
# ============================================================

_CACHE_MISS = object()  # Sentinel for "not in cache"


# ------------------------------------------------------------
# yfinance health circuit breaker
# ------------------------------------------------------------
# When Yahoo's finance endpoints are unreachable, each yfinance call
# blocks for ~7s on DNS/TCP timeout. Since the scorer invokes 9+ yf
# calls per symbol across the universe, this turns Tier 1 into a
# 10-minute stall. Track consecutive failures and skip yf calls when
# the service looks dead.
import threading
_YF_LOCK = threading.Lock()
_YF_CONSECUTIVE_FAILURES = 0
_YF_COOLDOWN_UNTIL = 0.0  # unix ts; yf calls skipped until this time
_YF_FAIL_THRESHOLD = 5    # trip breaker after this many consecutive fails
_YF_COOLDOWN_SECS = 300   # skip yf calls for 5 minutes after tripping


def _yf_is_disabled() -> bool:
    """Return True when yfinance is in cooldown due to prior failures."""
    import time as _t
    with _YF_LOCK:
        return _t.time() < _YF_COOLDOWN_UNTIL


def _yf_record_success() -> None:
    global _YF_CONSECUTIVE_FAILURES, _YF_COOLDOWN_UNTIL
    with _YF_LOCK:
        _YF_CONSECUTIVE_FAILURES = 0
        _YF_COOLDOWN_UNTIL = 0.0


def _yf_record_failure(err: Exception) -> None:
    global _YF_CONSECUTIVE_FAILURES, _YF_COOLDOWN_UNTIL
    import time as _t
    msg = str(err).lower()
    # Only count network-level failures toward the breaker — symbol-not-found
    # or shape errors shouldn't trip it.  Rate-limit (429 / "too many
    # requests") counts: yfinance returns YFRateLimitError when throttled and
    # hammering through a rate-limit just prolongs the problem.
    network_markers = ("curl", "connect", "timeout", "ssl", "resolve", "network")
    ratelimit_markers = ("429", "too many requests", "rate limit", "ratelimit", "yfratelimiterror")
    if not any(k in msg for k in network_markers + ratelimit_markers):
        return
    # Rate-limit failures get a longer cooldown (5 min -> 15 min) because
    # yfinance rate-limit windows are typically 10-15 min wide.
    is_ratelimit = any(k in msg for k in ratelimit_markers)
    cooldown_secs = _YF_COOLDOWN_SECS * 3 if is_ratelimit else _YF_COOLDOWN_SECS
    with _YF_LOCK:
        _YF_CONSECUTIVE_FAILURES += 1
        if _YF_CONSECUTIVE_FAILURES >= _YF_FAIL_THRESHOLD and _YF_COOLDOWN_UNTIL < _t.time():
            _YF_COOLDOWN_UNTIL = _t.time() + cooldown_secs
            logger.warning(
                "yfinance appears unreachable (%d consecutive failures, "
                "reason=%s) — disabling yf calls for %ds",
                _YF_CONSECUTIVE_FAILURES,
                "rate-limit" if is_ratelimit else "network",
                cooldown_secs,
            )


class DataProvider:
    """
    Unified data provider wrapping MarketDataClient (official SDK).

    Data sources:
    - Market Data App (MDA) - quotes, candles, options
    - yfinance - fundamentals, analyst data, news, screening

    All responses include 'source' field for visibility.
    """

    def __init__(self):
        """Initialize the data provider."""
        # Import from layer2_data (moved from layer1_execution 2026-01-28)
        from data.marketdata_client import get_marketdata_client
        self._mda_client = get_marketdata_client()
        self._cache: Dict[str, tuple] = {}  # Per-type TTL cache
        # TTLs in seconds — fast data gets short TTL, slow data gets long TTL
        self._ttl_map = {
            'quote': 15,       # Prices move constantly
            'candles': 30,     # Bars update within the bar interval
            'atr': 60,         # Derived from candles, changes slowly
            'iv_info': 120,    # IV shifts slowly; 2-min cache covers ~4 rounds
            'option_chain': 180, # 3-min cache: Tier 2 reads share across rounds
            'option_chain_hist': 3600,
            'option_quote': 30,
            'option_quote_hist': 3600,
            'option_quote_series': 3600,
            'option_greeks': 30,
            # Fundamentals rarely change intraday — cache aggressively
            # so we don't pay yfinance roundtrip costs every scoring round.
            'fundamentals': 3600,
            'ext_fundamentals': 3600,
            'earnings': 3600,
            'earnings_hist': 3600,
            'analyst': 1800,
            'institutional': 3600,  # Quarterly filings
            'insider': 1800,
            'news': 300,       # Headlines useful for a few minutes
            'peer': 900,
            'screen': 60,      # Screens can refresh moderately
        }
        self._default_ttl = 60  # Fallback
        self._executor = ThreadPoolExecutor(max_workers=4)

    def get_mda_usage(self) -> Dict[str, Any]:
        """Return MarketData.app client stats (credits, circuit breaker, request counts)."""
        return self._mda_client.get_usage()

    @staticmethod
    def _normalize_expiration(exp: str) -> str:
        """Convert expiration to YYYYMMDD format for IBKR compatibility.
        
        Handles Unix timestamps (e.g. '1773432000') and YYYY-MM-DD.
        """
        if not exp:
            return exp
        exp = str(exp).strip()
        if exp.isdigit() and len(exp) >= 10:
            try:
                dt = datetime.utcfromtimestamp(int(exp))
                return dt.strftime('%Y%m%d')
            except (ValueError, OSError):
                return exp
        if len(exp) == 10 and exp[4] == '-' and exp[7] == '-':
            return exp.replace('-', '')
        return exp

    def _run_async(self, coro):
        """Run an async coroutine synchronously.

        Uses nest_asyncio to allow running coroutines even when an event
        loop is already running (e.g., when called from async context).
        """
        import nest_asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread - create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Patch to allow nested event loops
            nest_asyncio.apply(loop)

        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            logger.debug(f"_run_async failed: {type(e).__name__}: {e}")
            raise

    def _get_cached(self, key: str) -> Any:
        """Get cached value if not expired (uses per-type TTL).
        
        Returns _CACHE_MISS sentinel when key is not in cache.
        Returns the cached value (including None) when found.
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            data_type = key.split(':')[0] if ':' in key else key
            ttl = self._ttl_map.get(data_type, self._default_ttl)
            if (datetime.now() - timestamp).total_seconds() < ttl:
                return value
            del self._cache[key]
        return _CACHE_MISS

    def _set_cached(self, key: str, value: Any):
        """Set cached value with timestamp."""
        self._cache[key] = (value, datetime.now())

    # ==================== QUOTES ====================

    def _ibkr_source(self):
        """Return the process-wide IBKRQuoteSource, or None if disabled/unavailable."""
        try:
            from data.ibkr_quote_source import get_ibkr_quote_source
            return get_ibkr_quote_source()
        except Exception as e:
            logger.debug("IBKRQuoteSource lookup failed: %s", e)
            return None

    @staticmethod
    def _ibkr_quote_to_quote(symbol: str, q) -> "Quote":
        """Convert an IBKRQuote into the public Quote dataclass."""
        return Quote(
            symbol=symbol,
            last=q.last,
            bid=q.bid,
            ask=q.ask,
            volume=int(q.volume or 0),
            change_pct=None,  # IBKR ticker doesn't expose change_pct directly
            source="ibkr",
            timestamp=datetime.now(),
            source_updated=int(q.ts) if q.ts else None,
            auction_imbalance=getattr(q, "auction_imbalance", None),
            auction_volume=getattr(q, "auction_volume", None),
            auction_price=getattr(q, "auction_price", None),
            regulatory_imbalance=getattr(q, "regulatory_imbalance", None),
        )

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get real-time quote for a symbol.

        Routing: when IBKR_QUOTES_ENABLED is True (and IBKR is connected),
        we read from the IBKR streaming subscription. If IBKR has no data
        the call returns None — we DO NOT fall back to MDA's 15-min delayed
        quotes. Signals must abstain rather than trade on stale data.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')

        Returns:
            Quote with source tracking, or None if unavailable
        """
        cache_key = f"quote:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        # Real-time path: IBKR streaming
        ibkr_src = self._ibkr_source()
        if ibkr_src is not None:
            try:
                ibkr_q = self._run_async(ibkr_src.get_quote(symbol))
            except Exception as e:
                logger.debug("IBKR get_quote(%s) raised: %s", symbol, e)
                ibkr_q = None
            if ibkr_q is not None:
                quote = self._ibkr_quote_to_quote(symbol, ibkr_q)
                self._set_cached(cache_key, quote)
                return quote
            # IBKR enabled but no data — fail honest.  Do NOT fall back to MDA.
            logger.debug("IBKR returned no quote for %s; abstaining (no MDA fallback)", symbol)
            return None

        # Legacy path: MDA (only when IBKR is disabled).
        try:
            raw = self._run_async(self._mda_client.get_quote(symbol))

            if raw:
                quote = Quote(
                    symbol=symbol,
                    last=raw.get('last'),
                    bid=raw.get('bid'),
                    ask=raw.get('ask'),
                    volume=raw.get('volume', 0),
                    change_pct=raw.get('change_pct'),
                    source=raw.get('source', 'unknown'),
                    timestamp=datetime.now(),
                    source_updated=raw.get('updated')
                )
                self._set_cached(cache_key, quote)
                return quote
            else:
                logger.debug(f"No quote data for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to get quote for {symbol}: {e}")

        return None

    def get_quotes_bulk(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols efficiently.

        IBKR path (when enabled): per-symbol streaming reads.  Symbols with
        no real-time data are simply absent from the result (NOT filled
        with MDA delayed quotes).

        Args:
            symbols: List of stock tickers

        Returns:
            Dict mapping symbol -> Quote
        """
        results: Dict[str, Quote] = {}

        ibkr_src = self._ibkr_source()
        if ibkr_src is not None:
            for sym in symbols:
                try:
                    ibkr_q = self._run_async(ibkr_src.get_quote(sym))
                except Exception as e:
                    logger.debug("IBKR bulk get_quote(%s) raised: %s", sym, e)
                    continue
                if ibkr_q is not None:
                    results[sym] = self._ibkr_quote_to_quote(sym, ibkr_q)
            return results

        try:
            raw_quotes = self._run_async(self._mda_client.get_quotes_bulk(symbols))

            for symbol, raw in raw_quotes.items():
                if raw:
                    results[symbol] = Quote(
                        symbol=symbol,
                        last=raw.get('last'),
                        bid=raw.get('bid'),
                        ask=raw.get('ask'),
                        volume=raw.get('volume', 0),
                        change_pct=raw.get('change_pct'),
                        source=raw.get('source', 'unknown'),
                        timestamp=datetime.now(),
                        source_updated=raw.get('updated')  # Unix timestamp from MarketData API
                    )

        except Exception as e:
            logger.warning(f"Bulk quote fetch failed: {e}")

        return results

    async def get_quotes_bulk_async(self, symbols: List[str]) -> Dict[str, Quote]:
        """Async version of get_quotes_bulk — avoids nest_asyncio deadlock.

        Use from async callers sharing the main event loop.
        """
        results: Dict[str, Quote] = {}

        ibkr_src = self._ibkr_source()
        if ibkr_src is not None:
            for sym in symbols:
                try:
                    ibkr_q = await ibkr_src.get_quote(sym)
                except Exception as e:
                    logger.debug("IBKR bulk get_quote(%s) raised: %s", sym, e)
                    continue
                if ibkr_q is not None:
                    results[sym] = self._ibkr_quote_to_quote(sym, ibkr_q)
            return results

        try:
            raw_quotes = await self._mda_client.get_quotes_bulk(symbols)
            for symbol, raw in raw_quotes.items():
                if raw:
                    results[symbol] = Quote(
                        symbol=symbol,
                        last=raw.get('last'),
                        bid=raw.get('bid'),
                        ask=raw.get('ask'),
                        volume=raw.get('volume', 0),
                        change_pct=raw.get('change_pct'),
                        source=raw.get('source', 'unknown'),
                        timestamp=datetime.now(),
                        source_updated=raw.get('updated'),
                    )
        except Exception as e:
            logger.warning(f"Bulk quote fetch (async) failed: {e}")
        return results

    def get_candles_bulk(self, symbols: List[str]) -> Dict[str, Candles]:
        """
        Get daily candles for multiple symbols in a single API call.

        Args:
            symbols: List of stock tickers

        Returns:
            Dict mapping symbol -> Candles
        """
        results = {}

        try:
            raw_bulk = self._run_async(
                self._mda_client.get_bulk_daily_candles(symbols)
            )

            for symbol, raw in raw_bulk.items():
                if raw and raw.get('close'):
                    results[symbol] = Candles(
                        symbol=symbol,
                        open=raw.get('open', []),
                        high=raw.get('high', []),
                        low=raw.get('low', []),
                        close=raw.get('close', []),
                        volume=[int(v) for v in raw.get('volume', []) if v is not None],
                        timestamps=raw.get('timestamps', []),
                        source=raw.get('source', 'marketdata_bulk'),
                    )

        except Exception as e:
            logger.warning(f"Bulk candle fetch failed: {e}")

        return results

    # ==================== CANDLES ====================

    def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 30,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[Candles]:
        """
        Get historical OHLCV candles.

        Args:
            symbol: Stock ticker
            resolution: 'D' (daily), 'H' (hourly), '1'/'5'/'15' (minutes)
            days_back: Number of bars (when from_date not specified)
            from_date: Start date 'YYYY-MM-DD' (overrides days_back)
            to_date: End date 'YYYY-MM-DD'

        Returns:
            Candles with source tracking, or None if unavailable
        """
        cache_key = f"candles:{symbol}:{resolution}:{from_date or days_back}:{to_date}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            raw = self._run_async(
                self._mda_client.get_candles(
                    symbol, resolution, days_back,
                    from_date=from_date, to_date=to_date,
                )
            )

            if raw and raw.get('close'):
                candles = Candles(
                    symbol=symbol,
                    open=raw.get('open', []),
                    high=raw.get('high', []),
                    low=raw.get('low', []),
                    close=raw.get('close', []),
                    volume=[int(v) for v in raw.get('volume', [])],
                    timestamps=raw.get('timestamps', []),
                    source=raw.get('source', 'unknown')
                )
                self._set_cached(cache_key, candles)
                return candles
            else:
                logger.debug(f"No candle data available for {symbol}")
                # Negative-cache to avoid re-requesting the same failure
                self._set_cached(cache_key, None)

        except Exception as e:
            logger.warning(f"Failed to get candles for {symbol}: {e}")
            self._set_cached(cache_key, None)

        return None

    # ==================== Async candles (for callers already in an event loop) ====

    async def get_candles_async(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 30,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[Candles]:
        """Async version of get_candles — avoids nest_asyncio issues.

        Use this from async callers (e.g. research agent) that share the
        main event loop.  Directly awaits the marketdata_client coroutine.
        """
        cache_key = f"candles:{symbol}:{resolution}:{from_date or days_back}:{to_date}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            raw = await self._mda_client.get_candles(
                symbol, resolution, days_back,
                from_date=from_date, to_date=to_date,
            )

            if raw and raw.get('close'):
                candles = Candles(
                    symbol=symbol,
                    open=raw.get('open', []),
                    high=raw.get('high', []),
                    low=raw.get('low', []),
                    close=raw.get('close', []),
                    volume=[int(v) for v in raw.get('volume', [])],
                    timestamps=raw.get('timestamps', []),
                    source=raw.get('source', 'unknown')
                )
                self._set_cached(cache_key, candles)
                return candles
            else:
                logger.debug(f"No candle data available for {symbol}")
                self._set_cached(cache_key, None)

        except Exception as e:
            logger.warning(f"Failed to get candles for {symbol}: {e}")
            self._set_cached(cache_key, None)

        return None

    # ==================== ATR ====================

    def get_atr(self, symbol: str, period: int = 14) -> Optional[ATRResult]:
        """
        Get Average True Range for a symbol.

        Args:
            symbol: Stock ticker
            period: ATR period (default 14)

        Returns:
            ATRResult with value and source tracking
        """
        cache_key = f"atr:{symbol}:{period}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            value = self._run_async(self._mda_client.calculate_atr(symbol, period))

            if value is not None:
                result = ATRResult(
                    symbol=symbol,
                    value=value,
                    period=period,
                    source='marketdata'
                )
                self._set_cached(cache_key, result)
                return result
            else:
                logger.debug(f"No ATR data available for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to get ATR for {symbol}: {e}")

        return None

    # ==================== OPTIONS ====================

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        side: Optional[str] = None,
        strike_range: Optional[tuple] = None,
        dte_range: Optional[tuple] = None,
        date: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        # Server-side filters
        delta: Optional[float] = None,
        strike_limit: Optional[int] = None,
        range_filter: Optional[str] = None,
        min_bid: Optional[float] = None,
        max_bid_ask_spread_pct: Optional[float] = None,
        min_open_interest: Optional[int] = None,
        min_volume: Optional[int] = None,
    ) -> Optional[OptionChain]:
        """
        Get options chain with Greeks and IV.

        Args:
            symbol: Underlying ticker (e.g., 'AAPL')
            expiration: Specific expiration (YYYY-MM-DD) or None for all
            side: 'call', 'put', or None for both
            strike_range: (min_strike, max_strike) tuple or None for all
            dte_range: (min_dte, max_dte) tuple to filter expirations
            date: Historical snapshot date (YYYY-MM-DD)
            from_date: Start of historical range (YYYY-MM-DD)
            to_date: End of historical range (YYYY-MM-DD)
            delta: Server-side delta filter
            strike_limit: Max strikes per expiration
            range_filter: 'itm', 'otm', 'all'
            min_bid: Minimum bid filter
            max_bid_ask_spread_pct: Max bid-ask spread %
            min_open_interest: Minimum OI filter
            min_volume: Minimum volume filter

        Returns:
            OptionChain with contracts and source tracking
        """
        cache_prefix = 'option_chain_hist' if (date or from_date or to_date) else 'option_chain'
        cache_key = (
            f"{cache_prefix}:{symbol.upper()}:{expiration or '*'}:{side or '*'}:"
            f"{strike_range or '*'}:{dte_range or '*'}:{date or '*'}:{from_date or '*'}:{to_date or '*'}:"
            f"{delta or '*'}:{strike_limit or '*'}:{range_filter or '*'}:{min_bid or '*'}:"
            f"{max_bid_ask_spread_pct or '*'}:{min_open_interest or '*'}:{min_volume or '*'}"
        )
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            raw = self._run_async(
                self._mda_client.get_option_chain(
                    symbol,
                    expiration=expiration,
                    side=side,
                    strike_range=strike_range,
                    dte_range=dte_range,
                    date=date,
                    from_date=from_date,
                    to_date=to_date,
                    delta=delta,
                    strike_limit=strike_limit,
                    range_filter=range_filter,
                    min_bid=min_bid,
                    max_bid_ask_spread_pct=max_bid_ask_spread_pct,
                    min_open_interest=min_open_interest,
                    min_volume=min_volume,
                )
            )

            if raw and raw.get('contracts'):
                contracts = []
                for c in raw['contracts']:
                    contracts.append(OptionContract(
                        option_symbol=c.get('option_symbol', ''),
                        underlying=c.get('underlying', symbol),
                        strike=c.get('strike', 0),
                        side=c.get('side', ''),
                        expiration=self._normalize_expiration(str(c.get('expiration', ''))),
                        dte=c.get('dte'),
                        bid=c.get('bid'),
                        ask=c.get('ask'),
                        mid=c.get('mid'),
                        last=c.get('last'),
                        volume=c.get('volume', 0),
                        open_interest=c.get('open_interest', 0),
                        delta=c.get('delta'),
                        gamma=c.get('gamma'),
                        theta=c.get('theta'),
                        vega=c.get('vega'),
                        iv=c.get('iv'),
                        source=raw.get('source', 'unknown'),
                        as_of_date=raw.get('as_of_date'),
                    ))

                chain = OptionChain(
                    symbol=symbol,
                    contracts=contracts,
                    source=raw.get('source', 'unknown'),
                    is_historical=raw.get('is_historical', False),
                    as_of_date=raw.get('as_of_date'),
                )
                self._set_cached(cache_key, chain)
                return chain

            self._set_cached(cache_key, None)

        except Exception as e:
            logger.warning(f"Failed to get option chain for {symbol}: {e}")
            self._set_cached(cache_key, None)

        return None

    def get_option_quote(
        self,
        option_symbol: str,
        date: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Get a single option contract quote, optionally for a historical date.

        Args:
            option_symbol: OCC-format symbol (e.g. 'AAPL230120C00150000')
            date: Historical snapshot date (YYYY-MM-DD)
            from_date: Range start (YYYY-MM-DD)
            to_date: Range end (YYYY-MM-DD)

        Returns:
            Quote dict or None
        """
        cache_prefix = 'option_quote_hist' if (date or from_date or to_date) else 'option_quote'
        cache_key = f"{cache_prefix}:{option_symbol}:{date or '*'}:{from_date or '*'}:{to_date or '*'}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            quote = self._run_async(
                self._mda_client.get_option_quote(
                    option_symbol, date=date, from_date=from_date, to_date=to_date
                )
            )
            self._set_cached(cache_key, quote)
            return quote
        except Exception as e:
            logger.warning(f"Failed to get option quote for {option_symbol}: {e}")
            self._set_cached(cache_key, None)
            return None

    def get_option_quote_series(
        self,
        option_symbol: str,
        from_date: str,
        to_date: str,
    ) -> Optional[List[Dict]]:
        """
        Get a time series of daily option quotes between two dates.

        Args:
            option_symbol: OCC-format symbol
            from_date: Start date inclusive (YYYY-MM-DD)
            to_date: End date inclusive (YYYY-MM-DD)

        Returns:
            List of daily quote dicts sorted by date, or None
        """
        cache_key = f"option_quote_series:{option_symbol}:{from_date}:{to_date}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            series = self._run_async(
                self._mda_client.get_option_quote_series(option_symbol, from_date, to_date)
            )
            self._set_cached(cache_key, series)
            return series
        except Exception as e:
            logger.warning(f"Failed to get option quote series for {option_symbol}: {e}")
            self._set_cached(cache_key, None)
            return None

    def get_option_expirations(self, symbol: str) -> Optional[List[str]]:
        """
        Get available option expiration dates.

        Args:
            symbol: Underlying ticker

        Returns:
            List of expiration dates (YYYY-MM-DD format)
        """
        try:
            return self._run_async(self._mda_client.get_option_expirations(symbol))
        except Exception as e:
            logger.warning(f"Failed to get option expirations for {symbol}: {e}")
            return None

    def get_option_strikes(self, symbol: str, expiration: str) -> Optional[List[float]]:
        """
        Get available strikes for an expiration.

        Args:
            symbol: Underlying ticker
            expiration: Expiration date (YYYY-MM-DD)

        Returns:
            List of strike prices
        """
        try:
            return self._run_async(self._mda_client.get_option_strikes(symbol, expiration))
        except Exception as e:
            logger.warning(f"Failed to get option strikes for {symbol}: {e}")
            return None

    def get_option_lookup(self, symbol: str) -> Optional[list[str]]:
        """Look up OCC-format option symbols for an underlying."""
        try:
            return self._run_async(self._mda_client.get_option_lookup(symbol))
        except Exception as e:
            logger.warning(f"Failed option lookup for {symbol}: {e}")
            return None

    def find_option_by_delta(
        self,
        symbol: str,
        target_delta: float,
        side: str,
        min_dte: int = 21,
        max_dte: int = 45
    ) -> Optional[OptionContract]:
        """
        Find option contract closest to target delta.

        Args:
            symbol: Underlying ticker
            target_delta: Target delta (e.g., 0.30 for 30-delta)
            side: 'call' or 'put'
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration

        Returns:
            OptionContract closest to target delta, or None
        """
        try:
            raw = self._run_async(
                self._mda_client.find_option_by_delta(
                    symbol, target_delta, side, min_dte, max_dte
                )
            )

            if raw:
                return OptionContract(
                    option_symbol=raw.get('option_symbol', ''),
                    underlying=raw.get('underlying', symbol),
                    strike=raw.get('strike', 0),
                    side=raw.get('side', side),
                    expiration=str(raw.get('expiration', '')),
                    dte=raw.get('dte'),
                    bid=raw.get('bid'),
                    ask=raw.get('ask'),
                    mid=raw.get('mid'),
                    last=raw.get('last'),
                    volume=raw.get('volume', 0),
                    open_interest=raw.get('open_interest', 0),
                    delta=raw.get('delta'),
                    gamma=raw.get('gamma'),
                    theta=raw.get('theta'),
                    vega=raw.get('vega'),
                    iv=raw.get('iv'),
                    source=raw.get('source', 'unknown')
                )

        except Exception as e:
            logger.warning(f"Failed to find option by delta for {symbol}: {e}")

        return None

    def get_iv_info(
        self,
        symbol: str,
        dte_min: int = None,
        dte_max: int = None,
        strike_pct: Optional[float] = None,
    ) -> Optional[IVInfo]:
        """
        Get implied volatility information.

        Args:
            symbol: Underlying ticker
            dte_min: Minimum DTE for IV sampling (required by caller)
            dte_max: Maximum DTE for IV sampling (required by caller)
            strike_pct: Strike range as fraction of price, or None for
                        price-adaptive heuristic

        Returns:
            IVInfo with current IV and rank (if available)
        """
        if dte_min is None or dte_max is None:
            logger.warning(f"get_iv_info({symbol}): dte_min/dte_max required")
            return None

        try:
            raw = self._run_async(
                self._mda_client.get_iv_rank(
                    symbol, dte_min=dte_min, dte_max=dte_max,
                    strike_pct=strike_pct
                )
            )

            if raw:
                iv_current = raw.get('iv_current', 0)
                iv_high = raw.get('iv_high')
                iv_low = raw.get('iv_low')
                iv_rank = raw.get('iv_rank')

                # Snapshot today's IV into iv_history (best-effort, idempotent
                # within a UTC day) so the trailing-percentile path below has
                # data to work with over time.
                try:
                    from memory import record_iv_snapshot, compute_iv_rank_percentile
                    if iv_current:
                        record_iv_snapshot(symbol, iv_current,
                                           source=raw.get('source', 'marketdata'))
                    # Prefer true trailing percentile once enough history exists.
                    if iv_rank is None:
                        pct = compute_iv_rank_percentile(symbol, iv_current)
                        if pct is not None:
                            iv_rank = pct
                except Exception as snap_err:
                    logger.debug(f"iv_history plumbing failed for {symbol}: {snap_err}")

                # Compute iv_rank from iv_high/iv_low if not provided by API
                if iv_rank is None and iv_high and iv_low and iv_high != iv_low:
                    iv_rank = (iv_current - iv_low) / (iv_high - iv_low) * 100
                    iv_rank = max(0.0, min(100.0, iv_rank))

                # Fallback: estimate IV rank from historical candle volatility
                if iv_rank is None and iv_current:
                    iv_rank = self._estimate_iv_rank_from_candles(symbol, iv_current)

                return IVInfo(
                    symbol=symbol,
                    iv_current=iv_current,
                    iv_rank=iv_rank,
                    iv_high=iv_high,
                    iv_low=iv_low,
                    source=raw.get('source', 'unknown')
                )

        except Exception as e:
            logger.warning(f"Failed to get IV info for {symbol}: {e}")

        return None

    def _estimate_iv_rank_from_candles(self, symbol: str, iv_current: float) -> Optional[float]:
        """Estimate IV rank by comparing current IV to 52-week realized vol distribution."""
        try:
            import numpy as np
            candles = self.get_candles(symbol, days_back=252)
            if not candles or len(candles) < 60:
                return None
            closes = np.array(candles.close, dtype=float)
            log_returns = np.diff(np.log(closes))
            # Rolling 20-day realized vol windows
            rv_history = []
            for i in range(20, len(log_returns)):
                rv = np.std(log_returns[i - 20:i]) * np.sqrt(252) * 100
                rv_history.append(rv)
            if len(rv_history) < 10:
                return None
            # IV rank = percentile of current IV vs historical RV distribution
            rank = sum(1 for rv in rv_history if rv <= iv_current) / len(rv_history) * 100
            return round(max(0.0, min(100.0, rank)), 1)
        except Exception:
            return None

    # ==================== FUNDAMENTALS ====================

    def get_fundamentals(self, symbol: str) -> Optional[Fundamentals]:
        """
        Get basic fundamental data for a symbol.

        Note: Currently uses yfinance as MDA doesn't provide fundamentals.

        Args:
            symbol: Stock ticker

        Returns:
            Fundamentals with source tracking
        """
        cache_key = f"fundamentals:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        if _yf_is_disabled():
            return None

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            earnings_date = None
            if info.get('earningsTimestamp'):
                earnings_date = datetime.fromtimestamp(info['earningsTimestamp'])

            result = Fundamentals(
                symbol=symbol,
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                earnings_date=earnings_date,
                source='yfinance'
            )
            self._set_cached(cache_key, result)
            _yf_record_success()
            return result

        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
            _yf_record_failure(e)
            self._set_cached(cache_key, None)
            return None

    # ==================== EARNINGS (MDA) ====================

    def get_earnings_info(self, symbol: str) -> Optional[EarningsInfo]:
        """
        Get earnings calendar information via MarketData.app.

        Fetches the most recent + upcoming earnings report and derives
        next_earnings_date / days_until_earnings from reportDate.
        """
        cache_key = f"earnings:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            data = self._run_async(self._mda_client.get_earnings(symbol, countback=4))
            if not data or not data.get("earnings"):
                return None

            now = datetime.now()
            earnings_date = None
            days_until = None

            # Find the nearest future report_date
            for rec in data["earnings"]:
                rd = rec.get("report_date")
                if rd is None:
                    continue
                # report_date is a unix timestamp
                try:
                    dt = datetime.utcfromtimestamp(int(rd))
                except (ValueError, TypeError, OSError):
                    continue
                if dt >= now:
                    earnings_date = dt
                    days_until = (dt.date() - now.date()).days
                    break

            # If no future date found, use the most recent as reference
            if earnings_date is None and data["earnings"]:
                rd = data["earnings"][-1].get("report_date")
                if rd is not None:
                    try:
                        dt = datetime.utcfromtimestamp(int(rd))
                        earnings_date = dt
                        days_until = (dt.date() - now.date()).days
                    except (ValueError, TypeError, OSError):
                        pass

            result = EarningsInfo(
                symbol=symbol,
                next_earnings_date=earnings_date,
                days_until_earnings=days_until,
                source='marketdata',
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.debug(f"Failed to get earnings for {symbol}: {e}")
            return None

    def get_earnings_history(self, symbol: str, countback: int = 8) -> Optional[list]:
        """
        Get historical earnings records (EPS, surprise, report dates).

        Tries MDA first, then falls back to yfinance.
        Returns list of dicts with surprise_eps_pct etc.
        """
        cache_key = f"earnings_hist:{symbol}:{countback}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        # Try MDA first
        try:
            data = self._run_async(self._mda_client.get_earnings(symbol, countback=countback))
            if data and data.get("earnings"):
                # Only use MDA if at least one record has surprise data
                has_surprise = any(
                    r.get("surprise_eps_pct") is not None for r in data["earnings"]
                )
                if has_surprise:
                    self._set_cached(cache_key, data["earnings"])
                    return data["earnings"]
        except Exception as e:
            logger.debug(f"MDA earnings history for {symbol}: {e}")

        # Fallback: yfinance earnings history (with timeout to prevent hangs)
        try:
            import concurrent.futures
            def _fetch_yf_earnings():
                ticker = yf.Ticker(symbol)
                return ticker.earnings_dates
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                hist = pool.submit(_fetch_yf_earnings).result(timeout=15)
            if hist is not None and not hist.empty:
                records = []
                for idx, row in hist.head(countback).iterrows():
                    surprise_pct = None
                    if "Surprise(%)" in row.index and pd.notna(row.get("Surprise(%)")):
                        surprise_pct = float(row["Surprise(%)"])
                    records.append({
                        "report_date": int(idx.timestamp()) if hasattr(idx, "timestamp") else None,
                        "reported_eps": float(row["Reported EPS"]) if pd.notna(row.get("Reported EPS")) else None,
                        "estimated_eps": float(row["EPS Estimate"]) if pd.notna(row.get("EPS Estimate")) else None,
                        "surprise_eps_pct": surprise_pct,
                    })
                if records:
                    self._set_cached(cache_key, records)
                    return records
        except Exception as e:
            logger.debug(f"yfinance earnings history for {symbol}: {e}")

        return None

    def get_sector(self, symbol: str) -> Optional[str]:
        """
        Get sector for a symbol (convenience method).

        Returns:
            Sector name or None
        """
        fundamentals = self.get_fundamentals(symbol)
        return fundamentals.sector if fundamentals else None

    # ==================== CONVENIENCE METHODS ====================

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price (convenience method).

        Returns:
            Current price or None
        """
        quote = self.get_quote(symbol)
        return quote.last if quote else None

    def get_atr_percent(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Get ATR as percentage of current price.

        Returns:
            ATR percentage (e.g., 2.5 for 2.5%) or None
        """
        atr = self.get_atr(symbol, period)
        price = self.get_price(symbol)

        if atr and price and price > 0:
            return round((atr.value / price) * 100, 2)
        return None

    # ==================== EXTENDED DATA (yfinance) ====================

    def get_extended_fundamentals(self, symbol: str) -> Optional[ExtendedFundamentals]:
        """
        Get extended fundamental data.

        Includes valuation, profitability, growth, and financial health metrics.
        """
        cache_key = f"ext_fundamentals:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        if _yf_is_disabled():
            return None

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not isinstance(info, dict):
                info = {}

            result = ExtendedFundamentals(
                symbol=symbol,
                enterprise_value=info.get('enterpriseValue'),
                ev_to_ebitda=info.get('enterpriseToEbitda'),
                ev_to_revenue=info.get('enterpriseToRevenue'),
                price_to_book=info.get('priceToBook'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),
                peg_ratio=info.get('pegRatio'),
                profit_margin=info.get('profitMargins'),
                return_on_equity=info.get('returnOnEquity'),
                return_on_assets=info.get('returnOnAssets'),
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                quick_ratio=info.get('quickRatio'),
                free_cash_flow=info.get('freeCashflow'),
                short_ratio=info.get('shortRatio'),
                short_percent_float=info.get('shortPercentOfFloat'),
                dividend_yield=info.get('dividendYield'),
                beta=info.get('beta'),
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get extended fundamentals for {symbol}: {e}")
            _yf_record_failure(e)
            self._set_cached(cache_key, None)
            return None

    def get_analyst_data(self, symbol: str) -> Optional[AnalystData]:
        """
        Get analyst ratings and price targets.
        """
        cache_key = f"analyst:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        if _yf_is_disabled():
            return None

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not isinstance(info, dict):
                info = {}

            # Calculate upside
            current = info.get('currentPrice') or info.get('regularMarketPrice')
            target_mean = info.get('targetMeanPrice')
            upside = None
            if current and target_mean:
                upside = round((target_mean / current - 1) * 100, 1)

            # Count recent upgrades/downgrades
            upgrades = 0
            downgrades = 0
            try:
                recs = ticker.recommendations
                if recs is not None and not recs.empty:
                    recent = recs.tail(30)
                    upgrades = len(recent[recent['To Grade'].str.contains('Buy|Strong|Outperform', case=False, na=False)])
                    downgrades = len(recent[recent['To Grade'].str.contains('Sell|Underperform|Hold', case=False, na=False)])
            except Exception:
                pass

            result = AnalystData(
                symbol=symbol,
                recommendation=info.get('recommendationKey'),
                recommendation_mean=info.get('recommendationMean'),
                num_analysts=info.get('numberOfAnalystOpinions'),
                target_high=info.get('targetHighPrice'),
                target_low=info.get('targetLowPrice'),
                target_mean=target_mean,
                target_median=info.get('targetMedianPrice'),
                upside_pct=upside,
                recent_upgrades=upgrades,
                recent_downgrades=downgrades,
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get analyst data for {symbol}: {e}")
            _yf_record_failure(e)
            self._set_cached(cache_key, None)
            return None

    def get_institutional_data(self, symbol: str) -> Optional[InstitutionalData]:
        """
        Get institutional ownership data.
        """
        cache_key = f"institutional:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        if _yf_is_disabled():
            return None

        try:
            ticker = yf.Ticker(symbol)

            insider_pct = None
            institutional_pct = None
            top_holders = []

            # Major holders percentages
            try:
                major = ticker.major_holders
                if major is not None and not major.empty:
                    insider_pct = major.iloc[0, 0] if len(major) > 0 else None
                    institutional_pct = major.iloc[1, 0] if len(major) > 1 else None
            except Exception:
                pass

            # Top institutional holders
            try:
                holders = ticker.institutional_holders
                if holders is not None and not holders.empty:
                    for _, row in holders.head(5).iterrows():
                        top_holders.append(InstitutionalHolding(
                            holder=row.get('Holder', 'Unknown'),
                            shares=int(row.get('Shares', 0)),
                            value=row.get('Value'),
                            pct_held=row.get('% Out'),
                        ))
            except Exception:
                pass

            result = InstitutionalData(
                symbol=symbol,
                insider_pct=insider_pct,
                institutional_pct=institutional_pct,
                top_holders=top_holders,
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get institutional data for {symbol}: {e}")
            _yf_record_failure(e)
            self._set_cached(cache_key, None)
            return None

    def get_insider_data(self, symbol: str) -> Optional[InsiderData]:
        """
        Get insider transaction data.
        """
        cache_key = f"insider:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        if _yf_is_disabled():
            return None

        try:
            ticker = yf.Ticker(symbol)

            buys = 0
            sells = 0
            buy_value = 0
            sell_value = 0
            transactions = []

            try:
                txns = ticker.insider_transactions
                if txns is not None and not txns.empty:
                    recent = txns.head(10)

                    for _, row in recent.iterrows():
                        # yfinance often returns an empty 'Transaction' column;
                        # the actual buy/sell classification lives in 'Text'
                        # (e.g. "Sale at price 69.57 - 70.17 per share." or
                        # "Purchase at price 12.34 per share.").  Fall back to
                        # 'Position'/'Ownership' hints only if both are empty.
                        txn_type = str(row.get('Transaction', '') or '').strip()
                        text = str(row.get('Text', '') or '').strip()
                        combined = f"{txn_type} {text}".lower()
                        is_buy = ('buy' in combined or 'purchase' in combined
                                  or 'acquired' in combined)
                        is_sell = ('sale' in combined or 'sell' in combined
                                   or 'disposed' in combined or 'sold' in combined)

                        # Guard against nan values from yfinance
                        raw_value = row.get('Value', 0)
                        try:
                            import math
                            if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
                                raw_value = 0
                        except Exception:
                            raw_value = 0

                        if is_buy and not is_sell:
                            buys += 1
                            buy_value += raw_value or 0
                        elif is_sell and not is_buy:
                            sells += 1
                            sell_value += raw_value or 0

                        transactions.append(InsiderTransaction(
                            insider=row.get('Insider', 'Unknown'),
                            relation=row.get('Relationship') or row.get('Position'),
                            transaction_type='Buy' if is_buy and not is_sell
                                             else 'Sale' if is_sell and not is_buy
                                             else (txn_type or 'Unknown'),
                            shares=int(row.get('Shares', 0) or 0),
                            value=raw_value if raw_value else None,
                        ))
            except Exception:
                pass

            sentiment = 'bullish' if buys > sells else 'bearish' if sells > buys else 'neutral'

            result = InsiderData(
                symbol=symbol,
                recent_buys=buys,
                recent_sells=sells,
                net_sentiment=sentiment,
                total_buy_value=buy_value,
                total_sell_value=sell_value,
                transactions=transactions,
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get insider data for {symbol}: {e}")
            _yf_record_failure(e)
            self._set_cached(cache_key, None)
            return None

    def get_news(self, symbol: str) -> Optional[NewsData]:
        """
        Get recent news and basic sentiment via MarketData.app.
        """
        cache_key = f"news:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            data = self._run_async(self._mda_client.get_news(symbol, countback=10))
            if not data or not data.get("articles"):
                return None

            positive_words = ['surge', 'jump', 'rise', 'gain', 'beat', 'strong', 'upgrade', 'buy']
            negative_words = ['fall', 'drop', 'decline', 'miss', 'weak', 'downgrade', 'sell', 'crash']

            items = []
            all_titles = []
            for article in data["articles"]:
                headline = article.get("headline") or ""
                items.append(NewsItem(
                    title=headline[:200],
                    publisher=None,  # MDA doesn't provide publisher name
                    link=article.get("source"),
                    timestamp=article.get("publication_date"),
                ))
                all_titles.append(headline.lower())

            combined = ' '.join(all_titles)
            pos_count = sum(1 for w in positive_words if w in combined)
            neg_count = sum(1 for w in negative_words if w in combined)
            sentiment = 'positive' if pos_count > neg_count else 'negative' if neg_count > pos_count else 'neutral'

            result = NewsData(
                symbol=symbol,
                items=items,
                sentiment=sentiment,
                source='marketdata',
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get news for {symbol}: {e}")
            return None

    def get_peer_comparison(self, symbol: str) -> Optional[PeerComparison]:
        """
        Compare symbol performance to sector peers.

        Uses yfinance for sector identification, MDA daily candles for performance.
        """
        cache_key = f"peer:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        try:
            # Sector lookup still from yfinance (MDA has no fundamentals)
            fundamentals = self.get_fundamentals(symbol)
            sector = fundamentals.sector if fundamentals else None

            if not sector:
                return PeerComparison(symbol=symbol)

            SECTOR_ETFS = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Industrials': 'XLI',
                'Consumer Defensive': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Basic Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC',
            }

            sector_etf = SECTOR_ETFS.get(sector)
            if not sector_etf:
                return PeerComparison(symbol=symbol, sector=sector)

            # Fetch 20-day daily candles from MDA for both symbol and sector ETF
            from_date = (datetime.now() - timedelta(days=35)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            sym_candles = self.get_candles(symbol, resolution='D', from_date=from_date, to_date=to_date)
            etf_candles = self.get_candles(sector_etf, resolution='D', from_date=from_date, to_date=to_date)

            if not sym_candles or not etf_candles or len(sym_candles) < 2 or len(etf_candles) < 2:
                return PeerComparison(symbol=symbol, sector=sector, sector_etf=sector_etf)

            sym_return = (sym_candles.close[-1] / sym_candles.close[0] - 1) * 100
            etf_return = (etf_candles.close[-1] / etf_candles.close[0] - 1) * 100
            vs_sector = sym_return - etf_return

            result = PeerComparison(
                symbol=symbol,
                sector=sector,
                sector_etf=sector_etf,
                symbol_return_20d=round(float(sym_return), 2),
                sector_return_20d=round(float(etf_return), 2),
                vs_sector=round(float(vs_sector), 2),
                outperforming_sector=vs_sector > 0,
                source='marketdata',
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get peer comparison for {symbol}: {e}")
            return None

    # =========================================================================
    # SCREENING - Delegated to ScreenerLibrary
    # =========================================================================
    # All screening is handled by src.layer2_data.screener_library
    # which provides:
    # - Finviz as primary source

# ============================================================
# SINGLETON ACCESS
# ============================================================

_provider: Optional[DataProvider] = None


def install_data_provider(provider: DataProvider | None) -> None:
    """Pin a provider instance (``None`` clears for next ``get_data_provider()``)."""
    global _provider
    _provider = provider


def get_data_provider() -> DataProvider:
    """
    Get the singleton DataProvider instance.

    Usage:
        from data.data_provider import get_data_provider
        provider = get_data_provider()
        quote = provider.get_quote('AAPL')
    """
    global _provider
    if _provider is None:
        _provider = DataProvider()
    return _provider
