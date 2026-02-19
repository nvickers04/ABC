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
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

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
    """Historical OHLCV data with source tracking."""
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
            'iv_info': 60,     # IV updates with options flow
            'option_chain': 30, # Chains shift with underlying
            'option_greeks': 30,
            'fundamentals': 300,   # Sector/PE/earnings date rarely change
            'earnings': 300,
            'extended_fundamentals': 300,
            'analysts': 300,
            'institutional': 600,  # Quarterly filings
            'insider': 600,
            'news': 120,       # Headlines useful for a few minutes
            'peer_comparison': 300,
            'screen': 60,      # Screens can refresh moderately
        }
        self._default_ttl = 60  # Fallback
        self._executor = ThreadPoolExecutor(max_workers=4)

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

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired (uses per-type TTL)."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            data_type = key.split(':')[0] if ':' in key else key
            ttl = self._ttl_map.get(data_type, self._default_ttl)
            if (datetime.now() - timestamp).seconds < ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Set cached value with timestamp."""
        self._cache[key] = (value, datetime.now())

    # ==================== QUOTES ====================

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')

        Returns:
            Quote with source tracking, or None if unavailable
        """
        cache_key = f"quote:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

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

        Args:
            symbols: List of stock tickers

        Returns:
            Dict mapping symbol -> Quote
        """
        results = {}

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

    # ==================== CANDLES ====================

    def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 30
    ) -> Optional[Candles]:
        """
        Get historical OHLCV candles.

        Args:
            symbol: Stock ticker
            resolution: 'D' (daily), 'H' (hourly), '1'/'5'/'15' (minutes)
            days_back: Number of days of history

        Returns:
            Candles with source tracking, or None if unavailable
        """
        cache_key = f"candles:{symbol}:{resolution}:{days_back}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            raw = self._run_async(
                self._mda_client.get_candles(symbol, resolution, days_back)
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
                logger.warning(f"No candle data available for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to get candles for {symbol}: {e}")

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
        if cached:
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
        dte_range: Optional[tuple] = None
    ) -> Optional[OptionChain]:
        """
        Get options chain with Greeks and IV.

        Args:
            symbol: Underlying ticker (e.g., 'AAPL')
            expiration: Specific expiration (YYYY-MM-DD) or None for all
            side: 'call', 'put', or None for both
            strike_range: (min_strike, max_strike) tuple or None for all
            dte_range: (min_dte, max_dte) tuple to filter expirations

        Returns:
            OptionChain with contracts and source tracking
        """
        try:
            raw = self._run_async(
                self._mda_client.get_option_chain(
                    symbol,
                    expiration=expiration,
                    side=side,
                    strike_range=strike_range,
                    dte_range=dte_range
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
                        source=raw.get('source', 'unknown')
                    ))

                return OptionChain(
                    symbol=symbol,
                    contracts=contracts,
                    source=raw.get('source', 'unknown')
                )

        except Exception as e:
            logger.warning(f"Failed to get option chain for {symbol}: {e}")

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
                return IVInfo(
                    symbol=symbol,
                    iv_current=raw.get('iv_current', 0),
                    iv_rank=raw.get('iv_rank'),
                    iv_high=raw.get('iv_high'),
                    iv_low=raw.get('iv_low'),
                    source=raw.get('source', 'unknown')
                )

        except Exception as e:
            logger.warning(f"Failed to get IV info for {symbol}: {e}")

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
        if cached:
            return cached

        try:
            import yfinance as yf

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
            return result

        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
            return None

    # ==================== EARNINGS ====================

    def get_earnings_info(self, symbol: str) -> Optional[EarningsInfo]:
        """
        Get earnings calendar information for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            EarningsInfo with next earnings date and days until
        """
        cache_key = f"earnings:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            earnings_date = None
            
            # yfinance returns dict in newer versions, DataFrame in older
            if isinstance(calendar, dict):
                # Dict format: {'Earnings Date': [Timestamp, ...], ...}
                earnings_dates = calendar.get('Earnings Date')
                if not earnings_dates:
                    # Also try lowercase or alternate keys
                    earnings_dates = calendar.get('earnings_date') or calendar.get('Earnings Dates')
                if earnings_dates:
                    raw = earnings_dates[0] if isinstance(earnings_dates, list) else earnings_dates
                    if hasattr(raw, 'to_pydatetime'):
                        earnings_date = raw.to_pydatetime()
                    elif isinstance(raw, str):
                        from dateutil.parser import parse as dateparse
                        earnings_date = dateparse(raw)
                    else:
                        earnings_date = raw
            elif calendar is not None and hasattr(calendar, 'empty') and not calendar.empty:
                # DataFrame format (older yfinance)
                if 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date'].iloc[0]
                    if hasattr(earnings_date, 'to_pydatetime'):
                        earnings_date = earnings_date.to_pydatetime()

            days_until = None
            if earnings_date:
                # Normalize: earnings_date may be date or datetime
                import datetime as dt_module
                if isinstance(earnings_date, dt_module.datetime):
                    days_until = (earnings_date - datetime.now()).days
                elif isinstance(earnings_date, dt_module.date):
                    days_until = (earnings_date - datetime.now().date()).days
                else:
                    # Last resort: try string conversion
                    try:
                        days_until = (earnings_date - datetime.now()).days
                    except TypeError:
                        days_until = None

            result = EarningsInfo(
                symbol=symbol,
                next_earnings_date=earnings_date,
                days_until_earnings=days_until,
                source='yfinance'
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.debug(f"Failed to get earnings for {symbol}: {e}")
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
        if cached:
            return cached

        try:
            import yfinance as yf
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
            return None

    def get_analyst_data(self, symbol: str) -> Optional[AnalystData]:
        """
        Get analyst ratings and price targets.
        """
        cache_key = f"analyst:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf
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
            return None

    def get_institutional_data(self, symbol: str) -> Optional[InstitutionalData]:
        """
        Get institutional ownership data.
        """
        cache_key = f"institutional:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf
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
            return None

    def get_insider_data(self, symbol: str) -> Optional[InsiderData]:
        """
        Get insider transaction data.
        """
        cache_key = f"insider:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf
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
                        txn_type = str(row.get('Transaction', ''))
                        is_buy = 'Buy' in txn_type or 'Purchase' in txn_type
                        is_sell = 'Sale' in txn_type or 'Sell' in txn_type

                        if is_buy:
                            buys += 1
                            buy_value += row.get('Value', 0) or 0
                        elif is_sell:
                            sells += 1
                            sell_value += row.get('Value', 0) or 0

                        transactions.append(InsiderTransaction(
                            insider=row.get('Insider', 'Unknown'),
                            relation=row.get('Relationship'),
                            transaction_type='Buy' if is_buy else 'Sale' if is_sell else txn_type,
                            shares=int(row.get('Shares', 0)),
                            value=row.get('Value'),
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
            return None

    def get_news(self, symbol: str) -> Optional[NewsData]:
        """
        Get recent news and basic sentiment.
        """
        cache_key = f"news:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            items = []
            sentiment = 'neutral'

            try:
                news = ticker.news
                if news:
                    positive_words = ['surge', 'jump', 'rise', 'gain', 'beat', 'strong', 'upgrade', 'buy']
                    negative_words = ['fall', 'drop', 'decline', 'miss', 'weak', 'downgrade', 'sell', 'crash']

                    all_titles = []
                    for item in news[:10]:
                        title = item.get('title', '')
                        items.append(NewsItem(
                            title=title[:200],
                            publisher=item.get('publisher'),
                            link=item.get('link'),
                            timestamp=item.get('providerPublishTime'),
                        ))
                        all_titles.append(title.lower())

                    combined = ' '.join(all_titles)
                    pos_count = sum(1 for w in positive_words if w in combined)
                    neg_count = sum(1 for w in negative_words if w in combined)
                    sentiment = 'positive' if pos_count > neg_count else 'negative' if neg_count > pos_count else 'neutral'
            except Exception:
                pass

            result = NewsData(
                symbol=symbol,
                items=items,
                sentiment=sentiment,
            )
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get news for {symbol}: {e}")
            return None

    def get_peer_comparison(self, symbol: str) -> Optional[PeerComparison]:
        """
        Compare symbol performance to sector peers.

        Uses sector ETFs to measure relative performance over 20 days.
        """
        cache_key = f"peer:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector')

            if not sector:
                return PeerComparison(symbol=symbol)

            # Sector ETF mapping
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

            # Compare 20-day performance
            symbol_data = yf.download(symbol, period='1mo', progress=False)
            etf_data = yf.download(sector_etf, period='1mo', progress=False)

            if symbol_data.empty or etf_data.empty:
                return PeerComparison(symbol=symbol, sector=sector, sector_etf=sector_etf)

            symbol_return = (symbol_data['Close'].iloc[-1] / symbol_data['Close'].iloc[0] - 1) * 100
            sector_return = (etf_data['Close'].iloc[-1] / etf_data['Close'].iloc[0] - 1) * 100
            vs_sector = symbol_return - sector_return

            result = PeerComparison(
                symbol=symbol,
                sector=sector,
                sector_etf=sector_etf,
                symbol_return_20d=round(float(symbol_return), 2),
                sector_return_20d=round(float(sector_return), 2),
                vs_sector=round(float(vs_sector), 2),
                outperforming_sector=vs_sector > 0,
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


# ============================================================
# CONVENIENCE FUNCTIONS (for migration)
# ============================================================

def get_quote(symbol: str) -> Optional[Quote]:
    """Convenience function to get a quote."""
    return get_data_provider().get_quote(symbol)


def get_atr(symbol: str, period: int = 14) -> Optional[ATRResult]:
    """Convenience function to get ATR."""
    return get_data_provider().get_atr(symbol, period)


def get_candles(symbol: str, resolution: str = 'D', days_back: int = 30) -> Optional[Candles]:
    """Convenience function to get candles."""
    return get_data_provider().get_candles(symbol, resolution, days_back)


def get_price(symbol: str) -> Optional[float]:
    """Convenience function to get current price."""
    return get_data_provider().get_price(symbol)
