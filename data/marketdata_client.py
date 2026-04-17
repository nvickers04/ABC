"""
Market Data App Client - Primary market data source.

Uses direct HTTP calls to the MarketData.app API.
The official SDK has pydantic-settings conflicts with multi-app .env files,
so we use httpx directly instead.

Provides:
- Stock quotes, historical data, ATR calculations
- Options chains with Greeks and IV

API Docs: https://www.marketdata.app/docs/api

Note: Moved from layer1_execution to layer2_data (2026-01-28)
      This is data access, not execution logic.
"""

import logging
import asyncio
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

from core.async_utils import safe_sleep as _safe_sleep
import httpx

logger = logging.getLogger(__name__)

# API Base URL
API_BASE = "https://api.marketdata.app/v1"
_OPTIONS_RETRY_STATUSES = {429, 500, 502, 503, 504}
_OPTIONS_MAX_CONCURRENCY = 3
_OPTIONS_RETRY_ATTEMPTS = 3
_MDA_MAX_CONCURRENT = 45  # MDA hard limit is 50; keep 5 headroom


class MarketDataClient:
    """
    Async client for the Market Data App API.

    Features:
    - Real-time quotes (bid/ask/last/volume)
    - Historical candles (OHLCV)
    - ATR calculation from candles
    - Options chains with Greeks
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client.

        Args:
            api_key: API key (uses MARKETDATA_TOKEN env var if not provided)
        """
        self.api_key = api_key or os.environ.get('MARKETDATA_TOKEN') or os.environ.get('MARKETDATA_API_KEY')
        # Per-loop HTTP clients and semaphores keyed by id(loop).
        # A single shared client bound to one loop is unsafe when worker
        # threads (asyncio.to_thread) run nested asyncio.run / run_until_complete
        # with their own loops — they would otherwise race to close and
        # recreate the main-loop client, corrupting in-flight requests.
        self._http_clients: Dict[int, httpx.AsyncClient] = {}
        self._options_semaphores: Dict[int, asyncio.Semaphore] = {}
        self._global_semaphores: Dict[int, asyncio.Semaphore] = {}
        self._request_count = 0
        self._daily_count = 0
        self._daily_reset: Optional[datetime] = None  # Resets at midnight UTC
        self._last_request_time: Optional[datetime] = None
        # Credit tracking from MDA response headers
        self._credits_remaining: Optional[int] = None
        self._credits_limit: Optional[int] = None
        self._credits_reset: Optional[int] = None  # UTC epoch seconds
        self._low_credit_warned: bool = False

        if not self.api_key:
            logger.warning("No Market Data App API key configured")

    def _get_http_client(self) -> Optional[httpx.AsyncClient]:
        """Get or create HTTP client for the current event loop.

        Uses a per-loop cache keyed by id(loop) so that worker threads
        running their own event loops (via asyncio.to_thread → _run_async)
        do not clobber the main loop's client.
        """
        if not self.api_key:
            return None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — cannot create an AsyncClient safely.
            return None
        loop_id = id(current_loop)
        client = self._http_clients.get(loop_id)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                base_url=API_BASE,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
                limits=httpx.Limits(
                    max_connections=_MDA_MAX_CONCURRENT,
                    max_keepalive_connections=25,
                    keepalive_expiry=30,
                ),
            )
            self._http_clients[loop_id] = client
        return client

    def _get_global_semaphore(self) -> asyncio.Semaphore:
        """Global semaphore enforcing MDA's 50 concurrent request limit (per loop)."""
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = 0
        sem = self._global_semaphores.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(_MDA_MAX_CONCURRENT)
            self._global_semaphores[loop_id] = sem
        return sem

    def _get_options_semaphore(self) -> asyncio.Semaphore:
        """Per-event-loop semaphore for high-volume option endpoints."""
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = 0
        sem = self._options_semaphores.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(_OPTIONS_MAX_CONCURRENCY)
            self._options_semaphores[loop_id] = sem
        return sem

    async def _get_with_retries(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        label: str,
        throttle_options: bool = False,
    ) -> Optional[httpx.Response]:
        """Issue a GET with bounded retry/backoff for transient option-data failures."""
        client = self._get_http_client()
        if not client:
            return None

        async def _do_request() -> Optional[httpx.Response]:
            for attempt in range(_OPTIONS_RETRY_ATTEMPTS):
                try:
                    if self._track_request():
                        return None
                    async with self._get_global_semaphore():
                        response = await client.get(path, params=params)
                    self._parse_rate_headers(response)
                    if response.status_code not in _OPTIONS_RETRY_STATUSES or attempt == _OPTIONS_RETRY_ATTEMPTS - 1:
                        return response
                    logger.debug(
                        f"Retrying {label} after HTTP {response.status_code} "
                        f"({attempt + 1}/{_OPTIONS_RETRY_ATTEMPTS})"
                    )
                except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout, httpx.ReadTimeout,
                        httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.NetworkError) as exc:
                    if attempt == _OPTIONS_RETRY_ATTEMPTS - 1:
                        raise
                    logger.debug(
                        f"Retrying {label} after transport error "
                        f"({attempt + 1}/{_OPTIONS_RETRY_ATTEMPTS}): {exc}"
                    )
                await _safe_sleep(0.5 * (attempt + 1))
            return None

        if not throttle_options:
            return await _do_request()

        async with self._get_options_semaphore():
            return await _do_request()

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    async def close(self):
        """Close the HTTP client bound to the current event loop.

        Safe to call from any loop; only touches the cached client for
        the running loop. Use `close_all()` to close every cached client
        (must be invoked on each loop that owns one).
        """
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            return
        client = self._http_clients.pop(loop_id, None)
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass

    async def close_all(self):
        """Close every cached HTTP client; caller must run on each loop."""
        clients = list(self._http_clients.values())
        self._http_clients.clear()
        for client in clients:
            try:
                await client.aclose()
            except Exception:
                pass

    def _parse_rate_headers(self, response: httpx.Response) -> None:
        """Extract MDA rate-limit headers and warn when credits run low."""
        try:
            remaining = response.headers.get('X-Api-Ratelimit-Remaining')
            limit = response.headers.get('X-Api-Ratelimit-Limit')
            reset = response.headers.get('X-Api-Ratelimit-Reset')
            if remaining is not None:
                self._credits_remaining = int(remaining)
            if limit is not None:
                self._credits_limit = int(limit)
            if reset is not None:
                self._credits_reset = int(reset)
            # Warn once when credits drop below 10%
            if self._credits_remaining is not None and self._credits_limit:
                pct = self._credits_remaining / self._credits_limit
                if pct < 0.10 and not self._low_credit_warned:
                    logger.warning(
                        f"MDA credits low: {self._credits_remaining:,} / {self._credits_limit:,} remaining ({pct:.1%})"
                    )
                    self._low_credit_warned = True
                elif pct >= 0.10:
                    self._low_credit_warned = False
        except (ValueError, TypeError):
            pass

    def _track_request(self) -> bool:
        """
        Track request count for observability.
        Resets daily counter at 9:30 AM ET (MDA reset time).
        """
        # ``datetime.utcnow()`` is deprecated in Python 3.12+; use a tz-aware
        # UTC instant instead.  ``.date()`` comparisons below still work.
        now = datetime.now(timezone.utc)

        # Reset daily counter at midnight UTC
        if self._daily_reset is None or now.date() > self._daily_reset.date():
            self._daily_count = 0
            self._daily_reset = now
        
        self._request_count += 1
        self._daily_count += 1
        self._last_request_time = now
        
        return False

    def get_usage(self) -> Dict[str, Any]:
        """Get API usage stats."""
        return {
            'total_requests': self._request_count,
            'daily_requests': self._daily_count,
            'last_request': self._last_request_time.isoformat() if self._last_request_time else None,
            'mda_credits_remaining': self._credits_remaining,
            'mda_credits_limit': self._credits_limit,
            'mda_credits_reset_epoch': self._credits_reset,
        }

    async def get_hybrid_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get hybrid quote: real-time price + delayed bid/ask/volume.
        
        Combines:
        - /stocks/prices/ for real-time mid price (fresh, accurate)
        - /stocks/quotes/ for bid/ask/volume (15-min delayed but still useful)
        
        This gives the LLM agent the best of both worlds for decision making.
        Fetches both endpoints concurrently for speed.
        """
        # Fetch real-time price and delayed quote concurrently
        realtime_task = self.get_realtime_price(symbol)
        delayed_task = self._get_delayed_quote(symbol)
        realtime, delayed = await asyncio.gather(realtime_task, delayed_task, return_exceptions=True)

        # Handle exceptions from gather
        if isinstance(realtime, Exception) or not realtime:
            return None

        # Enrich with delayed bid/ask/volume if available
        if not isinstance(delayed, Exception) and delayed:
            realtime['bid'] = delayed.get('bid')
            realtime['ask'] = delayed.get('ask')
            realtime['volume'] = delayed.get('volume')
            realtime['source'] = 'marketdata_hybrid'

        return realtime

    async def _get_delayed_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get delayed quote (bid/ask/volume) from /quotes endpoint."""
        try:
            client = self._get_http_client()
            if not client:
                return None
            
            async with self._get_global_semaphore():
                response = await client.get(f"/stocks/quotes/{symbol}/")
            self._parse_rate_headers(response)
            if response.status_code not in (200, 203):
                return None
            
            data = response.json()
            if data.get('s') != 'ok':
                return None
            
            def first(arr):
                return arr[0] if isinstance(arr, list) and arr else arr
            
            return {
                'bid': first(data.get('bid')),
                'ask': first(data.get('ask')),
                'volume': first(data.get('volume')),
            }
        except Exception:
            return None

    async def get_realtime_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time midpoint price for a symbol.
        
        Uses /stocks/prices/ endpoint which provides TRUE real-time data.
        Available to all users, no exchange entitlement required.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')

        Returns:
            Dict with mid, change, change_pct, updated
        """
        if not self.is_configured:
            logger.warning(f"MarketData not configured, cannot get price for {symbol}")
            return None

        try:
            response = await self._get_with_retries(
                f"/stocks/prices/{symbol}/",
                params={},
                label=f"price({symbol})",
            )
            if response is None:
                return None

            if response.status_code not in (200, 203):
                # VIX endpoints are not supported by MarketData.app in some plans.
                # Keep this quiet to avoid log spam.
                if symbol.upper() in ("VIX", "^VIX"):
                    logger.debug(f"Price request failed for {symbol}: {response.status_code}")
                else:
                    logger.warning(f"Price request failed for {symbol}: {response.status_code}")
                return None

            data = response.json()
            if data.get('s') != 'ok':
                logger.warning(f"Price API error for {symbol}: {data.get('errmsg', 'Unknown')}")
                return None

            def first(arr):
                return arr[0] if isinstance(arr, list) and arr else arr

            mid = first(data.get('mid'))
            return {
                'symbol': symbol,
                'mid': mid,
                'last': mid,  # Use mid as last for compatibility
                'bid': None,  # Not provided by /prices endpoint
                'ask': None,
                'change': first(data.get('change')),
                'change_pct': first(data.get('changepct')),
                'updated': first(data.get('updated')),
                'source': 'marketdata_realtime'
            }
        except asyncio.TimeoutError:
            logger.warning(f"Price request timed out for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Price request failed for {symbol}: {e}")
            return None

    async def get_quote(self, symbol: str, realtime: bool = True, hybrid: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get quote for a symbol.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            realtime: If True, use real-time /prices endpoint. If False, use delayed /quotes.
            hybrid: If True AND realtime=True, merge delayed bid/ask/volume with real-time price.

        Returns:
            Dict with bid, ask, last, volume, change, etc.
        """
        # Default: hybrid mode gives real-time price + delayed bid/ask/volume
        if realtime and hybrid:
            return await self.get_hybrid_quote(symbol)
        elif realtime:
            return await self.get_realtime_price(symbol)
        
        if not self.is_configured:
            logger.warning(f"MarketData not configured, cannot get quote for {symbol}")
            return None

        try:
            client = self._get_http_client()
            if not client:
                return None

            if self._track_request():
                return None
            async with self._get_global_semaphore():
                response = await client.get(f"/stocks/quotes/{symbol}/")
            self._parse_rate_headers(response)

            # Accept 200 and 203 (Non-Authoritative - cached/proxy data is still valid)
            if response.status_code not in (200, 203):
                logger.warning(f"Quote request failed for {symbol}: {response.status_code}")
                return None

            data = response.json()
            if data.get('s') != 'ok':
                logger.warning(f"Quote API error for {symbol}: {data.get('errmsg', 'Unknown')}")
                return None

            # Extract first element from arrays
            def first(arr):
                return arr[0] if isinstance(arr, list) and arr else arr

            return {
                'symbol': symbol,
                'bid': first(data.get('bid')),
                'ask': first(data.get('ask')),
                'last': first(data.get('last')),
                'mid': first(data.get('mid')),
                'volume': first(data.get('volume')),
                'change': first(data.get('change')),
                'change_pct': first(data.get('changepct')),
                'updated': first(data.get('updated')),
                'source': 'marketdata'
            }
        except asyncio.TimeoutError:
            logger.warning(f"Quote request timed out for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Quote request failed for {symbol}: {e}")
            return None

    async def get_quotes_bulk(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for multiple symbols in a single API call.

        Args:
            symbols: List of tickers

        Returns:
            Dict mapping symbol -> quote data
        """
        if not symbols:
            return {}
        if not self.is_configured:
            return {}

        try:
            client = self._get_http_client()
            if not client:
                return {}

            symbols_upper = [s.upper().strip() for s in symbols]
            if self._track_request():
                return {}
            async with self._get_global_semaphore():
                response = await client.get(
                    "/stocks/bulkquotes/",
                    params={'symbols': ','.join(symbols_upper)}
                )
            self._parse_rate_headers(response)

            if response.status_code not in (200, 203):
                logger.warning(f"Bulk quotes request failed: {response.status_code}")
                # Fallback to individual calls
                return await self._get_quotes_bulk_fallback(symbols_upper)

            data = response.json()
            if data.get('s') != 'ok':
                logger.warning(f"Bulk quotes API error: {data.get('errmsg', 'Unknown')}")
                return await self._get_quotes_bulk_fallback(symbols_upper)

            results = {}
            syms = data.get('symbol', [])
            for i, sym in enumerate(syms):
                def at(arr, idx):
                    return arr[idx] if isinstance(arr, list) and idx < len(arr) else None

                results[sym] = {
                    'symbol': sym,
                    'bid': at(data.get('bid', []), i),
                    'ask': at(data.get('ask', []), i),
                    'last': at(data.get('last', []), i),
                    'mid': at(data.get('mid', []), i),
                    'volume': at(data.get('volume', []), i),
                    'change': at(data.get('change', []), i),
                    'change_pct': at(data.get('changepct', []), i),
                    'updated': at(data.get('updated', []), i),
                    'source': 'marketdata'
                }
            return results

        except Exception as e:
            logger.warning(f"Bulk quotes request failed: {e}")
            return await self._get_quotes_bulk_fallback(symbols_upper)

    async def _get_quotes_bulk_fallback(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fallback: individual quote calls when bulk endpoint fails."""
        results = {}
        if not symbols:
            return results
        tasks = [self.get_quote(symbol) for symbol in symbols]
        quotes = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, quote in zip(symbols, quotes):
            if isinstance(quote, Exception):
                logger.warning(f"Failed to get quote for {symbol}: {quote}")
            elif quote:
                results[symbol] = quote
        return results

    async def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 30,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical OHLCV candles.

        Args:
            symbol: Stock ticker
            resolution: 'D' (daily), 'H' (hourly), '1'/'5'/'15' (minutes)
            days_back: Number of days of history (if from_date not specified)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            Dict with open, high, low, close, volume, timestamps arrays
        """
        if not self.is_configured:
            logger.warning(f"MarketData not configured, cannot get candles for {symbol}")
            return None

        try:
            # Build query params (resolution is already in the URL path)
            params = {}
            if from_date:
                params['from'] = from_date
                if to_date:
                    params['to'] = to_date
            else:
                params['countback'] = days_back

            response = await self._get_with_retries(
                f"/stocks/candles/{resolution}/{symbol}/",
                params=params,
                label=f"candles({symbol})",
            )
            if response is None:
                return None

            if response.status_code not in (200, 203):
                # 404 is expected outside market hours or for symbols with no intraday data
                level = logging.DEBUG if response.status_code == 404 else logging.WARNING
                logger.log(level, f"Candles request failed for {symbol}: {response.status_code}")
                return None

            data = response.json()
            if data.get('s') != 'ok':
                logger.warning(f"Candles API error for {symbol}: {data.get('errmsg', 'Unknown')}")
                return None

            return {
                'symbol': symbol,
                'open': data.get('o', []),
                'high': data.get('h', []),
                'low': data.get('l', []),
                'close': data.get('c', []),
                'volume': data.get('v', []),
                'timestamps': data.get('t', []),
                'source': 'marketdata'
            }
        except asyncio.TimeoutError:
            logger.warning(f"Candles request timed out for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Candles request failed for {symbol}: {e}")
            return None

    async def get_bulk_daily_candles(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get daily candles for multiple symbols in a single API call.

        Uses /stocks/bulkcandles/D/?symbols=... endpoint.
        Only supports daily resolution.

        Args:
            symbols: List of tickers

        Returns:
            Dict mapping symbol -> candle data (same format as get_candles)
        """
        if not symbols or not self.is_configured:
            return {}

        try:
            client = self._get_http_client()
            if not client:
                return {}

            symbols_upper = [s.upper().strip() for s in symbols]
            if self._track_request():
                return {}
            async with self._get_global_semaphore():
                response = await client.get(
                    "/stocks/bulkcandles/D/",
                    params={'symbols': ','.join(symbols_upper)}
                )
            self._parse_rate_headers(response)

            if response.status_code not in (200, 203):
                logger.warning(f"Bulk candles request failed: {response.status_code}")
                return {}

            data = response.json()
            if data.get('s') != 'ok':
                logger.warning(f"Bulk candles API error: {data.get('errmsg', 'Unknown')}")
                return {}

            # Parse bulk response — each row has a symbol field
            results: Dict[str, Dict] = {}
            syms = data.get('symbol', [])
            opens = data.get('o', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            closes = data.get('c', [])
            volumes = data.get('v', [])
            timestamps = data.get('t', [])

            for i, sym in enumerate(syms):
                if sym not in results:
                    results[sym] = {
                        'symbol': sym,
                        'open': [], 'high': [], 'low': [],
                        'close': [], 'volume': [], 'timestamps': [],
                        'source': 'marketdata'
                    }
                r = results[sym]
                if i < len(opens): r['open'].append(opens[i])
                if i < len(highs): r['high'].append(highs[i])
                if i < len(lows): r['low'].append(lows[i])
                if i < len(closes): r['close'].append(closes[i])
                if i < len(volumes): r['volume'].append(volumes[i])
                if i < len(timestamps): r['timestamps'].append(timestamps[i])

            return results

        except Exception as e:
            logger.warning(f"Bulk candles request failed: {e}")
            return {}

    async def calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range from candle data.

        Args:
            symbol: Stock ticker
            period: ATR period (default 14)

        Returns:
            ATR value or None
        """
        candles = await self.get_candles(symbol, 'D', days_back=period + 5)

        if not candles or len(candles.get('close', [])) < period + 1:
            logger.warning(f"Insufficient candle data for ATR: {symbol}")
            return None

        highs = candles['high']
        lows = candles['low']
        closes = candles['close']

        # Calculate True Range
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        # ATR is SMA of True Range
        atr = sum(true_ranges[-period:]) / period
        return round(atr, 2)

    # ========== OPTIONS DATA ==========

    async def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        side: Optional[str] = None,
        strike_range: Optional[tuple] = None,
        dte_range: Optional[tuple] = None,
        date: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        # Server-side filters (Phase 2)
        delta: Optional[float] = None,
        strike_limit: Optional[int] = None,
        range_filter: Optional[str] = None,
        min_bid: Optional[float] = None,
        max_bid_ask_spread_pct: Optional[float] = None,
        min_open_interest: Optional[int] = None,
        min_volume: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get options chain with Greeks and IV.

        Args:
            symbol: Underlying ticker (e.g., 'AAPL')
            expiration: Specific expiration (YYYY-MM-DD) or None for nearest
            side: 'call', 'put', or None for both
            strike_range: (min_strike, max_strike) or None for all
            dte_range: (min_dte, max_dte) to filter expirations
            date: Historical snapshot date (YYYY-MM-DD); returns chain as-of that date
            from_date: Start of historical range (YYYY-MM-DD)
            to_date: End of historical range (YYYY-MM-DD)
            delta: Server-side delta filter (e.g. 0.30)
            strike_limit: Max strikes to return per expiration
            range_filter: 'itm', 'otm', 'all' (API range param)
            min_bid: Minimum bid price filter
            max_bid_ask_spread_pct: Maximum bid-ask spread as percentage
            min_open_interest: Minimum open interest filter
            min_volume: Minimum volume filter

        Returns:
            Dict with options data including Greeks
        """
        if not self.is_configured:
            logger.warning(f"MarketData not configured, cannot get options for {symbol}")
            return None

        try:
            symbol = symbol.upper().strip()
            is_historical = bool(date or from_date or to_date)

            # Build query params
            params: Dict[str, Any] = {}

            if expiration:
                params['expiration'] = expiration
            elif is_historical:
                params['expiration'] = 'all'
            if side:
                params['side'] = side
            if strike_range:
                params['strike'] = f"{strike_range[0]}-{strike_range[1]}"
            if date:
                params['date'] = date
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date

            # Server-side filters
            if delta is not None:
                params['delta'] = str(delta)
            if strike_limit is not None:
                params['strikeLimit'] = str(strike_limit)
            if range_filter:
                params['range'] = range_filter
            if min_bid is not None:
                params['minBid'] = str(min_bid)
            if max_bid_ask_spread_pct is not None:
                params['maxBidAskSpreadPct'] = str(max_bid_ask_spread_pct)
            if min_open_interest is not None:
                params['minOpenInterest'] = str(min_open_interest)
            if min_volume is not None:
                params['minVolume'] = str(min_volume)

            if dte_range and not expiration:
                target_dte = (dte_range[0] + dte_range[1]) // 2
                params['dte'] = str(target_dte)

            response = await self._get_with_retries(
                f"/options/chain/{symbol}/",
                params=params,
                label=f"options chain {symbol}",
                throttle_options=True,
            )
            if response is None:
                return None

            # 404/400 are common when filters are too narrow (or symbol truly has no listed options).
            # Retry once dropping only the strike filter — keep side, expiration, dte.
            used_fallback = False
            if response.status_code in (400, 404) and params:
                fallback_params: Dict[str, Any] = {}
                if expiration:
                    fallback_params['expiration'] = expiration
                if side:
                    fallback_params['side'] = side
                if date:
                    fallback_params['date'] = date
                if from_date:
                    fallback_params['from'] = from_date
                if to_date:
                    fallback_params['to'] = to_date
                # Preserve DTE so API returns the right expiration window
                if 'dte' in params:
                    fallback_params['dte'] = params['dte']

                response = await self._get_with_retries(
                    f"/options/chain/{symbol}/",
                    params=fallback_params,
                    label=f"options chain fallback {symbol}",
                    throttle_options=True,
                )
                if response is None:
                    return None
                used_fallback = True

            if response.status_code == 404:
                logger.debug(f"No options chain available for {symbol}")
                return None

            if response.status_code == 400:
                logger.debug(f"No valid options chain match for {symbol} with current filters")
                return None

            if response.status_code not in (200, 203):
                logger.warning(f"Options chain request failed for {symbol}: {response.status_code}")
                return None

            data = response.json()
            if data.get('s') != 'ok':
                logger.warning(f"Options chain API error for {symbol}: {data.get('errmsg', 'Unknown')}")
                return None

            # Parse into list of option contracts
            contracts = []
            option_symbols = data.get('optionSymbol', [])
            num_contracts = len(option_symbols)

            def safe_get(arr, idx):
                return arr[idx] if arr and idx < len(arr) else None

            for i in range(num_contracts):
                contract = {
                    'option_symbol': safe_get(option_symbols, i),
                    'underlying': symbol,
                    'expiration': safe_get(data.get('expiration'), i),
                    'strike': safe_get(data.get('strike'), i),
                    'side': safe_get(data.get('side'), i),
                    'bid': safe_get(data.get('bid'), i),
                    'ask': safe_get(data.get('ask'), i),
                    'mid': safe_get(data.get('mid'), i),
                    'last': safe_get(data.get('last'), i),
                    'volume': safe_get(data.get('volume'), i) or 0,
                    'open_interest': safe_get(data.get('openInterest'), i) or 0,
                    'delta': safe_get(data.get('delta'), i),
                    'gamma': safe_get(data.get('gamma'), i),
                    'theta': safe_get(data.get('theta'), i),
                    'vega': safe_get(data.get('vega'), i),
                    'iv': safe_get(data.get('iv'), i),
                    'dte': safe_get(data.get('dte'), i),
                }

                # Filter by DTE if specified
                if dte_range and contract['dte'] is not None:
                    if contract['dte'] < dte_range[0] or contract['dte'] > dte_range[1]:
                        continue

                # Filter by strike range client-side — skip if we used the fallback
                # (fallback already dropped the strike filter to broaden results)
                if strike_range and not used_fallback and contract['strike'] is not None:
                    if contract['strike'] < strike_range[0] or contract['strike'] > strike_range[1]:
                        continue

                contracts.append(contract)

            if not contracts:
                return None

            result: Dict[str, Any] = {
                'symbol': symbol,
                'contracts': contracts,
                'source': 'marketdata',
                'is_historical': bool(date or from_date or to_date),
            }
            if date:
                result['as_of_date'] = date
            elif from_date or to_date:
                result['from_date'] = from_date
                result['to_date'] = to_date
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Options chain request timed out for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Options chain request failed for {symbol}: {e}")
            return None

    async def get_option_expirations(self, symbol: str) -> Optional[List[str]]:
        """
        Get available option expiration dates.

        Args:
            symbol: Underlying ticker

        Returns:
            List of expiration dates (YYYY-MM-DD format)
        """
        if not self.is_configured:
            return None

        try:
            client = self._get_http_client()
            if not client:
                return None

            if self._track_request():
                return None
            response = await client.get(f"/options/expirations/{symbol}/")

            if response.status_code not in (200, 203):
                return None

            data = response.json()
            if data.get('s') == 'ok':
                return data.get('expirations', [])
            return None
        except Exception as e:
            logger.warning(f"Expirations request failed for {symbol}: {e}")
            return None

    async def get_option_strikes(
        self,
        symbol: str,
        expiration: str
    ) -> Optional[List[float]]:
        """
        Get available strikes for an expiration.

        Args:
            symbol: Underlying ticker
            expiration: Expiration date (YYYY-MM-DD)

        Returns:
            List of strike prices
        """
        if not self.is_configured:
            return None

        try:
            client = self._get_http_client()
            if not client:
                return None

            if self._track_request():
                return None
            response = await client.get(f"/options/strikes/{symbol}/", params={'expiration': expiration})

            if response.status_code not in (200, 203):
                return None

            data = response.json()
            if data.get('s') == 'ok':
                return data.get('strikes', [])
            return None
        except Exception as e:
            logger.warning(f"Strikes request failed for {symbol}: {e}")
            return None

    async def get_option_lookup(self, symbol: str) -> Optional[List[str]]:
        """
        Look up OCC-format option symbols for an underlying via MDA /options/lookup/.

        Args:
            symbol: Underlying ticker (e.g. 'AAPL')

        Returns:
            List of OCC option symbols, or None on failure.
        """
        if not self.is_configured:
            return None

        try:
            client = self._get_http_client()
            if not client:
                return None

            if self._track_request():
                return None
            response = await client.get(f"/options/lookup/{symbol}/")

            if response.status_code not in (200, 203):
                return None

            data = response.json()
            if data.get('s') == 'ok':
                return data.get('optionSymbol', [])
            return None
        except Exception as e:
            logger.warning(f"Option lookup failed for {symbol}: {e}")
            return None

    async def get_option_quote(
        self,
        option_symbol: str,
        date: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get quote for a specific option contract.

        Args:
            option_symbol: OCC option symbol (e.g., 'AAPL230120C00150000')
            date: Historical snapshot date (YYYY-MM-DD)
            from_date: Start of historical range (YYYY-MM-DD)
            to_date: End of historical range (YYYY-MM-DD)

        Returns:
            Dict with bid, ask, Greeks, IV, etc.
        """
        if not self.is_configured:
            return None

        try:
            # Build historical query params if provided
            q_params: Dict[str, Any] = {}
            if date:
                q_params['date'] = date
            if from_date:
                q_params['from'] = from_date
            if to_date:
                q_params['to'] = to_date
            response = await self._get_with_retries(
                f"/options/quotes/{option_symbol}/",
                params=q_params,
                label=f"option quote {option_symbol}",
                throttle_options=True,
            )
            if response is None:
                return None

            if response.status_code not in (200, 203):
                return None

            data = response.json()
            if data.get('s') != 'ok':
                return None

            def first(arr):
                return arr[0] if isinstance(arr, list) and arr else arr

            result: Dict[str, Any] = {
                'option_symbol': option_symbol,
                'underlying': first(data.get('underlying')),
                'strike': first(data.get('strike')),
                'side': first(data.get('side')),
                'expiration': first(data.get('expiration')),
                'bid': first(data.get('bid')),
                'ask': first(data.get('ask')),
                'mid': first(data.get('mid')),
                'last': first(data.get('last')),
                'volume': first(data.get('volume')) or 0,
                'open_interest': first(data.get('openInterest')) or 0,
                'delta': first(data.get('delta')),
                'gamma': first(data.get('gamma')),
                'theta': first(data.get('theta')),
                'vega': first(data.get('vega')),
                'iv': first(data.get('iv')),
                'dte': first(data.get('dte')),
                'source': 'marketdata',
                'is_historical': bool(date or from_date or to_date),
            }
            if date:
                result['as_of_date'] = date
            return result
        except Exception as e:
            logger.warning(f"Option quote request failed for {option_symbol}: {e}")
            return None

    async def get_option_quote_series(
        self,
        option_symbol: str,
        from_date: str,
        to_date: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get a time series of option quotes between two dates.

        Returns one record per trading day in ascending date order.
        Each record has the same fields as get_option_quote() plus 'date'.

        Args:
            option_symbol: OCC option symbol
            from_date: Start date inclusive (YYYY-MM-DD)
            to_date: End date inclusive (YYYY-MM-DD)

        Returns:
            List of quote dicts ordered by date, or None on failure.
        """
        if not self.is_configured:
            return None

        try:
            response = await self._get_with_retries(
                f"/options/quotes/{option_symbol}/",
                params={'from': from_date, 'to': to_date},
                label=f"option quote series {option_symbol}",
                throttle_options=True,
            )
            if response is None:
                return None

            if response.status_code not in (200, 203):
                return None

            data = response.json()
            if data.get('s') != 'ok':
                return None

            updated_list = data.get('updated', [])
            n = len(data.get('bid', []))
            records = []
            for i in range(n):
                def g(key, idx=i):
                    arr = data.get(key, [])
                    return arr[idx] if idx < len(arr) else None

                ts_raw = g('updated')
                record_date: Optional[str] = None
                if ts_raw is not None:
                    try:
                        from datetime import datetime as _dt, timezone as _tz
                        record_date = _dt.fromtimestamp(
                            int(ts_raw), tz=_tz.utc
                        ).strftime('%Y-%m-%d')
                    except Exception:
                        pass

                records.append({
                    'option_symbol': option_symbol,
                    'date': record_date,
                    'bid': g('bid'),
                    'ask': g('ask'),
                    'mid': g('mid'),
                    'last': g('last'),
                    'volume': g('volume') or 0,
                    'open_interest': g('openInterest') or 0,
                    'delta': g('delta'),
                    'gamma': g('gamma'),
                    'theta': g('theta'),
                    'vega': g('vega'),
                    'iv': g('iv'),
                    'dte': g('dte'),
                    'source': 'marketdata',
                    'is_historical': True,
                })
            return records or None
        except Exception as e:
            logger.warning(f"Option quote series request failed for {option_symbol}: {e}")
            return None

    async def find_option_by_delta(
        self,
        symbol: str,
        target_delta: float,
        side: str,
        min_dte: int = 21,
        max_dte: int = 45
    ) -> Optional[Dict[str, Any]]:
        """
        Find option contract closest to target delta.

        Args:
            symbol: Underlying ticker
            target_delta: Target delta (e.g., 0.30 for 30-delta)
            side: 'call' or 'put'
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration

        Returns:
            Option contract dict or None
        """
        chain = await self.get_option_chain(
            symbol,
            side=side,
            dte_range=(min_dte, max_dte),
            delta=target_delta,
            strike_limit=5,
            min_bid=0.01,
        )

        if not chain or not chain.get('contracts'):
            return None

        # Filter to valid contracts with delta
        candidates = [
            c for c in chain['contracts']
            if c.get('delta') is not None
            and c.get('bid') and c.get('bid') > 0
        ]

        if not candidates:
            return None

        # For puts, delta is negative, so compare absolute values
        if side == 'put':
            target_delta = -abs(target_delta)
            best = min(candidates, key=lambda c: abs(c['delta'] - target_delta))
        else:
            best = min(candidates, key=lambda c: abs(c['delta'] - target_delta))

        return best

    async def get_iv_rank(
        self,
        symbol: str,
        dte_min: int = None,
        dte_max: int = None,
        strike_pct: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate IV Rank (current IV percentile vs 52-week range).

        Args:
            symbol: Underlying ticker
            dte_min: Minimum DTE for contracts to sample (required by caller)
            dte_max: Maximum DTE for contracts to sample (required by caller)
            strike_pct: Strike range as fraction of price (e.g. 0.15 = ±15%).
                        If None, uses a price-adaptive heuristic based on
                        option strike increments ($0.50/$1/$2.50/$5).

        Returns:
            Dict with iv_current, iv_rank, iv_high, iv_low
        """
        if dte_min is None or dte_max is None:
            logger.warning(f"get_iv_rank({symbol}): dte_min/dte_max required")
            return None

        # Get current ATM option IV as proxy for current IV
        quote = await self.get_quote(symbol)
        if not quote or not quote.get('last'):
            return None

        current_price = quote['last']

        # Determine strike window — use caller's value or price-adaptive heuristic.
        # Option strikes come in $0.50/$1/$2.50/$5 increments depending on price;
        # narrow ranges miss everything on cheap stocks.
        if strike_pct is None:
            if current_price < 5:
                strike_pct = 0.50   # ±50% for penny/micro-cap
            elif current_price < 20:
                strike_pct = 0.25   # ±25% for small-cap
            elif current_price < 100:
                strike_pct = 0.15   # ±15% for mid-range
            else:
                strike_pct = 0.10   # ±10% for large-cap

        chain = await self.get_option_chain(
            symbol,
            side='call',
            strike_range=(current_price * (1 - strike_pct), current_price * (1 + strike_pct)),
            dte_range=(dte_min, dte_max)
        )

        if not chain or not chain.get('contracts'):
            return None

        # Get average IV from ATM calls
        ivs = [c['iv'] for c in chain['contracts'] if c.get('iv')]
        if not ivs:
            return None

        current_iv = sum(ivs) / len(ivs)

        return {
            'symbol': symbol,
            'iv_current': round(current_iv * 100, 1),  # As percentage
            'iv_rank': None,  # Would need historical IV data
            'source': 'marketdata'
        }

    async def get_earnings(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        countback: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get earnings data for a symbol via MDA /stocks/earnings/{symbol}/.

        Returns dict with arrays: symbol, fiscalYear, fiscalQuarter, date,
        reportDate, reportTime, reportedEPS, estimatedEPS, surpriseEPS, surpriseEPSpct.
        """
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if countback is not None:
            params["countback"] = countback

        resp = await self._get_with_retries(
            f"/stocks/earnings/{symbol}/",
            params=params or None,
            label=f"earnings({symbol})",
        )
        if not resp or resp.status_code not in (200, 203):
            return None

        data = resp.json()
        if data.get("s") != "ok":
            return None

        # Zip parallel arrays into a list of earnings records
        n = len(data.get("fiscalYear", []))
        records = []
        for i in range(n):
            records.append({
                "fiscal_year": data["fiscalYear"][i] if i < len(data.get("fiscalYear", [])) else None,
                "fiscal_quarter": data["fiscalQuarter"][i] if i < len(data.get("fiscalQuarter", [])) else None,
                "date": data["date"][i] if i < len(data.get("date", [])) else None,
                "report_date": data["reportDate"][i] if i < len(data.get("reportDate", [])) else None,
                "report_time": data["reportTime"][i] if i < len(data.get("reportTime", [])) else None,
                "reported_eps": data["reportedEPS"][i] if i < len(data.get("reportedEPS", [])) else None,
                "estimated_eps": data["estimatedEPS"][i] if i < len(data.get("estimatedEPS", [])) else None,
                "surprise_eps": data["surpriseEPS"][i] if i < len(data.get("surpriseEPS", [])) else None,
                "surprise_eps_pct": data["surpriseEPSpct"][i] if i < len(data.get("surpriseEPSpct", [])) else None,
            })

        return {
            "symbol": symbol,
            "earnings": records,
            "count": n,
            "source": "marketdata",
        }

    async def get_news(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        countback: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get news for a symbol via MDA /stocks/news/{symbol}/.

        Returns dict with headlines, content, sources, publication dates.
        """
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if countback is not None:
            params["countback"] = countback

        resp = await self._get_with_retries(
            f"/stocks/news/{symbol}/",
            params=params or None,
            label=f"news({symbol})",
        )
        if not resp or resp.status_code not in (200, 203):
            return None

        data = resp.json()
        if data.get("s") != "ok":
            return None

        n = len(data.get("headline", []))
        articles = []
        for i in range(n):
            articles.append({
                "headline": data["headline"][i] if i < len(data.get("headline", [])) else None,
                "content": data["content"][i] if i < len(data.get("content", [])) else None,
                "source": data["source"][i] if i < len(data.get("source", [])) else None,
                "publication_date": data["publicationDate"][i] if i < len(data.get("publicationDate", [])) else None,
            })

        return {
            "symbol": symbol,
            "articles": articles,
            "count": n,
            "source": "marketdata",
        }

    async def get_market_status(
        self,
        date: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get market open/closed status via MDA /markets/status/.

        Args:
            date: Check a specific date
            from_date: Start of date range
            to_date: End of date range
        """
        params: Dict[str, Any] = {}
        if date:
            params["date"] = date
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        resp = await self._get_with_retries(
            "/markets/status/",
            params=params or None,
            label="market_status",
        )
        if not resp or resp.status_code not in (200, 203):
            return None

        data = resp.json()
        if data.get("s") != "ok":
            return None

        n = len(data.get("date", []))
        statuses = []
        for i in range(n):
            statuses.append({
                "date": data["date"][i] if i < len(data.get("date", [])) else None,
                "status": data["status"][i] if i < len(data.get("status", [])) else None,
            })

        return {
            "statuses": statuses,
            "count": n,
            "source": "marketdata",
        }


# Singleton instance
_client: Optional[MarketDataClient] = None


def get_marketdata_client() -> MarketDataClient:
    """Get or create the singleton client."""
    global _client
    if _client is None:
        _client = MarketDataClient()
    return _client


async def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get a quote."""
    client = get_marketdata_client()
    return await client.get_quote(symbol)


async def get_atr(symbol: str, period: int = 14) -> Optional[float]:
    """Convenience function to get ATR."""
    client = get_marketdata_client()
    return await client.calculate_atr(symbol, period)


async def get_option_chain(symbol: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Convenience function to get options chain."""
    client = get_marketdata_client()
    return await client.get_option_chain(symbol, **kwargs)


async def find_option_by_delta(
    symbol: str,
    target_delta: float,
    side: str,
    min_dte: int = 21,
    max_dte: int = 45
) -> Optional[Dict[str, Any]]:
    """Convenience function to find option by delta."""
    client = get_marketdata_client()
    return await client.find_option_by_delta(symbol, target_delta, side, min_dte, max_dte)
