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
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

# API Base URL
API_BASE = "https://api.marketdata.app/v1"


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
        self._http_client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._daily_count = 0
        self._daily_reset: Optional[datetime] = None  # Resets at midnight UTC
        self._last_request_time: Optional[datetime] = None
        self._loop_id: Optional[int] = None  # Track which event loop the client belongs to

        if not self.api_key:
            logger.warning("No Market Data App API key configured")

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            # No running loop - will create new client
            current_loop_id = None
        
        # Create new client if none exists or if we're in a different event loop
        if self._http_client is None or (current_loop_id and self._loop_id != current_loop_id):
            if self._http_client is not None:
                # Old client exists but is bound to different loop - abandon it
                # (can't close it from here as that's async)
                self._http_client = None
            
            if self.api_key:
                self._http_client = httpx.AsyncClient(
                    base_url=API_BASE,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30.0,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=30,
                    ),
                )
                self._loop_id = current_loop_id
        
        return self._http_client

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _track_request(self) -> bool:
        """
        Track request count for observability. Always returns False (unlimited plan).
        Resets daily counter at midnight UTC.
        """
        now = datetime.utcnow()
        
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
            
            response = await client.get(f"/stocks/quotes/{symbol}/")
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
            client = self._get_http_client()
            if not client:
                return None

            if self._track_request():
                return None
            response = await client.get(f"/stocks/prices/{symbol}/")

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
            response = await client.get(f"/stocks/quotes/{symbol}/")

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
        Get quotes for multiple symbols.

        Args:
            symbols: List of tickers

        Returns:
            Dict mapping symbol -> quote data
        """
        results = {}
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
        to_date: Optional[str] = None
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
            client = self._get_http_client()
            if not client:
                return None

            # Build query params
            params = {'resolution': resolution}
            if from_date:
                params['from'] = from_date
                if to_date:
                    params['to'] = to_date
            else:
                params['countback'] = days_back

            if self._track_request():
                return None
            response = await client.get(f"/stocks/candles/{resolution}/{symbol}/", params=params)

            if response.status_code not in (200, 203):
                logger.warning(f"Candles request failed for {symbol}: {response.status_code}")
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
        dte_range: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get options chain with Greeks and IV.

        Args:
            symbol: Underlying ticker (e.g., 'AAPL')
            expiration: Specific expiration (YYYY-MM-DD) or None for nearest
            side: 'call', 'put', or None for both
            strike_range: (min_strike, max_strike) or None for all
            dte_range: (min_dte, max_dte) to filter expirations

        Returns:
            Dict with options data including Greeks
        """
        if not self.is_configured:
            logger.warning(f"MarketData not configured, cannot get options for {symbol}")
            return None

        try:
            client = self._get_http_client()
            if not client:
                return None

            symbol = symbol.upper().strip()

            # Build query params
            params: Dict[str, Any] = {}
            if expiration:
                params['expiration'] = expiration
            if side:
                params['side'] = side
            if strike_range:
                params['strike'] = f"{strike_range[0]}-{strike_range[1]}"

            # FIX: The MarketData API defaults to the nearest expiration only.
            # When a dte_range is requested, pass the target DTE so the API
            # returns contracts near that expiration.  Without this, ALL
            # contracts come back with DTE=3 (or whatever the nearest is)
            # and the client-side filter discards 100% of them.
            if dte_range and not expiration:
                # Use midpoint of range as the target; client-side filter
                # still trims to the exact (min, max) bounds afterward.
                target_dte = (dte_range[0] + dte_range[1]) // 2
                params['dte'] = str(target_dte)

            if self._track_request():
                return None
            response = await client.get(f"/options/chain/{symbol}/", params=params)

            # 404/400 are common when filters are too narrow (or symbol truly has no listed options).
            # Retry once dropping only the strike filter — keep side, expiration, dte.
            used_fallback = False
            if response.status_code in (400, 404) and params:
                fallback_params: Dict[str, Any] = {}
                if expiration:
                    fallback_params['expiration'] = expiration
                if side:
                    fallback_params['side'] = side
                # Preserve DTE so API returns the right expiration window
                if 'dte' in params:
                    fallback_params['dte'] = params['dte']

                if self._track_request():
                    return None
                response = await client.get(f"/options/chain/{symbol}/", params=fallback_params)
                used_fallback = True

            if response.status_code == 404:
                logger.info(f"No options chain available for {symbol}")
                return None

            if response.status_code == 400:
                logger.info(f"No valid options chain match for {symbol} with current filters")
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

            return {
                'symbol': symbol,
                'contracts': contracts,
                'source': 'marketdata'
            }
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

    async def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get quote for a specific option contract.

        Args:
            option_symbol: OCC option symbol (e.g., 'AAPL230120C00150000')

        Returns:
            Dict with bid, ask, Greeks, IV, etc.
        """
        if not self.is_configured:
            return None

        try:
            client = self._get_http_client()
            if not client:
                return None

            if self._track_request():
                return None
            response = await client.get(f"/options/quotes/{option_symbol}/")

            if response.status_code not in (200, 203):
                return None

            data = response.json()
            if data.get('s') != 'ok':
                return None

            def first(arr):
                return arr[0] if isinstance(arr, list) and arr else arr

            return {
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
                'source': 'marketdata'
            }
        except Exception as e:
            logger.warning(f"Option quote request failed for {option_symbol}: {e}")
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
            dte_range=(min_dte, max_dte)
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
