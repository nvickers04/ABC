"""
Market Hours Utility
Provides comprehensive market session detection and trading hours information.
Integrates with IBKR API to get actual contract trading hours including premarket/postmarket.
"""

import logging
from datetime import datetime, time, timezone, timedelta
from typing import Dict, Any, Optional
from enum import Enum

import exchange_calendars as ecals

logger = logging.getLogger(__name__)


def _load_market_hours_config() -> dict:
    """Return empty dict — market hours use exchange_calendars defaults."""
    return {}


class MarketSession(Enum):
    """Market trading session types"""
    CLOSED = "closed"
    PREMARKET = "premarket"
    REGULAR = "regular"
    POSTMARKET = "postmarket"


class MarketHoursProvider:
    """
    Provides market hours information using exchange calendars and IBKR contract details.
    Supports premarket (4:00 AM - 9:30 AM ET) and postmarket (4:00 PM - 8:00 PM ET) detection.
    """

    def __init__(self, exchange: str = 'NYSE'):
        """
        Initialize market hours provider.
        
        Args:
            exchange: Exchange calendar to use (NYSE, NASDAQ, etc.)
        """
        # Load config
        mh_cfg = _load_market_hours_config()
        self.exchange = mh_cfg.get('exchange', exchange)
        try:
            self.calendar = ecals.get_calendar(self.exchange)
            logger.info(f"Initialized market hours provider for {self.exchange}")
        except Exception as e:
            logger.error(f"Failed to load exchange calendar for {self.exchange}: {e}")
            self.calendar = None

        # Parse times from config (HH:MM strings) or use defaults
        def _parse_time(key: str, default: time) -> time:
            val = mh_cfg.get(key)
            if val:
                try:
                    parts = str(val).split(':')
                    return time(int(parts[0]), int(parts[1]))
                except (ValueError, IndexError):
                    logger.warning(f"Invalid time format for {key}: {val}, using default")
            return default

        self.premarket_start = _parse_time('premarket_start', time(4, 0))
        self.regular_open = _parse_time('regular_open', time(9, 30))
        self.regular_close = _parse_time('regular_close', time(16, 0))
        self.postmarket_end = _parse_time('postmarket_end', time(20, 0))

        # Try to use zoneinfo for DST-aware ET, fall back to fixed offset
        try:
            from zoneinfo import ZoneInfo
            self._et_tz = ZoneInfo('America/New_York')
        except ImportError:
            self._et_tz = None
            logger.warning("zoneinfo not available, using fixed UTC-5 offset (may be wrong during DST)")

    def _to_eastern(self, dt: datetime) -> datetime:
        """Convert datetime to Eastern Time (DST-aware)."""
        if self._et_tz is not None:
            return dt.astimezone(self._et_tz)
        else:
            # Fallback: assume EST (UTC-5) - may be wrong during DST
            return dt.astimezone(timezone(timedelta(hours=-5)))

    def get_current_session(self, dt: Optional[datetime] = None) -> MarketSession:
        """
        Determine current market session.
        
        Args:
            dt: Datetime to check (defaults to now in UTC)
            
        Returns:
            MarketSession enum indicating current session
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Convert to ET for session detection (DST-aware)
        et_time = self._to_eastern(dt)
        current_time = et_time.time()

        # Check if it's a trading day
        if not self.is_trading_day(et_time):
            return MarketSession.CLOSED

        # Determine session based on time
        if current_time >= self.premarket_start and current_time < self.regular_open:
            return MarketSession.PREMARKET
        elif current_time >= self.regular_open and current_time < self.regular_close:
            return MarketSession.REGULAR
        elif current_time >= self.regular_close and current_time < self.postmarket_end:
            return MarketSession.POSTMARKET
        else:
            return MarketSession.CLOSED

    def is_trading_day(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if given date is a trading day.
        
        Args:
            dt: Datetime to check (defaults to now)
            
        Returns:
            True if trading day, False otherwise
        """
        if self.calendar is None:
            # Fallback: assume Mon-Fri are trading days
            if dt is None:
                dt = datetime.now(timezone.utc)
            return dt.weekday() < 5

        try:
            if dt is None:
                dt = datetime.now(timezone.utc)
            return self.calendar.is_session(dt.date())
        except Exception as e:
            logger.warning(f"Error checking trading day: {e}")
            return dt.weekday() < 5  # Fallback

    def is_market_open(self, include_extended: bool = False, dt: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.
        
        Args:
            include_extended: If True, includes premarket and postmarket hours
            dt: Datetime to check (defaults to now)
            
        Returns:
            True if market is open
        """
        session = self.get_current_session(dt)

        if include_extended:
            return session in [MarketSession.PREMARKET, MarketSession.REGULAR, MarketSession.POSTMARKET]
        else:
            return session == MarketSession.REGULAR

    def get_session_info(self, dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get comprehensive session information.
        
        Args:
            dt: Datetime to check (defaults to now)
            
        Returns:
            Dict with session details
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        session = self.get_current_session(dt)
        et_time = self._to_eastern(dt)

        info = {
            'session': session.value,
            'is_trading_day': self.is_trading_day(dt),
            'current_time_et': et_time.strftime('%H:%M:%S'),
            'current_date': et_time.date().isoformat(),
            'exchange': self.exchange,
        }

        # Compute minutes to open / close
        now_minutes = et_time.hour * 60 + et_time.minute
        open_minutes = self.regular_open.hour * 60 + self.regular_open.minute
        close_minutes = self.regular_close.hour * 60 + self.regular_close.minute
        if session == MarketSession.PREMARKET:
            info['minutes_to_open'] = open_minutes - now_minutes
        elif session == MarketSession.REGULAR:
            info['minutes_to_close'] = close_minutes - now_minutes

        # Add session-specific details
        if session == MarketSession.PREMARKET:
            next_open = datetime.combine(et_time.date(), self.regular_open)
            info['next_transition'] = 'regular_open'
            info['next_transition_time'] = next_open.isoformat()
        elif session == MarketSession.REGULAR:
            next_close = datetime.combine(et_time.date(), self.regular_close)
            info['next_transition'] = 'regular_close'
            info['next_transition_time'] = next_close.isoformat()
        elif session == MarketSession.POSTMARKET:
            next_end = datetime.combine(et_time.date(), self.postmarket_end)
            info['next_transition'] = 'market_closed'
            info['next_transition_time'] = next_end.isoformat()
        else:
            # Market closed - find next open
            next_open = self._get_next_market_open(et_time)
            info['next_transition'] = 'premarket_open'
            info['next_transition_time'] = next_open.isoformat() if next_open else None

        return info

    def _get_next_market_open(self, current_et: datetime) -> Optional[datetime]:
        """Get the next market opening time."""
        if self.calendar is None:
            return None

        try:
            # Get next trading day
            next_session = self.calendar.next_session(current_et.date())
            # Combine with premarket start time
            next_open = datetime.combine(next_session, self.premarket_start)
            # Return with ET timezone
            if self._et_tz is not None:
                return next_open.replace(tzinfo=self._et_tz)
            else:
                return next_open.replace(tzinfo=timezone(timedelta(hours=-5)))
        except Exception as e:
            logger.warning(f"Error getting next market open: {e}")
            return None

    def get_trading_hours_info(self) -> Dict[str, Any]:
        """
        Get static trading hours information.
        
        Returns:
            Dict with trading hours configuration
        """
        return {
            'exchange': self.exchange,
            'timezone': 'America/New_York',
            'sessions': {
                'premarket': {
                    'start': self.premarket_start.strftime('%H:%M'),
                    'end': self.regular_open.strftime('%H:%M'),
                    'description': 'Extended hours - premarket trading'
                },
                'regular': {
                    'start': self.regular_open.strftime('%H:%M'),
                    'end': self.regular_close.strftime('%H:%M'),
                    'description': 'Regular trading hours'
                },
                'postmarket': {
                    'start': self.regular_close.strftime('%H:%M'),
                    'end': self.postmarket_end.strftime('%H:%M'),
                    'description': 'Extended hours - after-hours trading'
                }
            }
        }


# Global instance
_market_hours_provider: Optional[MarketHoursProvider] = None


def get_market_hours_provider(exchange: str = 'NYSE') -> MarketHoursProvider:
    """Get or create global market hours provider instance."""
    global _market_hours_provider
    if _market_hours_provider is None or _market_hours_provider.exchange != exchange:
        _market_hours_provider = MarketHoursProvider(exchange)
    return _market_hours_provider


def get_session_info() -> Dict[str, Any]:
    """Get comprehensive session information."""
    provider = get_market_hours_provider()
    return provider.get_session_info()
