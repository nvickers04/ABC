"""Market hours provider pinned to simulation clock."""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Optional

from data.market_hours import MarketHoursProvider, MarketSession


class SimulatedMarketHoursProvider(MarketHoursProvider):
    """Delegates calendar logic but uses an injected ``now`` for session detection."""

    def __init__(self, exchange: str = "NYSE") -> None:
        super().__init__(exchange=exchange)
        self._fixed_now: Optional[datetime] = None

    def set_now(self, now_utc: datetime) -> None:
        self._fixed_now = now_utc

    def get_current_session(self, dt: Optional[datetime] = None) -> MarketSession:
        return super().get_current_session(self._fixed_now if dt is None else dt)

    def is_trading_day(self, dt: Optional[datetime] = None) -> bool:
        return super().is_trading_day(self._fixed_now if dt is None else dt)

    def get_session_info(self, dt: Optional[datetime] = None) -> dict[str, Any]:
        now = self._fixed_now if dt is None else dt
        info = super().get_session_info(now)
        # Gap guard expects minutes_to_close during regular session
        et = self._to_eastern(now)
        close = self._effective_close(et)
        if info.get("session") == "regular":
            close_dt = et.replace(hour=close.hour, minute=close.minute, second=0, microsecond=0)
            mins_to_close = max(0, int((close_dt - et).total_seconds() // 60))
            info["minutes_to_close"] = mins_to_close
        return info
