"""Shared helpers for ProfitConfig / simulation / logger / optimizer tests."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from core.central_profit_config import ProfitConfig, get_profit_config
from core.profit_profiles import PROFIT_PROFILE_ENV, VALID_PROFILES


def load_real_profit_config(profile: str | None = None) -> ProfitConfig:
    """Load a real composed :class:`ProfitConfig` (clears cache)."""
    if profile:
        os.environ[PROFIT_PROFILE_ENV] = profile
    else:
        os.environ.pop(PROFIT_PROFILE_ENV, None)
    return get_profit_config().reload(dotenv=False)


def make_archive_payload(
    symbol: str,
    session_dates: list[str],
    *,
    close: float = 100.0,
) -> dict[str, Any]:
    """Minimal daily bar archive for replay simulation."""
    bars = []
    for i, d in enumerate(session_dates):
        bars.append(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close + i * 0.1,
                "volume": 1_000_000,
                "timestamp": int(datetime.strptime(d, "%Y-%m-%d").timestamp()),
                "date": d,
            }
        )
    return {"symbol": symbol.upper(), "source": "test", "bars": bars}


def nyse_session_dates(start: str, end: str) -> list[str]:
    """Trading sessions between inclusive dates (XNYS calendar)."""
    import exchange_calendars as ecals

    cal = ecals.get_calendar("XNYS")
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out: list[str] = []
    d = s
    while d <= e:
        if cal.is_session(d):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def mock_archive_for_range(
    start: str,
    end: str,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT"),
) -> dict[str, dict[str, Any]]:
    """Map symbol -> archive payload covering ``start``..``end`` sessions."""
    dates = nyse_session_dates(start, end)
    return {sym: make_archive_payload(sym, dates) for sym in symbols}
