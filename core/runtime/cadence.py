"""
Session-aware cadence for the research daemon.

Returns the seconds-between-rounds the daemon should sleep based on
the current ET wall-clock.  Pure-function helpers — no IBKR, no DB —
so it's trivially testable and can run in any process.

Tiers (Mon–Fri only; weekends always overnight):

    * Regular hours      09:30 – 16:00 ET   →  30s
    * Extended hours     04:00 – 09:30 ET   →  300s   (5 min)
                         16:00 – 20:00 ET   →  300s
    * Overnight          20:00 – 04:00 ET   →  1800s  (30 min)
    * Weekend            anytime Sat/Sun    →  1800s

These thresholds mirror docs/PLAN_COGNITIVE_ARCHITECTURE.md §1.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo


_ET = ZoneInfo("America/New_York")

# Boundaries (ET local time-of-day).
_PREMARKET_OPEN = time(4, 0)
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)
_POSTMARKET_CLOSE = time(20, 0)

# Cadence values (seconds).
CADENCE_REGULAR_S: int = 30
CADENCE_EXTENDED_S: int = 5 * 60
CADENCE_OVERNIGHT_S: int = 30 * 60


def _now_et(dt: Optional[datetime] = None) -> datetime:
    if dt is None:
        return datetime.now(_ET)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def session_label(dt: Optional[datetime] = None) -> str:
    """Return 'regular' | 'extended' | 'overnight'.

    Weekends are always 'overnight' (no extended-hours trading).
    """
    now = _now_et(dt)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return "overnight"
    t = now.time()
    if _REGULAR_OPEN <= t < _REGULAR_CLOSE:
        return "regular"
    if _PREMARKET_OPEN <= t < _REGULAR_OPEN:
        return "extended"
    if _REGULAR_CLOSE <= t < _POSTMARKET_CLOSE:
        return "extended"
    return "overnight"


def cadence_seconds(dt: Optional[datetime] = None) -> int:
    """Seconds the research daemon should sleep before the next round."""
    label = session_label(dt)
    if label == "regular":
        return CADENCE_REGULAR_S
    if label == "extended":
        return CADENCE_EXTENDED_S
    return CADENCE_OVERNIGHT_S
