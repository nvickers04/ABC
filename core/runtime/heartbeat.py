"""
Research host heartbeat helpers.

Stores the research host's last scoring-round timestamp in ``research_config``
under ``HEARTBEAT_KEY`` (``daemon_heartbeat_ts`` — key name is historical).

The trader reads ``is_research_host_alive(stale_after_s)`` on startup and each cycle.
When fresh, the trader skips its in-process scorer. When stale, it may fall back
to single-process scoring (unless ``--require-daemon`` / ``TRADER_IN_PROCESS_SCORER=never``).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


HEARTBEAT_KEY: str = "daemon_heartbeat_ts"

# Hard floor when no cadence is supplied — regular-hours rounds stay fresh;
# a dead research host is detected within ~3 minutes.
DEFAULT_STALE_AFTER_S: float = 180.0


def write_heartbeat(now: Optional[float] = None) -> float:
    """Write the current timestamp to research_config. Returns ts written."""
    ts = float(now) if now is not None else time.time()
    try:
        from memory import set_research_config

        set_research_config(HEARTBEAT_KEY, ts, reason="research_host round")
    except Exception as e:
        logger.debug("heartbeat write failed: %s", e)
        return 0.0
    return ts


def read_heartbeat() -> float:
    """Return the last heartbeat timestamp (0.0 if none / on error)."""
    try:
        from memory import get_research_config
        return float(get_research_config(HEARTBEAT_KEY, 0.0))
    except Exception as e:
        logger.debug("heartbeat read failed: %s", e)
        return 0.0


def is_research_host_alive(stale_after_s: Optional[float] = None,
                         *, now: Optional[float] = None) -> bool:
    """True iff the research host heartbeat exists and is fresh for the cadence tier.

    When ``stale_after_s`` is None (the common case) the threshold is
    derived from the current cadence: ``3 × cadence_seconds() + 60``,
    floored at ``DEFAULT_STALE_AFTER_S``.  Overnight long cadence avoids false
    negatives; a crashed research host during regular hours is detected quickly.

    Pass an explicit value to opt out of the cadence-aware default.
    """
    last = read_heartbeat()
    if last <= 0.0:
        return False
    if stale_after_s is None:
        try:
            from core.runtime.cadence import cadence_seconds
            cadence_s = float(cadence_seconds())
        except Exception:
            cadence_s = 30.0
        stale_after_s = max(DEFAULT_STALE_AFTER_S, 3.0 * cadence_s + 60.0)
    cur = float(now) if now is not None else time.time()
    return (cur - last) <= float(stale_after_s)


def heartbeat_age_s(now: Optional[float] = None) -> float:
    """Seconds since the last heartbeat.  ``float('inf')`` if never written."""
    last = read_heartbeat()
    if last <= 0.0:
        return float("inf")
    cur = float(now) if now is not None else time.time()
    return max(0.0, cur - last)


# Back-compat alias — prefer is_research_host_alive in new code.
is_daemon_alive = is_research_host_alive
