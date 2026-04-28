"""
Research daemon heartbeat helpers.

Stores the daemon's last-round timestamp in the existing
``research_config`` key/value table under
``HEARTBEAT_KEY = "daemon_heartbeat_ts"``.

The trading agent reads ``is_daemon_alive(stale_after_s)`` on startup
and once per cycle.  When fresh, the agent skips spawning its own
in-process scorer — the daemon owns that work.  When stale, the agent
falls back to single-process mode.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


HEARTBEAT_KEY: str = "daemon_heartbeat_ts"
DEFAULT_STALE_AFTER_S: float = 60.0


def write_heartbeat(now: Optional[float] = None) -> float:
    """Write the current timestamp to the research_config key.  Returns
    the timestamp written.  Fail-soft: logs DEBUG and returns 0.0 on
    error so a transient DB hiccup never crashes the daemon."""
    ts = float(now) if now is not None else time.time()
    try:
        from memory import set_research_config
        set_research_config(HEARTBEAT_KEY, ts, reason="research_daemon round")
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


def is_daemon_alive(stale_after_s: float = DEFAULT_STALE_AFTER_S,
                    *, now: Optional[float] = None) -> bool:
    """True iff a heartbeat exists and is younger than ``stale_after_s``."""
    last = read_heartbeat()
    if last <= 0.0:
        return False
    cur = float(now) if now is not None else time.time()
    return (cur - last) <= float(stale_after_s)


def heartbeat_age_s(now: Optional[float] = None) -> float:
    """Seconds since the last heartbeat.  ``float('inf')`` if never written."""
    last = read_heartbeat()
    if last <= 0.0:
        return float("inf")
    cur = float(now) if now is not None else time.time()
    return max(0.0, cur - last)
