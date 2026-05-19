"""
Research host heartbeat helpers.

Stores the research host's last scoring-round timestamp in ``research_config``
under :data:`HEARTBEAT_KEY` (``research_host_heartbeat_ts``). Deployments that
still have the legacy key ``daemon_heartbeat_ts`` are read transparently; new
writes use only the canonical key.

The trader reads :func:`is_research_host_alive` on startup and each cycle.
When fresh, the trader skips its in-process scorer. When stale, it may fall back
to single-process scoring (unless ``--require-research-host`` /
``TRADER_IN_PROCESS_SCORER=never``).
"""

from __future__ import annotations

import time
from typing import Optional

from core.log_context import CTX_RESEARCH_HEARTBEAT, bind_log_context, get_logger

logger = get_logger(__name__)

HEARTBEAT_KEY: str = "research_host_heartbeat_ts"
LEGACY_HEARTBEAT_KEY: str = "daemon_heartbeat_ts"

# Numeric status codes in research_config (float values for config_repo).
RESEARCH_HOST_STATUS_KEY: str = "research_host_status"
RESEARCH_HOST_ROUND_KEY: str = "research_host_round"
RESEARCH_HOST_USAGE_PCT_KEY: str = "research_host_usage_pct"


class ResearchHostStatus:
    """``research_config[research_host_status]`` values."""

    STARTING = 1.0
    RUNNING = 2.0
    SCORING = 3.0
    SHUTTING_DOWN = 4.0
    CAP_STOPPED = 5.0
    STOPPED = 6.0

def _heartbeat_cfg():
    from core.risk_execution_config import get_risk_execution_config

    return get_risk_execution_config()


# Hard floor when no cadence is supplied — regular-hours rounds stay fresh.
DEFAULT_STALE_AFTER_S: float = _heartbeat_cfg().heartbeat_default_stale_after_s


def write_heartbeat(now: Optional[float] = None) -> float:
    """Write the current timestamp to research_config. Returns ts written."""
    ts = float(now) if now is not None else time.time()
    try:
        from memory import set_research_config

        set_research_config(HEARTBEAT_KEY, ts, reason="research_host round", log=False)
    except Exception as e:
        logger.debug("heartbeat_write_failed", error=str(e))
        return 0.0
    bind_log_context(**{CTX_RESEARCH_HEARTBEAT: 0.0})
    logger.debug("heartbeat_written", ts=ts)
    return ts


def read_research_host_status() -> float:
    """Read ``research_host_status`` from research_config (0 if unset)."""
    try:
        from memory import get_research_config

        return float(get_research_config(RESEARCH_HOST_STATUS_KEY, 0.0))
    except Exception:
        return 0.0


def publish_research_host_heartbeat(
    *,
    now: Optional[float] = None,
    status: float = ResearchHostStatus.RUNNING,
    round_num: int = 0,
    usage_pct: float = 0.0,
) -> float:
    """Write heartbeat timestamp plus trader-visible status metadata.

    Args:
        now: Optional unix timestamp (default: now).
        status: :class:`ResearchHostStatus` code.
        round_num: Last completed or in-progress scoring round.
        usage_pct: Researcher daily cap usage percent (0–100+).

    Returns:
        Heartbeat timestamp written (0.0 on failure).
    """
    ts = write_heartbeat(now=now)
    if ts <= 0.0:
        return 0.0
    profile_label = "balanced"
    try:
        from core.central_profit_config import get_research_settings
        from core.research_settings import RESEARCH_HOST_PROFILE_KEY
        from memory import set_research_config

        settings = get_research_settings()
        profile_label = settings.profile_label
        set_research_config(
            RESEARCH_HOST_STATUS_KEY, float(status), reason="heartbeat", log=False
        )
        set_research_config(
            RESEARCH_HOST_ROUND_KEY, float(max(0, round_num)), reason="heartbeat", log=False
        )
        set_research_config(
            RESEARCH_HOST_USAGE_PCT_KEY, float(usage_pct), reason="heartbeat", log=False
        )
        set_research_config(
            RESEARCH_HOST_PROFILE_KEY,
            profile_label,
            reason="heartbeat",
            log=False,
        )
    except Exception as e:
        logger.debug("research_host_status_publish_failed", error=str(e))
    logger.info(
        "research_host_heartbeat_published",
        status=status,
        round=round_num,
        usage_pct=round(usage_pct, 1),
        profit_profile=profile_label,
    )
    return ts


def is_research_host_operational(
    stale_after_s: Optional[float] = None,
    *,
    now: Optional[float] = None,
) -> bool:
    """True when heartbeat is fresh and host is not shutting down or cap-stopped."""
    if not is_research_host_alive(stale_after_s=stale_after_s, now=now):
        return False
    st = read_research_host_status()
    if st in (
        ResearchHostStatus.SHUTTING_DOWN,
        ResearchHostStatus.CAP_STOPPED,
        ResearchHostStatus.STOPPED,
        ResearchHostStatus.STARTING,
    ):
        return False
    return True


def read_heartbeat() -> float:
    """Return the last heartbeat timestamp (0.0 if none / on error).

    Prefers :data:`HEARTBEAT_KEY`; falls back to :data:`LEGACY_HEARTBEAT_KEY`
    so existing databases keep working without a manual migration.
    """
    try:
        from memory import get_research_config

        ts = float(get_research_config(HEARTBEAT_KEY, 0.0))
        if ts > 0.0:
            return ts
        return float(get_research_config(LEGACY_HEARTBEAT_KEY, 0.0))
    except Exception as e:
        logger.debug("heartbeat read failed: %s", e)
        return 0.0


def is_research_host_alive(
    stale_after_s: Optional[float] = None,
    *,
    now: Optional[float] = None,
) -> bool:
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
        cfg = _heartbeat_cfg()
        stale_after_s = cfg.heartbeat_stale_after_s(cadence_s)
    cur = float(now) if now is not None else time.time()
    return (cur - last) <= float(stale_after_s)


def heartbeat_age_s(now: Optional[float] = None) -> float:
    """Seconds since the last heartbeat.  ``float('inf')`` if never written."""
    last = read_heartbeat()
    if last <= 0.0:
        return float("inf")
    cur = float(now) if now is not None else time.time()
    return max(0.0, cur - last)
