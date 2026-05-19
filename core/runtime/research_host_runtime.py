"""Research-host process runtime — decoupled from the trader.

The dedicated research machine (``python -m research``) sets
:data:`ENV_RESEARCH_HOST` and uses this module for:

* Daily token / activity cap tracking with graceful shutdown
* Coordinated stop of scorer + template evolution
* Optional in-process wake signals (trader-only; skipped on research host)

The trader process must **not** set :data:`ENV_RESEARCH_HOST`; it uses
separate Grok LLM caps and may run an in-process scorer when heartbeat is stale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.log_context import get_logger

logger = get_logger(__name__)

ENV_RESEARCH_HOST = "ABC_RESEARCH_HOST"

_shutdown_reason: Optional[str] = None


def mark_research_host_process() -> None:
    """Mark the current OS process as the research host (not the trader)."""
    os.environ[ENV_RESEARCH_HOST] = "1"


def is_research_host_process() -> bool:
    """True when running under ``research/host.py`` (split-host research machine)."""
    return os.environ.get(ENV_RESEARCH_HOST, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def shutdown_requested() -> bool:
    """True after :func:`request_shutdown` (token cap, SIGTERM, etc.)."""
    return _shutdown_reason is not None


def shutdown_reason() -> Optional[str]:
    return _shutdown_reason


def request_shutdown(reason: str) -> None:
    """Graceful shutdown: status in Postgres, stop scorer + evolution loops."""
    global _shutdown_reason
    if _shutdown_reason is not None:
        return
    _shutdown_reason = reason
    logger.warning("research_host_shutdown_requested", reason=reason)

    try:
        from core.runtime.heartbeat import (
            ResearchHostStatus,
            publish_research_host_heartbeat,
        )

        publish_research_host_heartbeat(status=ResearchHostStatus.SHUTTING_DOWN)
    except Exception as e:
        logger.debug("shutdown_heartbeat_publish_failed", error=str(e))

    try:
        from signals.scorer import stop_scorer

        stop_scorer()
    except Exception as e:
        logger.debug("stop_scorer_failed", error=str(e))

    try:
        from signals.template_evolution import stop_evolution

        stop_evolution()
    except Exception as e:
        logger.debug("stop_evolution_failed", error=str(e))


def finalize_shutdown() -> None:
    """Call from research host ``finally`` after loops exit."""
    try:
        from core.runtime.heartbeat import (
            ResearchHostStatus,
            publish_research_host_heartbeat,
        )

        publish_research_host_heartbeat(status=ResearchHostStatus.STOPPED)
    except Exception as e:
        logger.debug("finalize_shutdown_heartbeat_failed", error=str(e))
    logger.info(
        "research_host_stopped",
        reason=_shutdown_reason or "clean_exit",
    )


def notify_scorer_round_complete(round_num: int) -> None:
    """Wake the trader loop when scorer runs in-process (dev single-box only)."""
    if is_research_host_process():
        return
    try:
        from core.wake_events import wake_bus

        wake_bus.signal(f"scorer_round_{round_num}")
    except Exception as e:
        logger.debug("wake_bus_signal_skipped", error=str(e))


def _usage_key_for_today() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"researcher_daily_usage_{today}"


@dataclass(frozen=True)
class TokenCapVerdict:
    """Result of incrementing researcher daily usage after a scoring round."""

    usage: float
    cap: int
    pct: float
    exceeded: bool
    warn: bool

    @property
    def should_stop(self) -> bool:
        return self.exceeded


def record_round_usage(*, round_delta: float = 75.0) -> TokenCapVerdict:
    """Increment researcher daily usage and return cap verdict.

    Args:
        round_delta: Conservative units charged per scoring round (MDA tracked separately).
    """
    from core.config import RESEARCHER_DAILY_TOKEN_CAP
    from memory import get_research_config, set_research_config

    cap = int(RESEARCHER_DAILY_TOKEN_CAP)
    key = _usage_key_for_today()
    current = float(get_research_config(key, 0.0))
    new_val = current + float(round_delta)
    set_research_config(key, new_val, reason="researcher daily cap", log=False)

    pct = (new_val / cap * 100.0) if cap > 0 else 0.0
    exceeded = cap > 0 and new_val >= cap
    warn = cap > 0 and new_val > cap * 0.85 and not exceeded

    if exceeded:
        logger.critical(
            "researcher_token_cap_exceeded",
            usage=new_val,
            cap=cap,
            pct=round(pct, 1),
            action="graceful_shutdown",
        )
    elif warn:
        logger.warning(
            "researcher_token_cap_warning",
            usage=new_val,
            cap=cap,
            pct=round(pct, 1),
        )

    return TokenCapVerdict(
        usage=new_val,
        cap=cap,
        pct=pct,
        exceeded=exceeded,
        warn=warn,
    )


def check_startup_token_cap() -> TokenCapVerdict:
    """Read-only cap check for research host boot (no increment)."""
    from core.config import RESEARCHER_DAILY_TOKEN_CAP
    from memory import get_research_config

    cap = int(RESEARCHER_DAILY_TOKEN_CAP)
    usage = float(get_research_config(_usage_key_for_today(), 0.0))
    pct = (usage / cap * 100.0) if cap > 0 else 0.0
    exceeded = cap > 0 and usage >= cap
    warn = cap > 0 and usage > cap * 0.85 and not exceeded
    return TokenCapVerdict(
        usage=usage,
        cap=cap,
        pct=pct,
        exceeded=exceeded,
        warn=warn,
    )
