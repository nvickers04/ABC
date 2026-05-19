"""Structured logging context (structlog + stdlib).

Binds consistent fields on every log line in trader / research / execution paths:

* ``cycle_id`` — agent ReAct cycle
* ``trade_id`` — order or execution identifier (often same as ``order_id``)
* ``quality_score`` — QualityMatrix ``overall_quality`` tier
* ``research_heartbeat`` — seconds since last research-host heartbeat (``None`` if never)

Call :func:`configure_structlog` from :func:`core.log_setup.configure_root_logging`.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

# Canonical context keys (also used in log processors).
CTX_CYCLE_ID = "cycle_id"
CTX_TRADE_ID = "trade_id"
CTX_QUALITY_SCORE = "quality_score"
CTX_RESEARCH_HEARTBEAT = "research_heartbeat"
CTX_ORDER_ID = "order_id"

_CONFIGURED = False


def _research_heartbeat_age_s() -> float | None:
    try:
        from core.runtime.heartbeat import heartbeat_age_s

        age = heartbeat_age_s()
        if age == float("inf"):
            return None
        return round(age, 1)
    except Exception:
        return None


def _current_quality_score() -> str | None:
    try:
        from core.runtime.operating_context import get_operating_context

        qm = get_operating_context().quality_matrix
        return str(getattr(qm, "overall_quality", None) or "") or None
    except Exception:
        return None


def configure_structlog(*, verbose: bool = False) -> None:
    """Wire structlog to the root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = logging.DEBUG if verbose else logging.INFO

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        handler.setFormatter(formatter)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to stdlib (use instead of ``logging.getLogger``)."""
    return structlog.get_logger(name)


def clear_log_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def bind_log_context(**kwargs: Any) -> None:
    """Merge fields into the current structlog context."""
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if clean:
        structlog.contextvars.bind_contextvars(**clean)


def bind_trader_cycle_context(*, cycle_id: int) -> None:
    """Bind trader cycle fields (call at the start of each agent cycle)."""
    bind_log_context(
        **{
            CTX_CYCLE_ID: cycle_id,
            CTX_QUALITY_SCORE: _current_quality_score(),
            CTX_RESEARCH_HEARTBEAT: _research_heartbeat_age_s(),
        }
    )


def refresh_trader_cycle_context(*, cycle_id: int | None = None) -> None:
    """Refresh quality + heartbeat without clearing trade context."""
    fields: dict[str, Any] = {
        CTX_QUALITY_SCORE: _current_quality_score(),
        CTX_RESEARCH_HEARTBEAT: _research_heartbeat_age_s(),
    }
    if cycle_id is not None:
        fields[CTX_CYCLE_ID] = cycle_id
    bind_log_context(**fields)


def bind_trade_context(
    *,
    trade_id: str | int | None = None,
    order_id: str | int | None = None,
) -> None:
    """Bind execution identifiers for tools / IBKR paths."""
    tid = trade_id if trade_id is not None else order_id
    if tid is None:
        return
    bind_log_context(
        **{
            CTX_TRADE_ID: str(tid),
            CTX_ORDER_ID: int(order_id) if order_id is not None else None,
        }
    )


def bind_research_host_context(*, cycle_id: int | None = None) -> None:
    """Bind context for ``python -m research`` (writes heartbeat each round)."""
    bind_log_context(
        **{
            CTX_CYCLE_ID: cycle_id,
            CTX_QUALITY_SCORE: "research_host",
            CTX_RESEARCH_HEARTBEAT: 0.0,
        }
    )


def unbind_trade_context() -> None:
    """Remove trade/order keys after a tool call completes."""
    structlog.contextvars.unbind_contextvars(CTX_TRADE_ID, CTX_ORDER_ID)


def log_banner(logger: structlog.stdlib.BoundLogger, title: str, *, lines: tuple[str, ...] = ()) -> None:
    """Operator-facing banner (replaces ``print`` for process entry points)."""
    logger.info(title)
    for line in lines:
        logger.info(line)


def ensure_utf8_stdio() -> None:
    """Force UTF-8 on stdio (Windows consoles)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
