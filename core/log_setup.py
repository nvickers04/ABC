"""Shared console + rotating file logging for process entry points."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_DEFAULT_QUIET = (
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "ib_insync.wrapper",
    "ib_insync.ib",
    "ib_insync.client",
    "ib_insync.decoder",
    "ib_insync.connection",
    "ib_insync.flexreport",
    "ib_insync.order",
    "charset_normalizer",
    "hpack",
    "h2",
    "nest_asyncio",
    "grpc",
    "xai_sdk",
)


def configure_root_logging(
    log_file: str,
    *,
    verbose: bool = False,
    extra_quiet_loggers: tuple[str, ...] = (),
) -> None:
    """Attach stdout + midnight-rotating file handlers to the root logger."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(fmt))
        root.addHandler(console)

        fh = TimedRotatingFileHandler(
            log_dir / log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)

    quiet = set(_DEFAULT_QUIET) | set(extra_quiet_loggers)
    for name in quiet:
        logging.getLogger(name).setLevel(logging.WARNING)

    try:
        from core.log_context import configure_structlog

        configure_structlog(verbose=verbose)
    except Exception:
        logging.getLogger(__name__).debug(
            "structlog setup skipped", exc_info=True
        )
