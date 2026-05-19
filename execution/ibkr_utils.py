"""IBKR endpoint resolution utilities.

Ports and host come from :mod:`core.risk_execution_config` (env / CLI / mode).
"""

from __future__ import annotations

from typing import Tuple

from core.log_context import get_logger
from core.risk_execution_config import get_risk_execution_config

logger = get_logger(__name__)


def get_ibkr_port() -> int:
    """Resolve TWS API port from :class:`RiskExecutionConfig`."""
    return get_risk_execution_config().resolve_ibkr_port()


def get_ibkr_host() -> str:
    """Resolve TWS API host."""
    return get_risk_execution_config().ibkr_host


def is_paper_trading() -> bool:
    """True when resolved port is the configured paper port."""
    rc = get_risk_execution_config()
    return get_ibkr_port() == rc.ibkr_paper_port


def is_live_trading() -> bool:
    """True when resolved port is the configured live port."""
    rc = get_risk_execution_config()
    return get_ibkr_port() == rc.ibkr_live_port


def resolve_ibkr_endpoint(mode: str | None = None) -> Tuple[str, int, str]:
    """Return (host, port, mode_string). ``mode`` is ignored (kept for API compat)."""
    del mode
    host = get_ibkr_host()
    port = get_ibkr_port()
    rc = get_risk_execution_config()
    mode_str = "paper" if port == rc.ibkr_paper_port else "live"
    return host, port, mode_str


def format_ibkr_endpoint(mode: str | None = None) -> str:
    """Human-readable host:port (mode) for logging."""
    host, port, resolved_mode = resolve_ibkr_endpoint(mode)
    return f"{host}:{port} ({resolved_mode})"
