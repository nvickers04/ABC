"""IBKR endpoint resolution utilities.

SIMPLE: IBKR_PORT env var is the ONLY config. Default 7497 (paper).
- 7497 = Paper trading (default, safe)
- 7496 = Live trading (set explicitly when ready)

PAPER_MODE env var provides an additional safety check.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
from contextlib import closing
from typing import Tuple

logger = logging.getLogger(__name__)

# Standard IBKR ports
PAPER_PORT = 7497
LIVE_PORT = 7496
DEFAULT_PORT = PAPER_PORT  # Safe default

# PAPER_MODE env var — ultimate safety toggle
PAPER_MODE = os.getenv("PAPER_MODE", "True").lower() == "true"

# TRADING_MODE takes priority when set (aggressive_paper | paper | live)
_TRADING_MODE = os.getenv("TRADING_MODE", "").lower().strip()


def get_ibkr_port() -> int:
    """Get IBKR port from env var, default to paper (7497).

    Port resolution order:
    1. If TRADING_MODE=live → 7496 (live port)
    2. If TRADING_MODE=paper or aggressive_paper → 7497 (paper port)
    3. If PAPER_MODE=True (default) → 7497 regardless of IBKR_PORT
    4. Else fall back to IBKR_PORT env var
    """
    if _TRADING_MODE == "live":
        return LIVE_PORT   # 7496 — live trading
    if _TRADING_MODE in ("paper", "aggressive_paper"):
        return PAPER_PORT  # 7497 — paper trading
    # Legacy: PAPER_MODE toggle
    if PAPER_MODE:
        return PAPER_PORT
    return int(os.getenv("IBKR_PORT", DEFAULT_PORT))


def _get_ibkr_port_raw() -> int:
    """Get raw IBKR port from env var without PAPER_MODE override."""
    return int(os.getenv("IBKR_PORT", DEFAULT_PORT))


def get_ibkr_host() -> str:
    """Get IBKR host, default localhost."""
    return os.getenv("IBKR_HOST", "127.0.0.1")


def is_paper_trading() -> bool:
    """Check if using paper trading port."""
    return get_ibkr_port() == PAPER_PORT


def is_live_trading() -> bool:
    """Check if using live trading port."""
    return get_ibkr_port() == LIVE_PORT


def resolve_ibkr_endpoint(mode: str | None = None) -> Tuple[str, int, str]:
    """Resolve IBKR host/port. Mode param is ignored - use IBKR_PORT env var.
    
    Returns:
        Tuple of (host, port, mode_string)
    """
    host = get_ibkr_host()
    port = get_ibkr_port()
    mode_str = "paper" if port == PAPER_PORT else "live"

    return host, port, mode_str


def format_ibkr_endpoint(mode: str | None = None) -> str:
    """Return a human-readable host:port string with mode for logging."""
    host, port, resolved_mode = resolve_ibkr_endpoint(mode)
    return f"{host}:{port} ({resolved_mode})"


def _probe_ibkr_socket(host: str, port: int, timeout: float) -> bool:
    """Synchronous TCP probe to validate the IBKR host:port is reachable."""

    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            return True
    except OSError:
        return False


async def check_ibkr_connectivity(
    host: str | None = None,
    port: int | None = None,
    *,
    timeout_seconds: float = 5.0,
    attempts: int = 2,
) -> bool:
    """Asynchronously verify IBKR connectivity via a lightweight socket probe.

    Args:
        host: IBKR host override; defaults to resolved endpoint.
        port: IBKR port override; defaults to resolved endpoint.
        timeout_seconds: Timeout per attempt.
        attempts: Number of probe attempts before giving up.

    Returns:
        True if a TCP connection succeeds within the allotted attempts.
    """

    resolved_host, resolved_port, _ = resolve_ibkr_endpoint()
    target_host = host or resolved_host
    target_port = port or resolved_port

    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_probe_ibkr_socket, target_host, target_port, timeout_seconds),
                timeout=timeout_seconds + 1,
            )
        except (asyncio.TimeoutError, OSError) as exc:
            logger.warning(
                "IBKR connectivity probe %s/%s failed for %s:%s (%s)",
                attempt,
                attempts,
                target_host,
                target_port,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "IBKR connectivity probe %s/%s hit unexpected error for %s:%s (%s)",
                attempt,
                attempts,
                target_host,
                target_port,
                exc,
            )

    return False
