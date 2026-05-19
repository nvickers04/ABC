"""IBKR TWS/Gateway connection lifecycle — error codes and reconnect hints.

Used by :mod:`execution.ibkr_core` for classifying disconnects (especially TWS
midnight restarts) and structured logging.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

# IB API connectivity messages (see IB TWS API Reference → Error Codes)
ERROR_CONNECTIVITY_LOST = 1100  # IB ↔ TWS lost; TWS may still be up
ERROR_CONNECTIVITY_RESTORED_DATA_LOST = 1101
ERROR_CONNECTIVITY_RESTORED = 1102
ERROR_TWS_SERVER_BROKEN = 2110  # TWS ↔ IB server broken (Gateway restart common)

TWS_RESTART_CODES: frozenset[int] = frozenset({ERROR_CONNECTIVITY_LOST, ERROR_TWS_SERVER_BROKEN})
TWS_RESTORED_CODES: frozenset[int] = frozenset(
    {ERROR_CONNECTIVITY_RESTORED_DATA_LOST, ERROR_CONNECTIVITY_RESTORED}
)
FARM_OK_CODES: frozenset[int] = frozenset({2104, 2106, 2158})


class DisconnectCause(str, Enum):
    """Why the API session dropped (best-effort attribution)."""

    UNKNOWN = "unknown"
    TWS_RESTART = "tws_restart"
    USER_DISCONNECT = "user_disconnect"
    HEARTBEAT_FAILED = "heartbeat_failed"
    CONNECT_FAILED = "connect_failed"


def classify_error_code(error_code: int) -> Optional[str]:
    """Return a lifecycle hint for ``ibkr_core`` error handling.

    Returns:
        ``tws_lost`` — schedule fast reconnect (1100, 2110).
        ``tws_restored`` — connectivity back; session may need refresh (1101, 1102).
        ``farm_ok`` — informational farm message (suppress noise).
        ``None`` — no special handling.
    """
    if error_code in TWS_RESTART_CODES:
        return "tws_lost"
    if error_code in TWS_RESTORED_CODES:
        return "tws_restored"
    if error_code in FARM_OK_CODES:
        return "farm_ok"
    return None


def reconnect_backoff_seconds(failure_count: int, *, base: float = 5.0, cap: float = 60.0) -> float:
    """Exponential backoff between reconnect attempts (seconds)."""
    n = max(0, int(failure_count))
    return min(cap, base * (2**min(n, 4)))
