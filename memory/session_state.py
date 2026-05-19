"""Process-local mutable state for the memory layer (not stored in PostgreSQL).

Pending order context bridges plan_order → insert_execution_snapshot across
tool calls. The calibration version counter lets the simulator detect stale
slippage tables without re-querying on every tick.
"""

from __future__ import annotations

pending_graduated_params: dict[str, int] = {}
pending_order_context: dict[str, dict] = {}
calibration_version: int = 0


def reset_session_state() -> None:
    """Clear pending lookups and reset the calibration counter (tests / reset_state)."""
    global calibration_version
    calibration_version = 0
    pending_graduated_params.clear()
    pending_order_context.clear()


def bump_calibration_version() -> None:
    """Increment after ``upsert_calibrated_slippage`` writes."""
    global calibration_version
    calibration_version += 1
