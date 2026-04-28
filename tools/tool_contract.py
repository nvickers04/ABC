"""Tool execution contract — single source of truth for tool I/O shape.

Every tool dispatched through :class:`tools.tools_executor.ToolExecutor`
returns a :class:`ToolResult` whose ``data`` field follows the envelope
defined here:

```python
{
    "success": bool,            # required
    "error":   Optional[str],   # required (None on success)
    "is_realtime":  bool,       # required (default False)
    "data_warning": Optional[str],  # required (default None)
    # ...handler-specific keys merged flat alongside the envelope...
}
```

The executor keeps its existing ``_standardize_tool_payload`` helper for
back-compat normalization. New work should construct envelopes via
:func:`make_envelope` / :func:`make_error` and validate them via
:func:`validate_envelope` before they reach the LLM.

This module exists so tool authors can import a single contract without
pulling in the full executor (which depends on broker / data providers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# Required envelope keys (must always be present after normalization).
ENVELOPE_KEYS: frozenset[str] = frozenset(
    {"success", "error", "is_realtime", "data_warning"}
)

# Reserved envelope keys plus aliases the legacy executor recognizes.
RESERVED_KEYS: frozenset[str] = ENVELOPE_KEYS | frozenset({"data"})


@dataclass
class ToolResult:
    """Structured wrapper carried from executor → agent loop.

    ``data`` holds the standardized envelope (see module docstring).
    ``raw_json`` is a pre-serialized form so the agent can inject the
    result back into the LLM chat without re-encoding.
    """

    action: str
    data: Any
    success: bool
    raw_json: str

    def __str__(self) -> str:  # back-compat: chat injection uses str()
        return self.raw_json

    @classmethod
    def from_envelope(cls, action: str, envelope: dict) -> "ToolResult":
        """Build a :class:`ToolResult` from a validated envelope."""
        validate_envelope(envelope)
        return cls(
            action=action,
            data=envelope,
            success=bool(envelope.get("success")),
            raw_json=json.dumps(envelope, separators=(",", ":"), default=str),
        )


# ── Envelope helpers ─────────────────────────────────────────────


def make_envelope(
    *,
    success: bool = True,
    error: Optional[str] = None,
    is_realtime: bool = False,
    data_warning: Optional[str] = None,
    **fields: Any,
) -> dict:
    """Build a standardized envelope.

    Handler-specific keys are merged flat; passing one of the reserved
    envelope keys via ``fields`` raises :class:`ValueError` so callers
    cannot accidentally shadow envelope semantics.
    """
    overlap = set(fields) & ENVELOPE_KEYS
    if overlap:
        raise ValueError(
            f"Reserved envelope keys cannot be passed as data fields: "
            f"{sorted(overlap)}"
        )
    envelope: dict = {
        "success": bool(success),
        "error": None if error is None else str(error),
        "is_realtime": bool(is_realtime),
        "data_warning": data_warning,
    }
    envelope.update(fields)
    return envelope


def make_error(
    error: str,
    *,
    is_realtime: bool = False,
    data_warning: Optional[str] = None,
    **fields: Any,
) -> dict:
    """Build a failure envelope (``success=False``)."""
    return make_envelope(
        success=False,
        error=error,
        is_realtime=is_realtime,
        data_warning=data_warning,
        **fields,
    )


def validate_envelope(envelope: Any) -> None:
    """Raise :class:`ValueError` if ``envelope`` is not a valid tool envelope.

    Validation is intentionally strict: future tooling work depends on
    being able to assume the shape. The existing executor normalizes
    legacy outputs *before* hitting the agent loop, so this check should
    only ever fire on programmer error.
    """
    if not isinstance(envelope, dict):
        raise ValueError(
            f"Tool envelope must be a dict, got {type(envelope).__name__}"
        )
    missing = ENVELOPE_KEYS - envelope.keys()
    if missing:
        raise ValueError(f"Tool envelope missing required keys: {sorted(missing)}")
    if not isinstance(envelope["success"], bool):
        raise ValueError(
            f"Tool envelope 'success' must be bool, got "
            f"{type(envelope['success']).__name__}"
        )
    if envelope["error"] is not None and not isinstance(envelope["error"], str):
        raise ValueError(
            f"Tool envelope 'error' must be str|None, got "
            f"{type(envelope['error']).__name__}"
        )
    if not isinstance(envelope["is_realtime"], bool):
        raise ValueError(
            f"Tool envelope 'is_realtime' must be bool, got "
            f"{type(envelope['is_realtime']).__name__}"
        )
    # Cross-field invariant: success implies no error message.
    if envelope["success"] and envelope["error"] is not None:
        raise ValueError(
            "Tool envelope inconsistency: success=True but error is not None"
        )


__all__ = [
    "ENVELOPE_KEYS",
    "RESERVED_KEYS",
    "ToolResult",
    "make_envelope",
    "make_error",
    "validate_envelope",
]
