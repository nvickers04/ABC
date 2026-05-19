"""Freeze wall-clock time for simulation (gap guard, EOD, safety)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator
from unittest.mock import patch
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

# Modules that call ``datetime.now(timezone.utc)`` during cycles.
_PATCH_TARGETS = (
    "core.agent.datetime",
    "core.runtime.cycle.datetime",
    "core.runtime.scheduler.datetime",
    "core.runtime.review.datetime",
    "core.runtime.operating_context.datetime",
)


class _FrozenDateTime:
    """Subset of datetime API with a fixed ``now``."""

    def __init__(self, real_dt, frozen_utc: datetime):
        self._real = real_dt
        self._frozen = frozen_utc

    def now(self, tz=None):
        if tz is None:
            return self._frozen.replace(tzinfo=None)
        return self._frozen.astimezone(tz)

    def __getattr__(self, name):
        # Patched name is ``datetime`` in target modules; expose class attrs (min, max, …).
        if hasattr(self._real.datetime, name):
            return getattr(self._real.datetime, name)
        return getattr(self._real, name)


@contextmanager
def frozen_utc(now_utc: datetime) -> Iterator[datetime]:
    """Patch datetime in agent/runtime modules to ``now_utc``."""
    import datetime as real_datetime

    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    fake = _FrozenDateTime(real_datetime, now_utc)
    patches = [patch(target, fake) for target in _PATCH_TARGETS]
    for p in patches:
        p.start()
    try:
        yield now_utc
    finally:
        for p in reversed(patches):
            p.stop()


def et_to_utc(session_date: str, hour: int, minute: int) -> datetime:
    """Build UTC instant for a US/Eastern session clock time on ``session_date``."""
    local = datetime.strptime(session_date, "%Y-%m-%d").replace(
        hour=hour, minute=minute, second=0, microsecond=0, tzinfo=_ET
    )
    return local.astimezone(timezone.utc)
