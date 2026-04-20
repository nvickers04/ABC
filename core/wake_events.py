"""Asyncio wake bus — lets the agent loop sleep until something interesting happens.

Producers (scorer, broker hooks, manual triggers) call ``wake_bus.signal(reason)``.
The agent loop calls ``await wake_bus.wait(timeout)`` instead of ``asyncio.sleep``
and gets back the reason it woke (or ``"timeout"``).

Single-loop only. If you need to signal from another thread/loop, use
``loop.call_soon_threadsafe(wake_bus.signal, reason)``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class WakeBus:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: Optional[str] = None
        self._signalled_at: float = 0.0

    def signal(self, reason: str) -> None:
        """Wake any waiter. Last-write-wins on reason if multiple signals stack
        before the waiter resumes."""
        self._reason = reason
        self._signalled_at = time.time()
        self._event.set()

    async def wait(self, timeout: float) -> str:
        """Wait up to ``timeout`` seconds for a signal. Returns the reason
        (e.g. ``'scorer_round'``) or ``'timeout'``. Always clears the event
        before returning so the next cycle starts fresh."""
        if timeout <= 0:
            self._event.clear()
            self._reason = None
            return "no_wait"
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            reason = self._reason or "unknown"
        except asyncio.TimeoutError:
            reason = "timeout"
        finally:
            self._event.clear()
            self._reason = None
        return reason


# Module-level singleton
wake_bus = WakeBus()
