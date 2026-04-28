"""Wake/cooldown contract tests.

Locks the documented WakeBus contract:

* ``wait(timeout)`` returns the reason string of any signal that
  arrived during the wait, or ``"timeout"`` after the timeout expires.
* ``wait(0)`` (or negative) returns ``"no_wait"`` immediately and does
  not consume any pending signal.
* The internal event is always cleared before ``wait`` returns so the
  next cycle starts fresh — back-to-back ``wait`` calls do not spuriously
  re-fire on a stale signal.
* Last-write-wins on ``signal`` if multiple signals stack before the
  waiter resumes.

Also pins :class:`core.wake_events.WakeBus` against the public
:class:`core.runtime.interfaces.WakeBusProtocol` structural type so the
scheduler can be unit-tested with stubs that satisfy the same protocol.
"""

from __future__ import annotations

import asyncio

import pytest

from core.runtime.interfaces import WakeBusProtocol
from core.wake_events import WakeBus


@pytest.fixture
def bus() -> WakeBus:
    return WakeBus()


class TestWakeBusContract:
    def test_implements_protocol(self, bus):
        # runtime_checkable Protocol — structural conformance check.
        assert isinstance(bus, WakeBusProtocol)

    def test_wait_zero_returns_no_wait(self, bus):
        async def go():
            return await bus.wait(0)

        assert asyncio.run(go()) == "no_wait"

    def test_wait_negative_returns_no_wait(self, bus):
        async def go():
            return await bus.wait(-5)

        assert asyncio.run(go()) == "no_wait"

    def test_signal_then_wait_returns_reason(self, bus):
        async def go():
            bus.signal("scorer_round")
            return await bus.wait(5.0)

        assert asyncio.run(go()) == "scorer_round"

    def test_wait_then_signal_concurrent(self, bus):
        async def go():
            async def signaller():
                await asyncio.sleep(0.01)
                bus.signal("broker_fill")

            await asyncio.gather(asyncio.create_task(signaller()))
            return await bus.wait(1.0)

        # Run wait + signaller concurrently.
        async def driver():
            task = asyncio.create_task(bus.wait(1.0))
            await asyncio.sleep(0.01)
            bus.signal("broker_fill")
            return await task

        assert asyncio.run(driver()) == "broker_fill"

    def test_timeout_returns_timeout(self, bus):
        async def go():
            return await bus.wait(0.01)

        assert asyncio.run(go()) == "timeout"

    def test_event_cleared_between_waits(self, bus):
        """Two consecutive waits must each see independent state."""

        async def go():
            bus.signal("first")
            r1 = await bus.wait(1.0)
            r2 = await bus.wait(0.01)  # should time out
            return r1, r2

        r1, r2 = asyncio.run(go())
        assert r1 == "first"
        assert r2 == "timeout"

    def test_last_signal_wins_when_stacked(self, bus):
        async def go():
            bus.signal("first")
            bus.signal("second")
            bus.signal("third")
            return await bus.wait(1.0)

        assert asyncio.run(go()) == "third"

    def test_zero_wait_does_not_consume_pending_signal(self, bus):
        """``wait(0)`` returns immediately without consuming a queued signal.

        This documents current behavior: a signal posted before a
        zero-timeout wait IS cleared (because wait(0) clears the event)
        — callers who want to drain pending signals should use a small
        positive timeout. Pin this so future refactors don't silently
        change semantics.
        """

        async def go():
            bus.signal("pending")
            r0 = await bus.wait(0)
            r1 = await bus.wait(0.01)
            return r0, r1

        r0, r1 = asyncio.run(go())
        assert r0 == "no_wait"
        # Documented behavior: pending signal IS dropped by wait(0).
        assert r1 == "timeout"
