"""CycleScheduler — outer loop / wake / cooldown supervisor.

Extracted from :func:`core.agent.run_agent`. The scheduler is responsible
for:

* Polling the agent's halted state and yielding while the market is closed.
* Detecting "after-hours sleep" windows (post 8 PM ET).
* Calling :meth:`TradingAgent.run_cycle` and waiting on the wake bus for
  the next cycle.
* Catching unhandled cycle errors with a fixed cooldown (parity with the
  original loop).

Design notes
------------

* The scheduler does *not* own the agent's safety / state-context modules
  — those are still owned by :class:`TradingAgent`. The scheduler only
  drives the *outer* loop.
* ``wake_bus`` and ``market_hours_provider`` are injected so the loop can
  be unit-tested without the global singletons.
* Behavior parity with the original inline loop is locked by
  ``tests/test_runtime_characterization.py::TestCycleScheduler``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable, Optional, Protocol
from zoneinfo import ZoneInfo

from core.async_utils import safe_sleep as _safe_sleep
from core.loop_config import get_loop_config
from core.log_context import get_logger
from core.runtime.interfaces import MarketHoursProtocol, WakeBusProtocol

logger = get_logger(__name__)


def _now_et() -> datetime:
    """Return the current time in America/New_York.

    Lifted verbatim from ``core.agent`` so the scheduler can be exercised
    in isolation.
    """
    return datetime.now(ZoneInfo("America/New_York"))


class _AgentProtocol(Protocol):
    """Subset of ``TradingAgent`` the scheduler reads / writes."""

    _halted: bool
    _start_of_day_cash: Optional[float]
    _last_wake_reason: str

    def _capture_start_of_day_cash(self) -> Any: ...
    def run_cycle(self) -> Awaitable[int]: ...


class CycleScheduler:
    """Drives the outer agent loop until shutdown.

    The scheduler is intentionally a thin coordinator — every piece of
    actual decision logic still lives on the agent. This isolates loop
    timing concerns (Era-B churn hotspot) from cycle / decision logic.
    """

    def __init__(
        self,
        agent: _AgentProtocol,
        wake_bus: WakeBusProtocol,
        market_hours_provider: MarketHoursProtocol,
        *,
        now_et: Callable[[], datetime] = _now_et,
        sleep: Callable[[float], Awaitable[Any]] = _safe_sleep,
    ) -> None:
        sched = get_loop_config().scheduler_defaults()
        self.halted_poll_seconds: int = sched["halted_poll_seconds"]
        self.after_hours_sleep_seconds: int = sched["after_hours_sleep_seconds"]
        self.after_hours_threshold_hour: int = sched["after_hours_threshold_hour"]
        self.cycle_error_cooldown_seconds: int = sched["cycle_error_cooldown_seconds"]
        self.agent = agent
        self.wake_bus = wake_bus
        self.market_hours = market_hours_provider
        self._now_et = now_et
        self._sleep = sleep
        self._cycle = 0

    # ── Single-iteration helpers (testable in isolation) ───────────

    async def _handle_halted(self) -> bool:
        """If the agent is halted, sleep + check for shutdown.

        Returns ``True`` when the outer loop should ``break`` (market is
        closed), ``False`` when it should ``continue`` polling.
        """
        logger.critical("AGENT HALTED — waiting for market close")
        await self._sleep(self.halted_poll_seconds)
        try:
            info = self.market_hours.get_session_info()
            if info.get("session") == "closed":
                return True
        except Exception:
            pass
        return False

    async def _maybe_after_hours_sleep(self) -> bool:
        """Return ``True`` when the loop should ``continue`` after sleeping.

        Mirrors the original guard: when session=='closed' AND ET hour >=
        after-hours threshold, sleep 30m and skip this iteration.
        """
        try:
            info = self.market_hours.get_session_info()
            session = info.get("session", "")
            if session == "closed":
                if self._now_et().hour >= self.after_hours_threshold_hour:
                    logger.info(
                        "Market closed, extended hours over — sleeping "
                        f"{self.after_hours_sleep_seconds // 60} min"
                    )
                    await self._sleep(self.after_hours_sleep_seconds)
                    return True
        except Exception:
            pass
        return False

    async def _run_one_cycle(self) -> None:
        """Run a single cycle and wait for the next wake event.

        Errors raised by ``run_cycle`` are caught here so the loop
        survives transient broker / model failures (parity with the
        original ``try/except`` around ``agent.run_cycle()``).
        """
        try:
            wait_seconds = await self.agent.run_cycle()
            logger.info(f"Cooldown: up to {wait_seconds}s (event-driven)")
            try:
                from core.runtime.operating_context import get_operating_context
                _qctx = get_operating_context()
                _qm = _qctx.quality_matrix
                logger.info(
                    "QualityMatrix@cycle: overall=%s risk_mult=%.2f blocked=%s",
                    _qm.overall_quality, _qctx.risk_multiplier, _qctx.get_blocked_tool_categories()
                )
            except Exception:
                pass
            wake_reason = await self.wake_bus.wait(wait_seconds)
            self.agent._last_wake_reason = wake_reason
            logger.info(f"Woke: {wake_reason}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            await self._sleep(self.cycle_error_cooldown_seconds)

    # ── Main loop ──────────────────────────────────────────────────

    async def run(self) -> None:
        """Run the supervised cycle loop until halted-and-closed or Ctrl-C."""
        try:
            while True:
                if self.agent._halted:
                    if await self._handle_halted():
                        break
                    continue

                if self.agent._start_of_day_cash is None:
                    self.agent._capture_start_of_day_cash()

                if await self._maybe_after_hours_sleep():
                    continue

                self._cycle += 1
                logger.info(f"{'='*40} CYCLE {self._cycle} {'='*40}")

                await self._run_one_cycle()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
