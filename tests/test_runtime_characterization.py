"""Runtime characterization tests.

These tests lock the *current* behavior of the freshly-extracted runtime
modules so that future refactor PRs cannot silently change semantics.

Coverage:

* :class:`SafetyController` — daily-loss / drawdown / LLM-cost gating,
  start-of-day capture, aggregate :meth:`evaluate`.
* :class:`StateContextBuilder` — section ordering, key prompt strings,
  graceful degradation when the gateway raises.
* :class:`TradingAgent` shim parity — the agent's legacy ``_check_*`` /
  ``_capture_start_of_day_cash`` / ``_build_state_context`` methods must
  remain wire-compatible with the runtime modules they delegate to.
"""

from __future__ import annotations

import asyncio
import types
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest# ──────────────────────────────────────────────────────────────────
# Stubs / fixtures
# ──────────────────────────────────────────────────────────────────


class StubGateway:
    """Minimal gateway implementing only what the runtime modules touch."""

    def __init__(
        self,
        net_liquidation: float = 100_000.0,
        cash_value: float = 100_000.0,
        account_summary: dict | None = None,
        positions: list | None = None,
        open_orders: list | None = None,
        flatten_result: dict | None = None,
        raises: dict[str, Exception] | None = None,
    ) -> None:
        self.net_liquidation = net_liquidation
        self.cash_value = cash_value
        self._account_summary = account_summary or {
            "totalcashvalue": cash_value,
            "netliquidation": net_liquidation,
            "dailypnl": 0,
            "unrealizedpnl": 0,
            "realizedpnl": 0,
        }
        self._positions = positions or []
        self._open_orders = open_orders or []
        self._flatten_result = flatten_result or {
            "positions_closed": 0,
            "positions_total": 0,
            "errors": [],
        }
        self._raises = raises or {}

    async def get_account_summary(self) -> dict:
        if "get_account_summary" in self._raises:
            raise self._raises["get_account_summary"]
        return self._account_summary

    async def get_positions(self) -> list:
        if "get_positions" in self._raises:
            raise self._raises["get_positions"]
        return self._positions

    async def get_open_orders(self) -> list:
        if "get_open_orders" in self._raises:
            raise self._raises["get_open_orders"]
        return self._open_orders

    async def flatten_all(self) -> dict:
        return self._flatten_result


class StubMarketHours:
    def __init__(self, info: dict) -> None:
        self._info = info

    def get_session_info(self) -> dict:
        return self._info


class StubBudgetSummary:
    def __init__(self, today_llm_cost: float) -> None:
        self.today_llm_cost = today_llm_cost


class StubCostTracker:
    def __init__(self, today_llm_cost: float = 0.0) -> None:
        self._today = today_llm_cost

    def set(self, value: float) -> None:
        self._today = value

    def get_budget_summary(self) -> StubBudgetSummary:
        return StubBudgetSummary(self._today)


# ──────────────────────────────────────────────────────────────────
# SafetyController
# ──────────────────────────────────────────────────────────────────


def _make_safety(
    *,
    net_liquidation: float = 100_000.0,
    cash_value: float = 100_000.0,
    today_llm_cost: float = 0.0,
    max_daily_loss_pct: float = 3.0,
    intraday_drawdown_pct: float = 5.0,
    max_daily_llm_cost: float = 50.0,
):
    from core.runtime import SafetyController

    gw = StubGateway(net_liquidation=net_liquidation, cash_value=cash_value)
    ct = StubCostTracker(today_llm_cost=today_llm_cost)
    safety = SafetyController(
        gw,
        ct,
        max_daily_loss_pct=max_daily_loss_pct,
        intraday_drawdown_pct=intraday_drawdown_pct,
        max_daily_llm_cost=max_daily_llm_cost,
    )
    return safety, gw, ct


class TestSafetyController:
    def test_capture_start_of_day_uses_net_liq_when_positive(self):
        safety, gw, _ = _make_safety(net_liquidation=120_000, cash_value=50_000)
        baseline = safety.capture_start_of_day_cash()
        assert baseline == 120_000
        assert safety.start_of_day_cash == 120_000

    def test_capture_start_of_day_falls_back_to_cash(self):
        safety, gw, _ = _make_safety(net_liquidation=0, cash_value=50_000)
        assert safety.capture_start_of_day_cash() == 50_000

    def test_capture_start_of_day_returns_none_when_both_zero(self):
        safety, _, _ = _make_safety(net_liquidation=0, cash_value=0)
        assert safety.capture_start_of_day_cash() is None
        assert safety.start_of_day_cash is None

    def test_capture_start_of_day_is_idempotent(self):
        safety, gw, _ = _make_safety(net_liquidation=100_000)
        safety.capture_start_of_day_cash()
        gw.net_liquidation = 200_000  # change underlying — must not re-capture
        safety.capture_start_of_day_cash()
        assert safety.start_of_day_cash == 100_000

    def test_check_daily_loss_below_threshold_returns_none(self):
        safety, gw, _ = _make_safety(net_liquidation=100_000, max_daily_loss_pct=3.0)
        safety.capture_start_of_day_cash()
        gw.net_liquidation = 98_000  # 2% loss
        assert safety.check_daily_loss() is None

    def test_check_daily_loss_at_threshold_triggers(self):
        safety, gw, _ = _make_safety(net_liquidation=100_000, max_daily_loss_pct=3.0)
        safety.capture_start_of_day_cash()
        gw.net_liquidation = 97_000  # exactly 3%
        result = safety.check_daily_loss()
        assert result is not None
        assert result == pytest.approx(3.0)

    def test_check_daily_loss_above_threshold_returns_pct(self):
        safety, gw, _ = _make_safety(net_liquidation=100_000, max_daily_loss_pct=3.0)
        safety.capture_start_of_day_cash()
        gw.net_liquidation = 95_000  # 5% loss
        result = safety.check_daily_loss()
        assert result == pytest.approx(5.0)

    def test_check_daily_loss_no_baseline_returns_none(self):
        safety, _, _ = _make_safety()
        # Never captured baseline
        assert safety.check_daily_loss() is None

    def test_check_daily_loss_uses_cash_when_net_liq_zero(self):
        """Original semantics: if net_liq is 0, fall back to cash."""
        safety, gw, _ = _make_safety(net_liquidation=100_000, max_daily_loss_pct=3.0)
        safety.capture_start_of_day_cash()
        gw.net_liquidation = 0
        gw.cash_value = 90_000  # 10% loss via cash fallback
        result = safety.check_daily_loss()
        assert result == pytest.approx(10.0)

    def test_check_intraday_drawdown_tracks_high_water(self):
        safety, gw, _ = _make_safety(net_liquidation=100_000, intraday_drawdown_pct=5.0)
        # First call sets high-water at 100k
        assert safety.check_intraday_drawdown() is None
        assert safety.session_high_water == 100_000
        # NLV climbs → high-water rises
        gw.net_liquidation = 110_000
        assert safety.check_intraday_drawdown() is None
        assert safety.session_high_water == 110_000
        # Drop 4% — below threshold
        gw.net_liquidation = 105_600  # 4% off 110k
        assert safety.check_intraday_drawdown() is None
        # Drop 6% — triggers
        gw.net_liquidation = 103_400  # ~6% off 110k
        result = safety.check_intraday_drawdown()
        assert result is not None
        assert result >= 5.0

    def test_check_intraday_drawdown_returns_none_when_no_value(self):
        safety, gw, _ = _make_safety(net_liquidation=0, cash_value=0)
        assert safety.check_intraday_drawdown() is None

    def test_check_llm_cost_below_ceiling(self):
        safety, _, _ = _make_safety(today_llm_cost=10.0, max_daily_llm_cost=50.0)
        assert safety.check_llm_cost() is False

    def test_check_llm_cost_at_or_above_ceiling(self):
        safety, _, ct = _make_safety(today_llm_cost=50.0, max_daily_llm_cost=50.0)
        assert safety.check_llm_cost() is True
        ct.set(75.0)
        assert safety.check_llm_cost() is True

    def test_evaluate_priority_daily_loss_wins(self):
        safety, gw, ct = _make_safety(
            net_liquidation=100_000,
            max_daily_loss_pct=3.0,
            intraday_drawdown_pct=5.0,
            today_llm_cost=999.0,
            max_daily_llm_cost=50.0,
        )
        safety.capture_start_of_day_cash()
        gw.net_liquidation = 90_000  # both loss + drawdown breach + cost breach
        verdict = safety.evaluate()
        assert verdict.triggered is True
        assert "Daily loss" in verdict.reason
        # Other checks should not be evaluated past the first trigger.
        assert verdict.drawdown_pct is None
        assert verdict.llm_cost_breached is False

    def test_evaluate_drawdown_when_no_daily_loss(self):
        safety, gw, _ = _make_safety(intraday_drawdown_pct=5.0)
        # No baseline → no daily-loss path possible.
        # Set up high-water then drop.
        safety.check_intraday_drawdown()
        gw.net_liquidation = 80_000
        verdict = safety.evaluate()
        assert verdict.triggered is True
        assert "drawdown" in verdict.reason.lower()

    def test_evaluate_llm_cost_when_clean(self):
        safety, _, _ = _make_safety(
            today_llm_cost=100.0, max_daily_llm_cost=50.0
        )
        verdict = safety.evaluate()
        assert verdict.triggered is True
        assert verdict.llm_cost_breached is True
        assert "LLM cost" in verdict.reason

    def test_evaluate_clean_returns_no_trigger(self):
        safety, _, _ = _make_safety()
        verdict = safety.evaluate()
        assert verdict.triggered is False
        assert verdict.reason == ""


# ──────────────────────────────────────────────────────────────────
# StateContextBuilder
# ──────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


class TestStateContextBuilder:
    def _builder(self, **gw_kwargs):
        from core.runtime import StateContextBuilder

        gw = StubGateway(**gw_kwargs)
        mh = StubMarketHours(
            {
                "session": "regular",
                "current_time_et": "10:30:00",
                "minutes_to_close": 330,
            }
        )
        return StateContextBuilder(gw, market_hours_provider=mh), gw

    def test_build_includes_required_sections(self):
        builder, _ = self._builder()
        text = _run(builder.build())
        # Required section headers (parity-locked).
        assert "MARKET: REGULAR" in text
        assert "ACCOUNT" in text
        assert "POSITIONS" in text
        assert "OPEN ORDERS" in text

    def test_build_premarket_warns_about_limit_only(self):
        from core.runtime import StateContextBuilder

        gw = StubGateway()
        mh = StubMarketHours(
            {"session": "premarket", "current_time_et": "08:00:00", "minutes_to_open": 90}
        )
        builder = StateContextBuilder(gw, market_hours_provider=mh)
        text = _run(builder.build())
        assert "PREMARKET" in text
        assert "Limit orders ONLY" in text

    def test_build_closed_session_warns_no_orders(self):
        from core.runtime import StateContextBuilder

        gw = StubGateway()
        mh = StubMarketHours(
            {"session": "closed", "current_time_et": "22:00:00"}
        )
        builder = StateContextBuilder(gw, market_hours_provider=mh)
        text = _run(builder.build())
        assert "CLOSED" in text
        assert "do NOT place orders" in text

    def test_build_no_positions_message(self):
        builder, _ = self._builder(positions=[])
        text = _run(builder.build())
        assert "No open positions." in text

    def test_build_with_positions_lists_each(self):
        builder, _ = self._builder(
            positions=[
                {
                    "symbol": "NVDA",
                    "quantity": 10,
                    "avg_cost": 100.0,
                    "market_price": 110.0,
                    "unrealized_pnl": 100.0,
                    "sec_type": "STK",
                }
            ]
        )
        text = _run(builder.build())
        assert "NVDA" in text
        assert "verdict required" in text.lower() or "verdict" in text.lower()

    def test_build_handles_account_summary_failure(self):
        builder, _ = self._builder(
            raises={"get_account_summary": RuntimeError("broker dropped")}
        )
        text = _run(builder.build())
        assert "Account error:" in text
        # Builder must not bubble the exception.

    def test_build_handles_positions_failure(self):
        builder, _ = self._builder(
            raises={"get_positions": RuntimeError("boom")}
        )
        text = _run(builder.build())
        assert "Position error:" in text

    def test_build_handles_open_orders_failure(self):
        builder, _ = self._builder(
            raises={"get_open_orders": RuntimeError("boom")}
        )
        text = _run(builder.build())
        assert "Order error:" in text

    def test_build_idle_cash_warning(self):
        builder, _ = self._builder(
            account_summary={
                "totalcashvalue": 80_000,
                "netliquidation": 100_000,
                "dailypnl": 0,
                "unrealizedpnl": 0,
                "realizedpnl": 0,
            },
            positions=[
                {
                    "symbol": "NVDA",
                    "quantity": 100,
                    "avg_cost": 200.0,
                    "market_price": 200.0,
                    "unrealized_pnl": 0,
                    "sec_type": "STK",
                }
            ],
        )
        text = _run(builder.build())
        # 80k cash / 100k NLV = 80% > 30% threshold
        assert "IDLE CASH" in text


# ──────────────────────────────────────────────────────────────────
# Agent shim parity — the agent's legacy methods must keep delegating.
# ──────────────────────────────────────────────────────────────────


class TestAgentShimParity:
    """Verify TradingAgent's legacy attributes still map to runtime modules."""

    def _make_agent(self, **gw_kwargs):
        # Avoid network/IO during agent construction.
        from core import agent as agent_mod
        from core.runtime import SafetyController, StateContextBuilder
        from data import cost_tracker as ct_mod

        gw = StubGateway(**gw_kwargs)
        cost = StubCostTracker()

        # Bypass the heavy __init__: fabricate just the fields under test.
        a = agent_mod.TradingAgent.__new__(agent_mod.TradingAgent)
        a.gateway = gw
        a.cost_tracker = cost
        a._state_context_builder = StateContextBuilder(
            gw,
            market_hours_provider=StubMarketHours(
                {"session": "regular", "current_time_et": "10:30:00"}
            ),
        )
        a._safety = SafetyController(
            gw,
            cost,
            max_daily_loss_pct=3.0,
            intraday_drawdown_pct=5.0,
            max_daily_llm_cost=50.0,
        )
        return a, gw, cost

    def test_start_of_day_cash_property_round_trips(self):
        a, gw, _ = self._make_agent()
        assert a._start_of_day_cash is None
        a._capture_start_of_day_cash()
        assert a._start_of_day_cash == 100_000
        # Setter still works for legacy code paths.
        a._start_of_day_cash = 50_000
        assert a._safety.start_of_day_cash == 50_000

    def test_session_high_water_property_round_trips(self):
        a, _, _ = self._make_agent()
        assert a._session_high_water is None
        a._check_intraday_drawdown()  # initializes high-water
        assert a._session_high_water == 100_000
        a._session_high_water = 200_000
        assert a._safety.session_high_water == 200_000

    def test_check_daily_loss_delegates(self):
        a, gw, _ = self._make_agent()
        a._capture_start_of_day_cash()
        gw.net_liquidation = 90_000  # 10% loss
        assert a._check_daily_loss() == pytest.approx(10.0)

    def test_check_intraday_drawdown_delegates(self):
        a, gw, _ = self._make_agent()
        a._check_intraday_drawdown()
        gw.net_liquidation = 80_000  # 20% drawdown
        assert a._check_intraday_drawdown() == pytest.approx(20.0)

    def test_check_llm_cost_delegates(self):
        a, _, ct = self._make_agent()
        assert a._check_llm_cost() is False
        ct.set(100.0)
        assert a._check_llm_cost() is True

    def test_build_state_context_delegates(self):
        a, _, _ = self._make_agent()
        text = _run(a._build_state_context())
        assert "MARKET" in text
        assert "ACCOUNT" in text
        assert "POSITIONS" in text


# ──────────────────────────────────────────────────────────────────
# CycleScheduler
# ──────────────────────────────────────────────────────────────────


class StubWakeBus:
    def __init__(self, reasons: list[str] | None = None) -> None:
        self._reasons = list(reasons or ["timer"])
        self.calls: list[float] = []

    async def wait(self, timeout: float) -> str:
        self.calls.append(timeout)
        return self._reasons.pop(0) if self._reasons else "timer"


class StubAgent:
    """Minimal agent stub matching CycleScheduler's protocol."""

    def __init__(
        self,
        cycle_returns: list[int] | None = None,
        cycle_raises: list[Exception | None] | None = None,
    ) -> None:
        self._halted = False
        self._start_of_day_cash: Optional[float] = None  # type: ignore[name-defined]
        self._last_wake_reason = ""
        self.captured = 0
        self.cycle_calls = 0
        self._cycle_returns = list(cycle_returns or [60])
        self._cycle_raises = list(cycle_raises or [])

    def _capture_start_of_day_cash(self) -> None:
        self.captured += 1
        if self._start_of_day_cash is None:
            self._start_of_day_cash = 100_000.0

    async def run_cycle(self) -> int:
        self.cycle_calls += 1
        if self._cycle_raises:
            exc = self._cycle_raises.pop(0)
            if exc is not None:
                raise exc
        return self._cycle_returns.pop(0) if self._cycle_returns else 60


# Optional import shim for type-hint compatibility above.
from typing import Optional  # noqa: E402


class TestCycleScheduler:
    def _make(self, **kwargs):
        from core.runtime.scheduler import CycleScheduler

        agent = kwargs.pop("agent", None) or StubAgent()
        wake = kwargs.pop("wake_bus", None) or StubWakeBus()
        mh = kwargs.pop(
            "market_hours_provider", None
        ) or StubMarketHours({"session": "regular"})

        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float):
            sleep_calls.append(seconds)

        sched = CycleScheduler(
            agent=agent,
            wake_bus=wake,
            market_hours_provider=mh,
            sleep=fake_sleep,
            **kwargs,
        )
        return sched, agent, wake, mh, sleep_calls

    def test_handle_halted_returns_true_when_market_closed(self):
        sched, _, _, _, sleeps = self._make(
            market_hours_provider=StubMarketHours({"session": "closed"})
        )
        result = _run(sched._handle_halted())
        assert result is True
        assert sleeps == [sched.halted_poll_seconds]

    def test_handle_halted_returns_false_when_market_open(self):
        sched, _, _, _, _ = self._make(
            market_hours_provider=StubMarketHours({"session": "regular"})
        )
        assert _run(sched._handle_halted()) is False

    def test_handle_halted_swallows_market_hours_failure(self):
        class Boom:
            def get_session_info(self):
                raise RuntimeError("api down")

        sched, _, _, _, _ = self._make(market_hours_provider=Boom())
        # Must NOT raise; must return False so the loop keeps polling.
        assert _run(sched._handle_halted()) is False

    def test_after_hours_sleep_only_when_closed_and_late(self):
        # Closed but before 8 PM ET — no sleep.
        sched, _, _, _, sleeps = self._make(
            market_hours_provider=StubMarketHours({"session": "closed"}),
            now_et=lambda: datetime(2026, 4, 27, 19, 30, tzinfo=ZoneInfo("America/New_York")),
        )
        assert _run(sched._maybe_after_hours_sleep()) is False
        assert sleeps == []

    def test_after_hours_sleep_skips_when_market_open(self):
        sched, _, _, _, sleeps = self._make(
            market_hours_provider=StubMarketHours({"session": "regular"}),
            now_et=lambda: datetime(2026, 4, 27, 22, 0, tzinfo=ZoneInfo("America/New_York")),
        )
        assert _run(sched._maybe_after_hours_sleep()) is False
        assert sleeps == []

    def test_after_hours_sleep_triggers_when_closed_and_late(self):
        sched, _, _, _, sleeps = self._make(
            market_hours_provider=StubMarketHours({"session": "closed"}),
            now_et=lambda: datetime(2026, 4, 27, 21, 0, tzinfo=ZoneInfo("America/New_York")),
        )
        assert _run(sched._maybe_after_hours_sleep()) is True
        assert sleeps == [sched.after_hours_sleep_seconds]

    def test_run_one_cycle_records_wake_reason(self):
        sched, agent, wake, _, _ = self._make(
            agent=StubAgent(cycle_returns=[42]),
            wake_bus=StubWakeBus(reasons=["scorer_round_5"]),
        )
        _run(sched._run_one_cycle())
        assert agent.cycle_calls == 1
        assert wake.calls == [42]
        assert agent._last_wake_reason == "scorer_round_5"

    def test_run_one_cycle_swallows_exception_with_cooldown(self):
        sched, agent, wake, _, sleeps = self._make(
            agent=StubAgent(cycle_raises=[RuntimeError("boom")]),
        )
        _run(sched._run_one_cycle())
        # Must not have called wake_bus when the cycle threw.
        assert wake.calls == []
        # Error cooldown sleep must have run.
        assert sleeps == [sched.cycle_error_cooldown_seconds]

    def test_run_one_cycle_propagates_keyboard_interrupt(self):
        sched, agent, _, _, _ = self._make(
            agent=StubAgent(cycle_raises=[KeyboardInterrupt()]),
        )
        with pytest.raises(KeyboardInterrupt):
            _run(sched._run_one_cycle())

    def test_run_loop_breaks_when_halted_and_market_closed(self):
        agent = StubAgent()
        agent._halted = True
        sched, _, _, _, _ = self._make(
            agent=agent,
            market_hours_provider=StubMarketHours({"session": "closed"}),
        )
        # Must terminate, not hang.
        _run(sched.run())

    def test_run_loop_terminates_on_keyboard_interrupt_in_cycle(self):
        agent = StubAgent(cycle_raises=[KeyboardInterrupt()])
        sched, _, _, _, _ = self._make(agent=agent)
        # KeyboardInterrupt inside _run_one_cycle is caught by run() and
        # terminates the loop cleanly.
        _run(sched.run())
        assert agent.cycle_calls == 1

    def test_run_loop_captures_start_of_day_cash_once(self):
        # When _start_of_day_cash is None at loop start, capture is invoked.
        # Use KeyboardInterrupt to terminate cleanly after one iteration.
        agent = StubAgent(cycle_raises=[KeyboardInterrupt()])
        assert agent._start_of_day_cash is None
        sched, _, _, _, _ = self._make(agent=agent)
        _run(sched.run())
        assert agent.captured >= 1
        assert agent._start_of_day_cash == 100_000.0
