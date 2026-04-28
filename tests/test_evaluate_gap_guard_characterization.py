"""
Characterization tests for ``core.runtime.cycle.evaluate_gap_guard``.

The gap-guard branch was previously embedded inline in
``TradingAgent.run_cycle``. These tests pin its behavior so future cycle
refactors stay byte-identical.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from core.runtime import cycle as cycle_mod
from core.runtime.cycle import evaluate_gap_guard


def _agent(session: str = "regular", gap_guard_until=None):
    return SimpleNamespace(_current_session=session, _gap_guard_until=gap_guard_until)


def _patch_mh(monkeypatch, minutes_to_close):
    """Patch ``get_market_hours_provider`` to return a stub session-info dict."""
    class _MH:
        def get_session_info(self):
            return {"minutes_to_close": minutes_to_close}

    monkeypatch.setattr(cycle_mod, "get_market_hours_provider", lambda: _MH())


def _patch_dp(monkeypatch, quote):
    """Patch ``get_data_provider`` to yield an awaitable returning ``quote``."""
    class _DP:
        async def get_quote(self, sym):
            return quote

    monkeypatch.setattr(cycle_mod, "get_data_provider", lambda: _DP())


# ---------------------------------------------------------------------------
# Session gating
# ---------------------------------------------------------------------------

def test_non_regular_session_returns_empty_no_mutation():
    agent = _agent(session="premarket", gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""
    assert agent._gap_guard_until is None


def test_postmarket_session_returns_empty():
    agent = _agent(session="postmarket", gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""


# ---------------------------------------------------------------------------
# Active-guard branch (until > now)
# ---------------------------------------------------------------------------

def test_active_guard_emits_wait_prompt_with_minutes():
    until = datetime.now(timezone.utc) + timedelta(minutes=10)
    agent = _agent(gap_guard_until=until)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert "GAP GUARD ACTIVE" in out
    assert "more minutes" in out
    # State must not be mutated when guard is already active.
    assert agent._gap_guard_until == until


def test_expired_guard_sentinel_returns_empty():
    """A sentinel ``datetime.min`` indicates a previously-checked-and-clear gap.
    The branch should be skipped (truthy but ``< now``) and no prompt emitted."""
    agent = _agent(gap_guard_until=datetime.min.replace(tzinfo=timezone.utc))
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""


# ---------------------------------------------------------------------------
# First-cycle gap detection (until is None)
# ---------------------------------------------------------------------------

def test_first_cycle_outside_open_window_does_not_check_quote(monkeypatch):
    """If we are >5 min past the open, no quote check happens and state is untouched."""
    _patch_mh(monkeypatch, minutes_to_close=300)  # 90 min elapsed -> outside window
    # Quote provider would raise if called -> proves it isn't.
    def _boom():
        raise AssertionError("data provider should not be called")
    monkeypatch.setattr(cycle_mod, "get_data_provider", _boom)

    agent = _agent(gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""
    assert agent._gap_guard_until is None


def test_first_cycle_small_gap_clears_with_sentinel(monkeypatch):
    """SPY gap below threshold -> sentinel set, no prompt."""
    _patch_mh(monkeypatch, minutes_to_close=388)  # 2 min after open
    _patch_dp(monkeypatch, {"last": 100.0, "previous_close": 99.5})  # 0.5% gap
    agent = _agent(gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""
    # Sentinel: datetime.min UTC means "checked and clear; do not re-check."
    assert agent._gap_guard_until == datetime.min.replace(tzinfo=timezone.utc)


def test_first_cycle_large_gap_triggers_guard(monkeypatch):
    """SPY gap >= 2.0% -> guard activated for OPEN_GUARD_DELAY_MINUTES."""
    _patch_mh(monkeypatch, minutes_to_close=388)
    _patch_dp(monkeypatch, {"last": 103.0, "previous_close": 100.0})  # 3% gap
    agent = _agent(gap_guard_until=None)
    before = datetime.now(timezone.utc)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert "GAP GUARD" in out
    assert "3.0%" in out
    assert agent._gap_guard_until is not None
    # Guard should expire ~15 min from now.
    delta = agent._gap_guard_until - before
    assert timedelta(minutes=14) <= delta <= timedelta(minutes=16)


def test_first_cycle_quote_failure_sets_sentinel(monkeypatch):
    """An exception inside the quote try-block sets the sentinel and clears."""
    _patch_mh(monkeypatch, minutes_to_close=388)

    class _DPBoom:
        async def get_quote(self, sym):
            raise RuntimeError("provider down")

    monkeypatch.setattr(cycle_mod, "get_data_provider", lambda: _DPBoom())
    agent = _agent(gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""
    assert agent._gap_guard_until == datetime.min.replace(tzinfo=timezone.utc)


def test_first_cycle_zero_quote_no_state_change(monkeypatch):
    """If both ``last`` and ``previous_close`` are zero, neither branch fires."""
    _patch_mh(monkeypatch, minutes_to_close=388)
    _patch_dp(monkeypatch, {"last": 0, "previous_close": 0, "close": 0})
    agent = _agent(gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""
    # No sentinel set when prices are unusable.
    assert agent._gap_guard_until is None


def test_first_cycle_none_quote_no_state_change(monkeypatch):
    """``get_quote`` returning ``None`` skips both branches without mutation."""
    _patch_mh(monkeypatch, minutes_to_close=388)
    _patch_dp(monkeypatch, None)
    agent = _agent(gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""
    assert agent._gap_guard_until is None


def test_outer_exception_swallowed(monkeypatch):
    """If ``get_market_hours_provider`` itself raises, the outer except returns ''."""
    def _boom():
        raise RuntimeError("mh provider down")
    monkeypatch.setattr(cycle_mod, "get_market_hours_provider", _boom)
    agent = _agent(gap_guard_until=None)
    out = asyncio.run(evaluate_gap_guard(agent))
    assert out == ""
    assert agent._gap_guard_until is None
