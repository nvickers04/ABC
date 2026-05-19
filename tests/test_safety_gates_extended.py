"""SafetyController gates beyond characterization (token + priority)."""

from __future__ import annotations

import pytest

from test_runtime_characterization import (
    StubCostTracker,
    StubGateway,
    _make_safety,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_evaluate_token_limit_before_llm_cost():
    """Daily token bucket breach halts before LLM dollar ceiling."""
    from core.runtime import SafetyController

    gw = StubGateway()
    tracker = StubCostTracker(today_llm_cost=0.0)
    tracker.check_daily_token_limits = lambda: "MAX_DAILY_LLM_COMPLETION_TOKENS"  # type: ignore[method-assign]

    safety = SafetyController(
        gw,
        tracker,
        max_daily_loss_pct=50.0,
        intraday_drawdown_pct=50.0,
        max_daily_llm_cost=999.0,
    )
    safety.capture_start_of_day_cash()
    verdict = safety.evaluate()
    assert verdict.triggered is True
    assert "token" in verdict.reason.lower()
    assert verdict.llm_cost_breached is False


def test_evaluate_llm_cost_when_tokens_ok():
    gw = StubGateway()
    tracker = StubCostTracker(today_llm_cost=100.0)
    tracker.check_daily_token_limits = lambda: None  # type: ignore[method-assign]

    safety, _, _ = _make_safety(
        today_llm_cost=100.0,
        max_daily_llm_cost=50.0,
    )
    safety.capture_start_of_day_cash()
    verdict = safety.evaluate()
    assert verdict.triggered is True
    assert verdict.llm_cost_breached is True
    assert "cost" in verdict.reason.lower()


def test_daily_loss_beats_drawdown_in_evaluate():
    """First breach wins: daily loss checked before drawdown."""
    safety, gw, _ = _make_safety(
        net_liquidation=100_000.0,
        max_daily_loss_pct=10.0,
        intraday_drawdown_pct=1.0,
    )
    safety.capture_start_of_day_cash()
    gw.net_liquidation = 70_000.0
    verdict = safety.evaluate()
    assert verdict.triggered is True
    assert "Daily loss" in verdict.reason
    assert verdict.daily_loss_pct is not None
    assert verdict.drawdown_pct is None
