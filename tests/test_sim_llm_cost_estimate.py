"""Simulation LLM cost / token estimate helpers."""

from __future__ import annotations

import pytest

from core.simulation.backtest_llm import BacktestLLM, CyclePlanContext
from core.simulation.llm_cost_estimate import (
    LlmUsageEstimate,
    estimate_backtest_llm_upper_bound,
    estimate_optimizer_llm_upper_bound,
)
from data.cost_tracker import CostTracker, estimate_llm_cost_usd


def test_estimate_llm_cost_matches_cost_tracker():
    tracker = CostTracker()
    direct = estimate_llm_cost_usd("grok-4.3", 10_000, 2_000)
    via_tracker = tracker._calculate_cost("grok-4.3", 10_000, 2_000)
    assert direct == pytest.approx(via_tracker)


def test_backtest_llm_accumulates_tokens():
    llm = BacktestLLM()
    llm.prepare_cycle(
        CyclePlanContext(
            session_date="2024-06-03",
            cycle_index=0,
            top_symbol="SPY",
            top_score=0.7,
            traded_today=False,
        )
    )
    assert llm.prompt_tokens > 0
    assert llm.completion_tokens > 0
    assert llm.estimated_cost_usd > 0


def test_optimizer_upper_bound_scales_with_candidates():
    one = estimate_backtest_llm_upper_bound(10, 4)
    many = estimate_optimizer_llm_upper_bound(5, 10, 4)
    assert many.prompt_tokens == one.prompt_tokens * 5
    assert many.estimated_cost_usd == pytest.approx(one.estimated_cost_usd * 5)


def test_llm_usage_add_incremental():
    u = LlmUsageEstimate(model="grok-4.3")
    u.add(1000, 200, samples=1)
    u.add(500, 100, samples=1)
    assert u.prompt_tokens == 1500
    assert u.completion_tokens == 300
    assert u.sample_calls == 2
