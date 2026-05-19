"""Profile optimization scoring and grid."""

from __future__ import annotations

import pytest

from core.profile_optimization import (
    DEFAULT_CYCLES_PER_DAY,
    REFERENCE_CYCLES_PER_DAY,
    ProfileCandidate,
    build_candidate_grid,
    composite_score,
    diff_against_baseline,
    lookback_dates,
    profit_factor_from_trades,
    score_backtest_result,
)
from core.simulation.sim_broker import SimFill
from core.simulation.stats import build_backtest_result


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_composite_score_weights():
    # Perfect normalized inputs -> 1.0
    assert composite_score(2.0, 2.5, 100.0) == pytest.approx(1.0)
    assert composite_score(0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_composite_score_cycle_normalization():
    # At reference density, raw and adjusted match
    assert composite_score(
        1.0, 1.5, 60.0, cycles_per_day=REFERENCE_CYCLES_PER_DAY
    ) == composite_score(1.0, 1.5, 60.0)
    # Fewer cycles boosts pf/win contribution toward reference comparability
    low = composite_score(1.0, 1.5, 60.0, cycles_per_day=DEFAULT_CYCLES_PER_DAY)
    raw = composite_score(1.0, 1.5, 60.0)
    assert low >= raw


def test_profit_factor_no_losses():
    trades = [SimFill("A", "SELL", 1, 10.0, pnl=50.0), SimFill("A", "SELL", 1, 10.0, pnl=30.0)]
    assert profit_factor_from_trades(trades) > 0


def test_build_candidate_grid_quick():
    quick = build_candidate_grid(include_perturbations=False)
    assert len(quick) == 3
    full = build_candidate_grid(include_perturbations=True)
    assert len(full) > len(quick)


def test_score_backtest_result():
    r = build_backtest_result(
        profile="balanced",
        start_date="2024-01-02",
        end_date="2024-01-05",
        trading_days=3,
        cycles_run=6,
        initial_equity=100_000.0,
        equity_curve=[("2024-01-02", 100_000.0), ("2024-01-05", 100_200.0)],
        closed_trades=[
            SimFill("AAPL", "SELL", 1, 100.0, pnl=120.0),
            SimFill("AAPL", "SELL", 1, 100.0, pnl=-40.0),
        ],
        llm_cost_usd=0.1,
    )
    scored = score_backtest_result(r, cycles_per_day=2)
    assert "composite_score" in scored
    assert scored["cycles_per_day"] == 2
    assert scored["reference_cycles_per_day"] == REFERENCE_CYCLES_PER_DAY
    assert scored["profit_factor"] == pytest.approx(3.0)  # 120/40


def test_diff_against_baseline_balanced():
    cand = ProfileCandidate("conservative", "conservative", patches={})
    diff = diff_against_baseline(cand, baseline_profile="balanced")
    assert "profit_profile" in diff or "risk" in diff


def test_lookback_dates():
    start, end = lookback_dates(7, end_date="2024-06-10")
    assert start == "2024-06-04"
    assert end == "2024-06-10"
