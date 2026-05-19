"""ProfitConfig per-profile cache for grid search."""

from __future__ import annotations

import os

import pytest

from core.central_profit_config import (
    clear_profit_profile_cache,
    get_profit_profile_cache_stats,
    load_cached_profit_profile,
)
from core.profile_optimization import build_candidate_grid
from core.profit_profiles import PROFIT_PROFILE_ENV


@pytest.fixture(autouse=True)
def _reset_profile_cache():
    clear_profit_profile_cache()
    yield
    clear_profit_profile_cache()


def test_cache_hit_on_second_load():
    a = load_cached_profit_profile("conservative", dotenv=False)
    b = load_cached_profit_profile("conservative", dotenv=False)
    assert a.risk.max_daily_llm_cost == b.risk.max_daily_llm_cost
    stats = get_profit_profile_cache_stats()
    assert stats.hits >= 1
    assert stats.misses == 1


def test_cache_miss_on_env_change():
    load_cached_profit_profile("balanced", dotenv=False)
    os.environ["MAX_DAILY_LLM_COST"] = "99.0"
    try:
        cfg = load_cached_profit_profile("balanced", dotenv=False)
        assert cfg.risk.max_daily_llm_cost == pytest.approx(99.0)
        stats = get_profit_profile_cache_stats()
        assert stats.misses >= 2
    finally:
        os.environ.pop("MAX_DAILY_LLM_COST", None)


def test_build_candidate_grid_uses_cache(caplog):
    import logging

    caplog.set_level(logging.INFO)
    build_candidate_grid(include_perturbations=True, use_cache=True)
    stats = get_profit_profile_cache_stats()
    assert stats.hits > 0
    assert any("ProfitConfig profile cache" in r.message for r in caplog.records)


def test_clear_resets_stats():
    load_cached_profit_profile("aggressive", dotenv=False)
    clear_profit_profile_cache()
    stats = get_profit_profile_cache_stats()
    assert stats.hits == 0
    assert stats.misses == 0


def test_active_profile_restored_after_cached_load():
    os.environ[PROFIT_PROFILE_ENV] = "balanced"
    load_cached_profit_profile("conservative", dotenv=False)
    assert os.environ.get(PROFIT_PROFILE_ENV) == "balanced"
