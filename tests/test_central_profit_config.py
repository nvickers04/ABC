"""ProfitConfig master composition and profitability profiles."""

from __future__ import annotations

import os

import pytest

from core.central_profit_config import (
    PROFIT_PROFILE_ENV,
    ProfitConfig,
    get_profit_config,
    reload_profit_config,
)
from core.profit_profiles import normalize_profit_profile


@pytest.fixture(autouse=True)
def _isolated_db():
    """Config tests do not need Postgres."""
    yield


def test_profit_config_compose():
    os.environ.pop(PROFIT_PROFILE_ENV, None)
    p = reload_profit_config()
    assert isinstance(p, ProfitConfig)
    assert p.prompt is not None
    assert p.tools.tools
    assert p.memory.wm_policy
    assert p.loop.react_max_consecutive_tool_failures > 0
    assert p.risk.max_daily_llm_cost > 0


def test_summary_prints(capsys):
    os.environ.pop(PROFIT_PROFILE_ENV, None)
    reload_profit_config().summary()
    out = capsys.readouterr().out
    assert "ProfitConfig" in out
    assert "P&L:" in out or "-> P&L:" in out
    assert "Risk & execution" in out


def test_singleton_after_reload():
    os.environ.pop(PROFIT_PROFILE_ENV, None)
    reload_profit_config()
    assert get_profit_config() is get_profit_config()


def test_normalize_profile_rejects_unknown():
    try:
        normalize_profit_profile("yolo")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_conservative_profile_tightens_llm_cap():
    os.environ[PROFIT_PROFILE_ENV] = "conservative"
    p = reload_profit_config()
    assert p.risk.max_daily_llm_cost == 3.0
    assert p.risk.multi_agent_research_enabled is False
    assert p.memory.cycle_wm_max_chars == 1800
    assert p.loop.react_max_consecutive_tool_failures == 3


def test_aggressive_profile_raises_caps():
    os.environ[PROFIT_PROFILE_ENV] = "aggressive"
    p = reload_profit_config()
    assert p.risk.max_daily_llm_cost == 6.5
    assert p.memory.cycle_wm_max_chars == 3200
    assert p.loop.react_tool_feedback_max_chars == 6000


def test_optimize_for_profit_method(capsys):
    os.environ.pop(PROFIT_PROFILE_ENV, None)
    base = reload_profit_config()
    tuned = base.optimize_for_profit("conservative", verbose=True)
    out = capsys.readouterr().out
    assert "optimize_for_profit" in out
    assert tuned.risk.max_daily_llm_cost == 3.0
    assert os.environ[PROFIT_PROFILE_ENV] == "conservative"
