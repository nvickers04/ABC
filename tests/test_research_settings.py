"""Research host ProfitConfig synchronization."""

from __future__ import annotations

import os

import pytest

from core.entry_cli import apply_profit_profile_cli_to_environ, parse_research_args

from core.central_profit_config import (
    get_research_settings,
    reload_profit_config,
    reset_profit_config_for_tests,
    sync_research_host_from_profit_config,
)
from core.profit_profiles import PROFIT_PROFILE_ENV
from core.research_topics import invalidate_research_topic_caches


@pytest.fixture(autouse=True)
def _reset_profit_config():
    reset_profit_config_for_tests()
    yield
    reset_profit_config_for_tests()


def test_research_cli_profile_flag_sets_env(monkeypatch):
    monkeypatch.delenv(PROFIT_PROFILE_ENV, raising=False)
    args = parse_research_args(["--profile", "aggressive"])
    apply_profit_profile_cli_to_environ(args)
    assert os.environ[PROFIT_PROFILE_ENV] == "aggressive"

    args2 = parse_research_args(["--profit-profile", "conservative"])
    apply_profit_profile_cli_to_environ(args2)
    assert os.environ[PROFIT_PROFILE_ENV] == "conservative"


def test_research_settings_follow_profit_profile(monkeypatch):
    monkeypatch.setenv(PROFIT_PROFILE_ENV, "aggressive")
    reload_profit_config()
    rs = get_research_settings()
    assert rs.profile_label == "aggressive"
    assert rs.deep_scan_top_n == 12
    assert rs.tier1_universe_size == 30

    monkeypatch.setenv(PROFIT_PROFILE_ENV, "conservative")
    reload_profit_config()
    rs2 = get_research_settings()
    assert rs2.profile_label == "conservative"
    assert rs2.deep_scan_top_n == 8
    assert rs2.combiner_ir_gate_min == 0.06


def test_sync_publishes_profile_to_research_config(monkeypatch):
    stored: dict[str, object] = {}

    def _set(key, val, **kwargs):
        stored[key] = val

    monkeypatch.setenv(PROFIT_PROFILE_ENV, "balanced")
    reload_profit_config()
    monkeypatch.setattr("memory.set_research_config", _set)

    sync_research_host_from_profit_config()
    from core.research_settings import RESEARCH_HOST_PROFILE_KEY

    assert stored[RESEARCH_HOST_PROFILE_KEY] == "balanced"


def test_invalidate_research_topic_caches():
    from core import research_topics as rt

    rt._TICKER_SYMBOLS.add("TEST")
    invalidate_research_topic_caches()
    assert rt._TICKER_SYMBOLS == set()
