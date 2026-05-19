"""Profile rollback guard (live mode trial drawdown)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture
def rollback_paths(tmp_path, monkeypatch):
    state = tmp_path / "profile_rollback_state.json"
    events = tmp_path / "events.jsonl"
    monkeypatch.setattr("core.profile_rollback._STATE_PATH", state)
    monkeypatch.setattr("core.profile_rollback._EVENTS_PATH", events)
    return state, events


def test_rollback_triggers_at_10pct_drawdown(rollback_paths, monkeypatch):
    from core.profile_rollback import (
        ProfileRollbackState,
        check_profile_rollback_live,
        load_state,
        save_state,
    )

    monkeypatch.setenv("ABC_PROFILE_ROLLBACK_FORCE", "1")
    monkeypatch.setenv("PROFIT_PROFILE", "aggressive")
    monkeypatch.setenv("ABC_PROFILE_ROLLBACK_DRAWDOWN_PCT", "10")

    save_state(
        ProfileRollbackState(
            known_good_profile="balanced",
            trial_profile="aggressive",
            trial_peak_nlv=100_000.0,
            trial_started_at="2026-01-01T00:00:00+00:00",
            last_applied_profile="aggressive",
        )
    )

    gw = MagicMock()
    gw.net_liquidation = 89_000.0  # 11% drawdown from peak

    with patch("core.central_profit_config.get_profit_config") as gp:
        gp.return_value.reload.return_value = MagicMock()
        with patch("core.central_profit_config.sync_research_host_from_profit_config"):
            event = check_profile_rollback_live(gw)

    assert event is not None
    assert event["known_good_profile"] == "balanced"
    assert event["trial_profile"] == "aggressive"
    assert event["drawdown_pct"] >= 10.0
    assert os.environ.get("PROFIT_PROFILE") == "balanced"

    st = load_state()
    assert st.last_applied_profile == "balanced"
    assert st.rollback_count == 1


def test_no_rollback_when_on_known_good(rollback_paths, monkeypatch):
    from core.profile_rollback import ProfileRollbackState, check_profile_rollback_live, save_state

    monkeypatch.setenv("ABC_PROFILE_ROLLBACK_FORCE", "1")
    monkeypatch.setenv("PROFIT_PROFILE", "balanced")

    save_state(
        ProfileRollbackState(
            known_good_profile="balanced",
            last_applied_profile="balanced",
        )
    )
    gw = MagicMock()
    gw.net_liquidation = 50_000.0
    assert check_profile_rollback_live(gw) is None


def test_on_profile_applied_sets_known_good(rollback_paths, monkeypatch):
    from core.profile_rollback import load_state, on_profile_applied

    monkeypatch.setenv("ABC_PROFILE_ROLLBACK_FORCE", "1")
    on_profile_applied("balanced", "aggressive")
    st = load_state()
    assert st.known_good_profile == "balanced"
    assert st.trial_profile == "aggressive"
    assert st.last_applied_profile == "aggressive"
