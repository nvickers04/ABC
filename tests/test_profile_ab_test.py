"""ProfitConfig live A/B test helpers."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_parse_ab_profiles_requires_two():
    from core.profile_ab_test import parse_ab_profiles

    assert parse_ab_profiles("conservative,balanced") == ("conservative", "balanced")
    with pytest.raises(ValueError, match="exactly two"):
        parse_ab_profiles("balanced")
    with pytest.raises(ValueError, match="distinct"):
        parse_ab_profiles("balanced,balanced")


def test_rotation_alternates():
    from core.profile_ab_test import AbTestState, select_profile_for_cycle

    state = AbTestState(
        profiles=("conservative", "balanced"),
        mode="rotation",
        started_at="2026-05-01T00:00:00+00:00",
    )
    state.cycle_index = 0
    assert select_profile_for_cycle(state) == "conservative"
    state.cycle_index = 1
    assert select_profile_for_cycle(state) == "balanced"


def test_risk_guard_blocks_live_account():
    from core.profile_ab_test import validate_ab_test_startup

    with pytest.raises(SystemExit):
        validate_ab_test_startup(account="live", trading_mode="paper")


def test_build_daily_winner_report():
    from core.profile_ab_test import AbTestState, build_daily_winner_report

    state = AbTestState(
        profiles=("conservative", "balanced"),
        run_id="testrun",
        started_at="2026-05-01T00:00:00+00:00",
    )
    entries = [
        {
            "ts": "2026-05-19T10:00:00+00:00",
            "session_date": "2026-05-19",
            "profit_profile": "conservative",
            "pnl": {"cycle_realized_pnl_usd": 50.0},
            "trade_outcome": {"action": "close_position", "order_id": "1"},
        },
        {
            "ts": "2026-05-19T11:00:00+00:00",
            "session_date": "2026-05-19",
            "profit_profile": "balanced",
            "pnl": {"cycle_realized_pnl_usd": -10.0},
            "trade_outcome": {"action": "close_position", "order_id": "2"},
        },
    ]

    def _fake_entries(_state):
        return entries

    import core.profile_ab_test as ab

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ab, "_entries_for_ab_run", _fake_entries)
    try:
        report = build_daily_winner_report(state, session_date="2026-05-19")
    finally:
        monkeypatch.undo()

    assert report["winner"] in ("conservative", "balanced")
    assert report["cycles_today"] == 2


def test_ab_meta_on_activate(monkeypatch, tmp_path):
    from core.central_profit_config import get_profit_config
    from core.memory_config import reload_memory_config
    from core.profile_ab_test import AbTestState, activate_ab_profile, get_ab_log_meta

    monkeypatch.setattr("core.profile_ab_test._STATE_PATH", tmp_path / "ab.json")
    reload_memory_config()
    get_profit_config().reload(dotenv=False)
    state = AbTestState(profiles=("conservative", "balanced"), run_id="r1")
    activate_ab_profile("conservative", state=state)
    meta = get_ab_log_meta()
    assert meta is not None
    assert meta["arm"] == "conservative"
    assert meta["profiles"] == ["conservative", "balanced"]
