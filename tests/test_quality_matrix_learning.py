"""QualityMatrix historical learning (bounded weight drift)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    """Uses in-file recent_outcomes ring buffer; no Postgres required."""
    yield


@pytest.fixture(autouse=True)
def _clean_learning(tmp_path, monkeypatch):
    from core.quality.quality_learning import reset_learning_for_tests

    monkeypatch.setattr("core.quality.quality_learning._STATE_PATH", tmp_path / "learned.json")
    reset_learning_for_tests()
    yield
    reset_learning_for_tests()


def _enable_learning(monkeypatch):
    from core.central_profit_config import get_profit_config
    from core.memory_config import reload_memory_config

    monkeypatch.setenv("QUALITY_MATRIX_LEARN_FROM_HISTORY", "1")
    reload_memory_config()
    get_profit_config().reload(dotenv=False)


def _disable_learning(monkeypatch):
    from core.central_profit_config import get_profit_config
    from core.memory_config import reload_memory_config

    monkeypatch.delenv("QUALITY_MATRIX_LEARN_FROM_HISTORY", raising=False)
    reload_memory_config()
    get_profit_config().reload(dotenv=False)


def test_risk_guard_blocks_riskier_than_profile():
    from core.quality.quality_learning import (
        clamp_not_riskier_than_profile,
        guard_learned_weights_against_profile,
    )

    assert clamp_not_riskier_than_profile("symbol_exec_quality_base", 0.75, 0.80) == 0.75
    assert clamp_not_riskier_than_profile("symbol_exec_quality_gap_coeff", 25.0, 20.0) == 25.0
    assert clamp_not_riskier_than_profile("global_exec_degraded_threshold", 0.35, 0.30) == 0.35

    safe, notes = guard_learned_weights_against_profile(
        {"symbol_exec_quality_base": 0.75, "symbol_exec_quality_gap_coeff": 25.0},
        {"symbol_exec_quality_base": 0.80, "symbol_exec_quality_gap_coeff": 22.0},
    )
    assert safe["symbol_exec_quality_base"] == 0.75
    assert safe["symbol_exec_quality_gap_coeff"] == 25.0
    assert len(notes) == 2


def test_refit_positive_reward_does_not_raise_riskier_weights(monkeypatch):
    from core.loop_config import install_loop_config, reload_loop_config
    from core.profit_config_state import set_active_profile_label
    from core.quality.quality_learning import (
        _RISKIER_IF_HIGHER,
        record_trade_outcome_and_maybe_refit,
        refit_weights_from_history,
        sync_bases_for_current_profile,
    )

    _enable_learning(monkeypatch)
    reload_loop_config()
    install_loop_config(None)
    set_active_profile_label("balanced")
    state = sync_bases_for_current_profile(force=True)

    for _ in range(30):
        record_trade_outcome_and_maybe_refit(
            {
                "symbol": "NVDA",
                "won": True,
                "realized_rr": 2.0,
                "profit_profile": "balanced",
                "pnl_usd": 200.0,
                "source": "test",
            }
        )

    summary = refit_weights_from_history()
    assert summary.get("refitted") is True
    learned = summary["learned_weights"]
    for key, base in state.base_weights.items():
        val = learned[key]
        if _RISKIER_IF_HIGHER.get(key, True):
            assert val <= base + 1e-6, f"{key} should not exceed profile base"
        else:
            assert val >= base - 1e-6, f"{key} should not go below profile base (riskier)"


def test_clamp_learned_value_respects_15pct():
    from core.quality.quality_learning import clamp_learned_value

    base = 100.0
    assert clamp_learned_value(base, 120.0) == pytest.approx(115.0)
    assert clamp_learned_value(base, 80.0) == pytest.approx(85.0)


def test_refit_moves_weights_within_bounds(monkeypatch):
    from core.loop_config import install_loop_config, reload_loop_config
    from core.profit_config_state import set_active_profile_label
    from core.quality.quality_learning import (
        capture_base_weights,
        record_trade_outcome_and_maybe_refit,
        refit_weights_from_history,
        sync_bases_for_current_profile,
    )

    _enable_learning(monkeypatch)
    reload_loop_config()
    install_loop_config(None)
    set_active_profile_label("balanced")

    bases = capture_base_weights("balanced")
    sync_bases_for_current_profile(force=True)

    for _ in range(30):
        record_trade_outcome_and_maybe_refit(
            {
                "symbol": "NVDA",
                "won": True,
                "realized_rr": 1.5,
                "profit_profile": "balanced",
                "pnl_usd": 100.0,
                "source": "test",
            }
        )

    summary = refit_weights_from_history()
    assert summary.get("refitted") is True
    learned = summary["learned_weights"]
    for key, base in bases.items():
        lo = base * 0.85
        hi = base * 1.15
        assert lo - 1e-6 <= learned[key] <= hi + 1e-6


def test_memory_config_env_enables_learning(monkeypatch):
    from core.memory_config import reload_memory_config

    monkeypatch.setenv("QUALITY_MATRIX_LEARN_FROM_HISTORY", "1")
    mem = reload_memory_config()
    assert mem.quality_matrix_learn_from_history is True


def test_learning_disabled_skips(monkeypatch):
    from core.quality.quality_learning import learning_enabled, record_trade_outcome_and_maybe_refit

    _disable_learning(monkeypatch)
    out = record_trade_outcome_and_maybe_refit({"symbol": "SPY", "won": True})
    assert out.get("skipped") is True
    assert learning_enabled() is False


def test_trade_outcomes_from_cycle_logs():
    from core.quality.quality_learning import trade_outcomes_from_cycle_logs

    entries = [
        {
            "ts": "2026-05-01T10:00:00+00:00",
            "session_date": "2026-05-01",
            "profit_profile": "balanced",
            "pnl": {"cycle_realized_pnl_usd": 120.0},
            "trade_outcome": {"action": "close_position", "symbol": "NVDA", "order_id": "1"},
        },
        {
            "ts": "2026-05-01T11:00:00+00:00",
            "profit_profile": "balanced",
            "trade_outcome": {"action": "quality_status"},
        },
    ]
    out = trade_outcomes_from_cycle_logs(entries)
    assert len(out) == 1
    assert out[0]["symbol"] == "NVDA"
    assert out[0]["won"] is True
    assert out[0]["pnl_usd"] == 120.0


def test_persist_learned_weights_to_active_profile_builtin(tmp_path, monkeypatch):
    from core.quality.quality_learning import (
        LearnedWeightState,
        persist_learned_weights_to_active_profile,
        reset_learning_for_tests,
    )

    monkeypatch.setattr("core.quality.quality_learning._STATE_PATH", tmp_path / "learned.json")
    monkeypatch.setattr("core.profit_profiles.evolved_profiles_path", lambda: tmp_path / "evolved.json")
    reset_learning_for_tests()

    state = LearnedWeightState(
        profile_label="balanced",
        base_weights={"symbol_exec_quality_base": 100.0},
        learned_weights={"symbol_exec_quality_base": 110.0},
        last_reward=0.2,
    )
    from core.quality import quality_learning as ql

    ql.save_state(state)
    summary = persist_learned_weights_to_active_profile(profile_label="balanced")
    assert summary.get("saved") is True
    assert summary.get("profile") == "balanced_qm_learned"
    assert (tmp_path / "evolved.json").is_file()


def test_quality_matrix_service_learn_from_trade(monkeypatch):
    from core.quality.quality_matrix import get_quality_matrix_service, reset_quality_matrix_service_for_tests

    _enable_learning(monkeypatch)
    reset_quality_matrix_service_for_tests()
    svc = get_quality_matrix_service()
    out = svc.learn_from_trade(
        {"symbol": "AAPL", "won": True, "realized_rr": 2.0, "profit_profile": "balanced", "pnl_usd": 50}
    )
    assert out.get("stored") is True or out.get("refitted") is True
