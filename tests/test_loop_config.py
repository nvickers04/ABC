"""LoopConfig singleton and gate validation."""

from __future__ import annotations

from core.loop_config import LoopConfig, get_loop_config, reload_loop_config


def test_loop_config_singleton():
    assert get_loop_config() is get_loop_config()


def test_clamp_cooldown():
    lc = LoopConfig()
    assert lc.clamp_cooldown(1) == lc.react_cooldown_min_seconds
    assert lc.clamp_cooldown(99999) == lc.react_cooldown_max_seconds
    assert lc.clamp_cooldown(30) == 30


def test_scheduler_defaults_match_fields():
    lc = LoopConfig()
    d = lc.scheduler_defaults()
    assert d["halted_poll_seconds"] == lc.scheduler_halted_poll_seconds


def test_gate_validation_rejects_bad_thresholds():
    try:
        LoopConfig(symbol_exec_quality_min=0.9, symbol_exec_quality_max=0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass
