"""Config validation invariants.

Locks the per-field bounds enforced by :func:`core.config.validate_config`.
We swap module-level constants in via ``monkeypatch`` so the tests don't
depend on the developer's actual ``.env``.
"""

from __future__ import annotations

import pytest

import core.config as cfg
from core.config import ConfigError, assert_config_valid, validate_config


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    """Snapshot module constants and restore via monkeypatch unwinding."""
    yield  # monkeypatch in tests is auto-rolled back


class TestValidateConfigHappyPath:
    def test_default_config_is_valid(self):
        # Whatever the project's defaults are, they must validate clean —
        # this catches accidental drift in the module-level constants.
        assert validate_config() == []


class TestRiskPerTrade:
    def test_zero_risk_rejected(self, monkeypatch):
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.0)
        errs = validate_config()
        assert any("RISK_PER_TRADE" in e for e in errs)

    def test_negative_risk_rejected(self, monkeypatch):
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", -0.01)
        assert any("RISK_PER_TRADE" in e for e in validate_config())

    def test_above_50pct_rejected(self, monkeypatch):
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.51)
        assert any("RISK_PER_TRADE" in e for e in validate_config())


class TestThresholds:
    def test_min_rr_must_be_positive(self, monkeypatch):
        monkeypatch.setattr(cfg, "MIN_RR_RATIO", 0.0)
        assert any("MIN_RR" in e for e in validate_config())

    def test_max_daily_loss_pct_bounds(self, monkeypatch):
        monkeypatch.setattr(cfg, "MAX_DAILY_LOSS_PCT", 0.0)
        assert any("MAX_DAILY_LOSS_PCT" in e for e in validate_config())
        monkeypatch.setattr(cfg, "MAX_DAILY_LOSS_PCT", 150.0)
        assert any("MAX_DAILY_LOSS_PCT" in e for e in validate_config())

    def test_intraday_drawdown_pct_bounds(self, monkeypatch):
        monkeypatch.setattr(cfg, "INTRADAY_DRAWDOWN_PCT", -1.0)
        assert any("INTRADAY_DRAWDOWN_PCT" in e for e in validate_config())

    def test_eod_flatten_minutes_must_be_positive(self, monkeypatch):
        monkeypatch.setattr(cfg, "EOD_FLATTEN_MINUTES", 0)
        assert any("EOD_FLATTEN_MINUTES" in e for e in validate_config())

    def test_open_gap_guard_pct_non_negative(self, monkeypatch):
        monkeypatch.setattr(cfg, "OPEN_GAP_GUARD_PCT", -0.5)
        assert any("OPEN_GAP_GUARD_PCT" in e for e in validate_config())

    def test_max_daily_llm_cost_must_be_positive(self, monkeypatch):
        monkeypatch.setattr(cfg, "MAX_DAILY_LLM_COST", 0.0)
        assert any("MAX_DAILY_LLM_COST" in e for e in validate_config())


class TestLLMParams:
    def test_temperature_lower_bound(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_TEMPERATURE", -0.1)
        assert any("LLM_TEMPERATURE" in e for e in validate_config())

    def test_temperature_upper_bound(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_TEMPERATURE", 2.5)
        assert any("LLM_TEMPERATURE" in e for e in validate_config())

    def test_max_tokens_must_be_positive(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_MAX_TOKENS", 0)
        assert any("LLM_MAX_TOKENS" in e for e in validate_config())


class TestLiveModeSafety:
    def test_live_with_high_risk_rejected(self, monkeypatch):
        monkeypatch.setattr(cfg, "TRADING_MODE", "live")
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.05)  # 5%
        errs = validate_config()
        assert any("live" in e and "2%" in e for e in errs)

    def test_live_with_safe_risk_passes(self, monkeypatch):
        monkeypatch.setattr(cfg, "TRADING_MODE", "live")
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.005)  # 0.5%
        # Strip any unrelated errors from default state.
        live_errs = [e for e in validate_config() if "live" in e]
        assert live_errs == []


class TestAssertConfigValid:
    def test_raises_config_error_with_messages(self, monkeypatch):
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.0)
        monkeypatch.setattr(cfg, "MIN_RR_RATIO", -1.0)
        with pytest.raises(ConfigError) as excinfo:
            assert_config_valid()
        msg = str(excinfo.value)
        assert "RISK_PER_TRADE" in msg
        assert "MIN_RR" in msg

    def test_clean_config_does_not_raise(self):
        assert_config_valid()
