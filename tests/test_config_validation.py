"""Config validation invariants (Pydantic + module-level snapshot)."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

import core.config as cfg
from core.config import ConfigError, assert_config_valid, validate_config
from core.risk_execution_config import RiskExecutionConfig
from core.settings import RuntimeConfigSnapshot


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override tests/conftest autouse — no Postgres required."""
    yield


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    """Snapshot module constants and restore via monkeypatch unwinding."""
    yield


class TestValidateConfigHappyPath:
    def test_default_config_is_valid(self):
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
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.05)
        monkeypatch.setenv("TRADING_MODE", "live")
        monkeypatch.setenv("IBKR_ACCOUNT_TYPE", "live")
        errs = validate_config()
        assert any("live" in e and "2%" in e for e in errs)

    def test_live_with_safe_risk_passes(self, monkeypatch):
        monkeypatch.setattr(cfg, "TRADING_MODE", "live")
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.005)
        monkeypatch.setenv("TRADING_MODE", "live")
        monkeypatch.setenv("IBKR_ACCOUNT_TYPE", "live")
        live_errs = [e for e in validate_config() if "live" in e.lower()]
        assert live_errs == []


class TestTradingModeCombinations:
    def test_live_mode_requires_live_ibkr_account(self, monkeypatch):
        monkeypatch.setattr(cfg, "TRADING_MODE", "live")
        monkeypatch.setattr(cfg, "RISK_PER_TRADE", 0.005)
        monkeypatch.setenv("TRADING_MODE", "live")
        monkeypatch.setenv("IBKR_ACCOUNT_TYPE", "paper")
        errs = validate_config()
        assert any("IBKR_ACCOUNT_TYPE" in e for e in errs)

    def test_aggressive_paper_rejects_live_ibkr(self, monkeypatch):
        monkeypatch.setattr(cfg, "TRADING_MODE", "aggressive_paper")
        monkeypatch.setenv("TRADING_MODE", "aggressive_paper")
        monkeypatch.setenv("IBKR_ACCOUNT_TYPE", "live")
        errs = validate_config()
        assert any("aggressive_paper" in e for e in errs)

    def test_invalid_trading_mode_rejected(self, monkeypatch):
        monkeypatch.setattr(cfg, "TRADING_MODE", "yolo")
        monkeypatch.setenv("TRADING_MODE", "yolo")
        assert any("TRADING_MODE" in e for e in validate_config())


class TestDatabaseUrl:
    def test_invalid_scheme_rejected(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "mysql://localhost/db")
        errs = validate_config()
        assert any("DATABASE_URL" in e or "postgresql" in e for e in errs)

    def test_valid_postgres_url_accepted(self, monkeypatch):
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@localhost:5432/abc",
        )
        with pytest.raises(ValidationError):
            RuntimeConfigSnapshot(
                trading_mode="paper",
                ibkr_account_type="paper",
                risk_per_trade=0.01,
                min_rr_ratio=2.0,
                max_daily_loss_pct=15.0,
                intraday_drawdown_pct=3.0,
                eod_flatten_minutes=5,
                open_gap_guard_pct=2.0,
                open_guard_delay_minutes=15,
                max_daily_llm_cost=4.5,
                max_daily_multi_agent_research_usd=0.75,
                cycle_sleep_seconds=30,
                llm_temperature=0.0,
                llm_max_tokens=8192,
                tool_playbook_max_chars=1200,
                agent_tool_feedback_max_chars=4500,
                max_daily_llm_noncached_prompt_text_tokens=1,
                max_daily_llm_cached_prompt_text_tokens=1,
                max_daily_llm_prompt_image_tokens=1,
                max_daily_llm_completion_tokens=1,
                max_daily_llm_reasoning_tokens=1,
                max_daily_llm_output_priced_tokens=1,
                database_url="mysql://localhost/db",
            )


class TestRiskExecutionConfig:
    def test_mode_defaults_apply_risk(self, monkeypatch):
        from core.risk_execution_config import RiskExecutionConfig, get_risk_execution_config

        monkeypatch.delenv("RISK_PER_TRADE", raising=False)
        monkeypatch.delenv("MIN_RR", raising=False)
        monkeypatch.setenv("TRADING_MODE", "live")
        monkeypatch.setenv("IBKR_ACCOUNT_TYPE", "live")
        get_risk_execution_config.cache_clear()
        s = RiskExecutionConfig(_env_file=None)
        assert s.risk_per_trade_pct == 0.5
        assert s.min_rr_ratio == 2.5


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
