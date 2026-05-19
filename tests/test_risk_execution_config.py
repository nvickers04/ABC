"""RiskExecutionConfig singleton and IBKR port resolution."""

from __future__ import annotations

from core.risk_execution_config import (
    MODE_DEFAULTS,
    RiskExecutionConfig,
    get_risk_execution_config,
    reload_risk_execution_config,
)


def test_singleton():
    assert get_risk_execution_config() is get_risk_execution_config()


def test_mode_defaults_table():
    assert MODE_DEFAULTS["paper"]["risk"] == 1.0


def test_resolve_ibkr_port_live(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("IBKR_PORT", raising=False)
    r = reload_risk_execution_config()
    assert r.resolve_ibkr_port() == r.ibkr_live_port


def test_cash_only_default_true():
    r = RiskExecutionConfig()
    assert r.cash_only is True
