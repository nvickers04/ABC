"""Structured logging context (no Postgres)."""

from __future__ import annotations

import pytest
import structlog

from core.log_context import (
    bind_log_context,
    bind_trade_context,
    bind_trader_cycle_context,
    clear_log_context,
    configure_structlog,
    get_logger,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture(autouse=True)
def _reset_context():
    clear_log_context()
    yield
    clear_log_context()


def _ctx() -> dict:
    return structlog.contextvars.get_contextvars()


def test_bind_trader_cycle_context():
    bind_trader_cycle_context(cycle_id=7)
    ctx = _ctx()
    assert ctx.get("cycle_id") == 7
    assert ctx.get("quality_score")
    # Omitted when Postgres/heartbeat is unreachable (None filtered).
    assert "research_heartbeat" not in ctx or ctx["research_heartbeat"] is not None


def test_bind_trade_context():
    bind_trade_context(trade_id="ord-99", order_id=99)
    assert _ctx().get("trade_id") == "ord-99"
    assert _ctx().get("order_id") == 99


def test_clear_log_context():
    bind_log_context(cycle_id=1)
    clear_log_context()
    bind_log_context(cycle_id=2)
    assert _ctx().get("cycle_id") == 2


def test_get_logger_after_configure():
    configure_structlog(verbose=False)
    log = get_logger("test.module")
    log.info("smoke_event", ok=True)
