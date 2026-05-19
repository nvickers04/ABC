"""ToolExecutor × QualityMatrix integration (quantity scaling, hard blocks)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from quality_test_support import make_fake_quality_db, reset_quality_runtime_state


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture(autouse=True)
def _clean_quality_state():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


def _minimal_executor():
    from tools.tools_executor import ToolExecutor

    gateway = SimpleNamespace(
        net_liquidation=100_000.0,
        get_cached_portfolio=lambda: [],
    )
    data_provider = SimpleNamespace()
    executor = ToolExecutor(gateway=gateway, data_provider=data_provider)
    executor._dispatch = AsyncMock(
        return_value={"success": True, "data": {"ok": True}},
    )
    return executor


@pytest.mark.asyncio
async def test_execute_scales_quantity_before_dispatch(monkeypatch):
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service

    monkeypatch.setattr(
        "tools.tools_executor._allowed_trade_universe",
        lambda: {"AAPL"},
    )

    ctx = get_operating_context()
    ctx.set_researcher_available()

    svc = get_quality_matrix_service()
    svc.populate(make_fake_quality_db())
    m = svc.get_matrix()
    m.overall_quality = "limited"
    m.risk_multiplier = 0.5
    m.force_conservative_reasoning = True

    executor = _minimal_executor()
    params = {"symbol": "AAPL", "side": "BUY", "quantity": 100, "intent": "entry"}

    result = await executor.execute("market_order", params)

    assert result.success is True
    assert params["quantity"] == 50
    executor._dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_hard_blocks_buy_in_minimal_quality():
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service

    ctx = get_operating_context()
    ctx.set_researcher_unavailable()
    svc = get_quality_matrix_service()
    svc.populate(make_fake_quality_db())
    assert svc.get_matrix().overall_quality == "minimal"

    executor = _minimal_executor()
    result = await executor.execute(
        "buy",
        {"symbol": "AAPL", "quantity": 10, "intent": "entry"},
    )

    assert result.success is False
    payload = result.data
    err_text = str(payload.get("error") or payload)
    assert "BLOCKED" in err_text.upper() or "QualityMatrix" in err_text
    executor._dispatch.assert_not_awaited()
