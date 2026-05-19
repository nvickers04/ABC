"""QualityMatrix policy enforcement: populate, gates, scaling, throttling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quality_test_support import (
    make_fake_quality_db,
    reset_quality_runtime_state,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture(autouse=True)
def _clean_quality_state():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


def test_populate_uses_legacy_risk_not_matrix_feedback():
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service

    ctx = get_operating_context()
    ctx.set_researcher_unavailable()

    svc = get_quality_matrix_service()
    svc.get_matrix().risk_multiplier = 1.0  # stale high value
    db = make_fake_quality_db(
        feedback_rows=[
            {"symbol": "AAPL", "n": 10, "avg_gap": 0.001, "winrate": 0.7},
        ],
    )
    svc.populate(db)
    m = svc.get_matrix()
    assert m.overall_quality == "minimal"
    assert m.risk_multiplier <= 0.4
    assert "research" in m.blocked_tool_categories


def test_populate_full_quality_when_researcher_connected():
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service

    ctx = get_operating_context()
    ctx.set_researcher_available()

    svc = get_quality_matrix_service()
    db = make_fake_quality_db(
        feedback_rows=[
            {"symbol": "AAPL", "n": 10, "avg_gap": 0.001, "winrate": 0.7},
        ],
    )
    svc.populate(db)
    m = svc.get_matrix()
    assert m.overall_quality == "full"
    assert m.risk_multiplier == 1.0
    assert m.blocked_tool_categories == []


def test_maybe_populate_skips_when_fresh():
    from core.quality.quality_matrix import get_quality_matrix_service

    svc = get_quality_matrix_service()
    svc.get_matrix().last_populated = datetime.now(timezone.utc)
    calls: list[str] = []

    def _spy_populate(db=None):
        calls.append("populate")

    svc.populate = _spy_populate  # type: ignore[method-assign]
    svc.maybe_populate(max_age_seconds=60.0)
    assert calls == []

    svc.get_matrix().last_populated = datetime.now(timezone.utc) - timedelta(seconds=120)
    svc.maybe_populate(max_age_seconds=60.0)
    assert calls == ["populate"]


def test_should_allow_tool_blocks_entries_in_minimal_mode():
    from core.quality.quality_matrix import QualityMatrix

    m = QualityMatrix(overall_quality="minimal", risk_multiplier=0.4)
    allowed, reason = m.should_allow_tool("buy", {"symbol": "AAPL", "intent": "entry"})
    assert allowed is False
    assert reason
    assert "HARD BLOCKED" in reason

    ok, _ = m.should_allow_tool("quote", {"symbol": "AAPL"})
    assert ok is True


def test_should_allow_tool_blocks_research_category():
    from core.quality.quality_matrix import QualityMatrix

    m = QualityMatrix(
        overall_quality="minimal",
        blocked_tool_categories=["research"],
    )
    allowed, reason = m.should_allow_tool("research", {})
    assert allowed is False
    assert "research" in (reason or "")


def test_get_scaled_quantity_single_application():
    from core.quality.quality_matrix import QualityMatrix

    m = QualityMatrix(overall_quality="limited", risk_multiplier=0.5)
    assert m.get_scaled_quantity(100, intent="entry") == 50
    assert m.get_scaled_quantity(1, intent="entry") >= 1


def test_get_llm_call_config_tightens_under_degraded():
    from core.quality.quality_matrix import QualityMatrix

    m = QualityMatrix(
        overall_quality="minimal",
        suggested_temperature=0.3,
        suggested_max_tokens=8192,
        force_conservative_reasoning=True,
    )
    cfg = m.get_llm_call_config()
    assert cfg["temperature"] <= 0.18
    assert cfg["max_tokens"] <= 5500
    assert cfg["reasoning_bias"] == "conservative"
