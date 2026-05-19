"""QualityMatrix provenance and compact prompt blocks (no Postgres)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quality_test_support import make_fake_quality_db, reset_quality_runtime_state


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture(autouse=True)
def _clean():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


def test_record_tool_usage_and_snapshot_decision():
    from core.quality.quality_matrix import (
        DecisionProvenanceSnapshot,
        QualityMatrixService,
        ToolUsageRecord,
        get_quality_matrix_service,
    )

    svc = get_quality_matrix_service()
    m = svc.get_matrix()
    rec = ToolUsageRecord(
        tool_name="quote",
        called_at=datetime.now(timezone.utc),
        symbol="AAPL",
        success=True,
    )
    svc.record_tool_usage(rec)
    assert len(m.recent_tool_usage) == 1

    snap = DecisionProvenanceSnapshot(
        ts=datetime.now(timezone.utc),
        cycle_id=3,
        decision_type="cycle_decision",
        symbol="AAPL",
        tools_used=[rec],
        quality_state={"overall": "paper"},
        context_quality="full",
        outcome="done",
        notes="hold",
    )
    svc.record_decision_snapshot(snap)
    assert len(m.recent_provenance) == 1


def test_service_populate_from_fake_db():
    from core.quality.quality_matrix import get_quality_matrix_service

    svc = get_quality_matrix_service()
    db = make_fake_quality_db(
        feedback_rows=[
            {"symbol": "NVDA", "n": 5, "avg_gap": -0.01, "winrate": 0.6},
        ],
        tool_log=[
            {
                "tool_name": "briefing",
                "symbol": None,
                "success": 1,
                "called_at": datetime.now(timezone.utc).isoformat(),
            },
        ],
    )
    svc.populate(db)
    m = svc.get_matrix()
    assert m.symbol_quality or m.global_execution_quality >= 0


def test_compact_prompt_preserves_blocked_categories():
    from core.quality.quality_matrix import QualityMatrix

    m = QualityMatrix(
        overall_quality="minimal",
        risk_multiplier=0.4,
        blocked_tool_categories=["research", "entries"],
        force_conservative_reasoning=True,
    )
    block = m.to_prompt_block(compact=True)
    assert "research" in block
    assert "entries" in block
    assert "×0.40" in block
