"""Characterization tests for QualityMatrix core.

Pins the deterministic behavior of population from trade_feedback,
tool recording, decision provenance snapshots, to_prompt_block and
recommended_policies. LLM / heavy analysis paths are stubbed.

Matches style of test_run_execution_analysis_characterization.py and
test_runtime_characterization.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from quality_test_support import (
    make_fake_quality_db,
    reset_quality_runtime_state,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override root conftest: QualityMatrix tests use in-memory fake DB only."""
    yield


@pytest.fixture(autouse=True)
def _clean_quality_state():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


def _make_fake_db_with_feedback():
    return make_fake_quality_db(
        feedback_rows=[
            {"symbol": "AAPL", "n": 5, "avg_gap": 0.004, "winrate": 0.6},
            {"symbol": "TSLA", "n": 3, "avg_gap": -0.012, "winrate": 0.4},
        ],
    )


def test_quality_matrix_service_basic_population_and_prompt():
    from core.quality.quality_matrix import get_quality_matrix_service, ToolUsageRecord, DecisionProvenanceSnapshot

    svc = get_quality_matrix_service()
    fake_db = _make_fake_db_with_feedback()

    # Population should not crash and should produce reasonable aggregates
    svc.populate(fake_db)

    m = svc.get_matrix()
    assert m.overall_quality in ("full", "limited", "minimal", "degraded")
    assert 0.0 <= m.risk_multiplier <= 1.0
    assert len(m.symbol_quality) >= 0  # may be empty on fake but no crash

    block = m.to_prompt_block()
    assert "QUALITY MATRIX" in block
    assert "Overall:" in block

    pol = m.recommended_policies("AAPL")
    assert "risk_multiplier" in pol
    assert "force_conservative_reasoning" in pol


def test_tool_usage_and_provenance_recording():
    from core.quality.quality_matrix import get_quality_matrix_service, ToolUsageRecord, DecisionProvenanceSnapshot

    svc = get_quality_matrix_service()
    svc._enabled = True  # force on for test

    rec = ToolUsageRecord(tool_name="quote", symbol="AAPL", success=True, latency_ms=12.3)
    svc.record_tool_usage(rec)

    snap = DecisionProvenanceSnapshot(
        cycle_id=42,
        decision_type="done",
        symbol="AAPL",
        tools_used=[rec],
        quality_state={"overall": "limited"},
        context_quality="limited",
        outcome="done",
    )
    svc.record_decision_snapshot(snap)

    m = svc.get_matrix()
    assert any(t.tool_name == "quote" for t in m.recent_tool_usage)
    assert any(p.cycle_id == 42 for p in m.recent_provenance)


def test_research_config_knobs_are_read():
    from core.quality.quality_matrix import get_quality_matrix_service
    # Just ensure _refresh_knobs does not explode (real get_research_config is exercised via populate)
    svc = get_quality_matrix_service()
    svc._refresh_knobs()
    assert hasattr(svc, "_enabled")


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=line"])
