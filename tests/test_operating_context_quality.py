"""Operating context + cycle guidance + heartbeat sync (no Postgres)."""

from __future__ import annotations

import pytest

from quality_test_support import reset_quality_runtime_state


@pytest.fixture(autouse=True)
def _isolated_db():
  """Override root conftest — these tests do not touch Postgres."""
  yield


@pytest.fixture(autouse=True)
def _clean_quality_state():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


def test_cycle_guidance_footer_independent_vs_connected():
    from core.runtime.operating_context import get_operating_context

    ctx = get_operating_context()
    ctx.set_researcher_available()
    connected = ctx.cycle_guidance_footer()
    assert "briefing()" in connected
    assert "INDEPENDENT MODE" not in connected

    ctx.set_researcher_unavailable()
    independent = ctx.cycle_guidance_footer()
    assert "INDEPENDENT MODE" in independent
    assert "quality_status()" in independent
    assert "briefing()" not in independent


def test_legacy_risk_multiplier_ignores_quality_matrix():
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service, reset_quality_matrix_service_for_tests
    from quality_test_support import make_fake_quality_db

    reset_quality_matrix_service_for_tests()
    ctx = get_operating_context()
    ctx.set_researcher_unavailable()
    assert ctx.legacy_risk_multiplier == 0.4

    svc = get_quality_matrix_service()
    svc.get_matrix().risk_multiplier = 0.99  # would pollute legacy if coupled
    assert ctx.legacy_risk_multiplier == 0.4


def test_risk_multiplier_is_conservative_blend():
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service

    ctx = get_operating_context()
    ctx.set_researcher_unavailable()
    svc = get_quality_matrix_service()
    svc.get_matrix().risk_multiplier = 0.9
    assert ctx.risk_multiplier <= 0.65


def test_sync_researcher_from_heartbeat(monkeypatch):
    from core.runtime.operating_context import get_operating_context

    ctx = get_operating_context()
    monkeypatch.setattr(
        "core.runtime.heartbeat.is_daemon_alive",
        lambda: True,
    )
    assert ctx.sync_researcher_from_heartbeat() is True
    assert ctx.quality.researcher_available is True
    assert ctx.quality.memory_source == "postgres"
    assert ctx.is_independent_mode is False

    monkeypatch.setattr(
        "core.runtime.heartbeat.is_daemon_alive",
        lambda: False,
    )
    assert ctx.sync_researcher_from_heartbeat() is False
    assert ctx.is_independent_mode is True


def test_limited_quality_when_local_wm_adequate():
    from core.runtime.operating_context import get_operating_context

    ctx = get_operating_context()
    ctx.quality.researcher_available = False
    ctx.quality.memory_source = "local_fallback"
    ctx.quality.working_memory_completeness = 0.5
    ctx._recalculate_overall_quality()
    assert ctx.quality.overall_quality == "limited"
    assert ctx.legacy_risk_multiplier == 0.65


def test_wm_recovery_summary_counts_local_entries(tmp_path, caplog):
    from pathlib import Path

    import logging

    from core.runtime.operating_context import get_operating_context
    from core.runtime.local_memory_fallback import get_local_working_memory
    from core.runtime.working_memory_access import log_wm_recovery_on_reconnect
    from quality_test_support import reset_quality_runtime_state

    local_wm_file = tmp_path / "wm.json"
    reset_quality_runtime_state(local_wm_file)
    wm = get_local_working_memory(filepath=Path(local_wm_file))
    wm.add("open_theses", "orphan thesis from outage", expires_in_minutes=120)

    caplog.set_level(logging.INFO)
    summary = log_wm_recovery_on_reconnect(had_local_fallback=True)

    assert summary["local"]["total"] >= 1
    assert summary["policy"] == "postgres_wins_no_merge"
    assert any("WM recovery" in r.message for r in caplog.records)
    assert "postgres_wins_no_merge" in caplog.text

    ctx = get_operating_context()
    ctx.set_researcher_unavailable()
    ctx.set_researcher_available()
    assert ctx.quality.memory_source == "postgres"
