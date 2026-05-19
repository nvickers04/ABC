"""Runtime cycle context assembly (mocked DB attention/intuition)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from quality_test_support import make_fake_quality_db, reset_quality_runtime_state


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture
def local_wm(tmp_path: Path) -> Path:
    return tmp_path / "wm.json"


@pytest.fixture(autouse=True)
def _clean(local_wm: Path):
    reset_quality_runtime_state(local_wm)
    yield
    reset_quality_runtime_state()


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.asyncio
async def test_build_cycle_user_context_includes_quality_matrix(monkeypatch, local_wm):
    from core.runtime.cycle_context import build_cycle_user_context
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service

    monkeypatch.setattr(
        "core.runtime.cycle_context.load_attention_block",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "core.runtime.cycle_context.load_intuition_block",
        AsyncMock(return_value=""),
    )

    ctx = get_operating_context()
    ctx.set_researcher_available()
    svc = get_quality_matrix_service()
    svc.populate(make_fake_quality_db())

    context, metrics = await build_cycle_user_context(
        operating_context=ctx,
        state_text="═══ MARKET: REGULAR ═══\n",
        cost_line="",
        continuity_text="",
        pre_scan_prompt="",
        gap_guard_prompt="",
        et_now=datetime.now(ZoneInfo("America/New_York")),
    )

    assert "QUALITY MATRIX" in context
    assert metrics.quality_chars > 0
    assert metrics.state_chars > 0
    assert "full" in context.lower() or "MINIMAL" in context or "limited" in context.lower()


@pytest.mark.asyncio
async def test_compact_wm_block_uses_local_store(local_wm):
    from core.runtime.cycle_context import compact_wm_block
    from core.runtime.local_memory_fallback import get_local_working_memory

    wm = get_local_working_memory(filepath=local_wm)
    wm.add("lessons_today", "test lesson", expires_in_minutes=60)
    text = compact_wm_block(wm)
    assert "lessons" in text.lower() or "WORKING MEMORY" in text


def test_count_live_wm_entries_local(local_wm):
    from core.runtime.local_memory_fallback import get_local_working_memory
    from core.runtime.working_memory_access import count_live_wm_entries

    wm = get_local_working_memory(filepath=local_wm)
    wm.add("open_theses", "live thesis", expires_in_minutes=60)
    counts = count_live_wm_entries(wm)
    assert counts["total"] >= 1
    assert counts["by_section"].get("open_theses", 0) >= 1


def test_postgres_failure_routes_to_local_and_marks_independent(local_wm, monkeypatch):
    from core.runtime.operating_context import get_operating_context
    from core.runtime.local_memory_fallback import LocalWorkingMemoryStore
    from core.runtime.working_memory_access import get_active_working_memory

    ctx = get_operating_context()
    ctx.set_researcher_available()

    def _boom():
        raise RuntimeError("postgres down")

    monkeypatch.setattr(
        "memory.working_memory.get_working_memory",
        _boom,
    )

    wm = get_active_working_memory()
    assert isinstance(wm, LocalWorkingMemoryStore)
    assert ctx.is_independent_mode is True
