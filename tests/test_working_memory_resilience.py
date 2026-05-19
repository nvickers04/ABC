"""Local WM fallback + get_active_working_memory routing (no Postgres)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from quality_test_support import reset_quality_runtime_state


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture
def local_wm_file(tmp_path: Path) -> Path:
    return tmp_path / "wm.json"


@pytest.fixture(autouse=True)
def _clean_state(local_wm_file: Path):
    reset_quality_runtime_state(local_wm_file)
    yield
    reset_quality_runtime_state()


def _run(coro):
    return asyncio.run(coro)


def test_local_store_add_clear_and_expiry(local_wm_file: Path):
    from core.runtime.local_memory_fallback import get_local_working_memory

    wm = get_local_working_memory(filepath=local_wm_file)
    eid = wm.add(
        "open_theses",
        "NVDA ramp thesis",
        expires_in_minutes=30,
        metadata={"symbol": "NVDA"},
    )
    assert eid > 0
    snap = wm.snapshot()
    assert len(snap["open_theses"]) == 1
    assert snap["open_theses"][0]["metadata"]["symbol"] == "NVDA"

    removed = wm.clear("open_theses", entry_id=eid)
    assert removed == 1
    assert wm.get_all("open_theses") == []


def test_local_store_enforces_section_cap(local_wm_file: Path):
    from core.runtime.local_memory_fallback import get_local_working_memory
    from memory.working_memory import SECTION_CAPS

    wm = get_local_working_memory(filepath=local_wm_file)
    cap = SECTION_CAPS["regime_notes"]
    for i in range(cap + 3):
        wm.add("regime_notes", f"note {i}", expires_in_minutes=60)
    assert len(wm.get_all("regime_notes")) <= cap


def test_get_active_working_memory_routes_independent_mode(local_wm_file: Path):
    from core.runtime.operating_context import get_operating_context
    from core.runtime.working_memory_access import get_active_working_memory
    from core.runtime.local_memory_fallback import LocalWorkingMemoryStore

    ctx = get_operating_context()
    ctx.set_researcher_unavailable()
    wm = get_active_working_memory()
    assert isinstance(wm, LocalWorkingMemoryStore)


def test_update_and_clear_tools_use_local_in_independent_mode(local_wm_file: Path):
    from core.runtime.operating_context import get_operating_context
    from tools.tools_working_memory import (
        handle_update_working_memory,
        handle_clear_working_memory_entry,
    )

    ctx = get_operating_context()
    ctx.set_researcher_unavailable()

    add = _run(handle_update_working_memory(
        None,
        {"section": "lessons_today", "entry": "wait for confirmation"},
    ))
    assert add["success"] is True
    eid = add["entry_id"]

    cleared = _run(handle_clear_working_memory_entry(
        None,
        {"section": "lessons_today", "entry_id": eid},
    ))
    assert cleared["success"] is True
    assert cleared["removed"] == 1


def test_calculate_size_advises_host_scaling_without_shrinking_shares(monkeypatch, local_wm_file: Path):
    """Sizing must not multiply shares; host scales at order execution."""
    from core.runtime.operating_context import get_operating_context
    from tools.tools_sizing import handle_calculate_size

    ctx = get_operating_context()
    ctx.set_researcher_unavailable()

    class _Quote:
        last = 100.0
        bid = 99.5
        ask = 100.5
        volume = 1_000_000

    class _Atr:
        value = 2.0

    class _Data:
        def get_quote(self, symbol):
            return _Quote()

        def get_atr(self, symbol):
            return _Atr()

    class _Gateway:
        net_liquidation = 100_000.0
        cash_value = 50_000.0

        def get_cached_portfolio(self):
            return []

    executor = type("E", (), {
        "cash_only": True,
        "gateway": _Gateway(),
        "data_provider": _Data(),
    })()

  # Force low risk posture
    monkeypatch.setattr(
        "tools.tools_sizing.get_operating_context",
        lambda: ctx,
    )

    res = _run(handle_calculate_size(
        executor,
        {"symbol": "AAPL", "side": "BUY", "stop_distance_pct": 5.0},
    ))
    assert "error" not in res
    recommended = int(res["recommended_quantity"])
    risk_shares = int(res["breakdown"]["risk_shares"])
    # Risk lane uses full 1.5% / 5% stop math — not pre-scaled by QualityMatrix (×0.4 would be ~120).
    assert risk_shares >= 250
    assert recommended <= risk_shares

    reasoning = " ".join(res.get("reasoning", []))
    assert "Host policy" in reasoning
