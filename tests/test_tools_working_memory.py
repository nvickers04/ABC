"""Tests for tools.tools_working_memory — agent's write tools for working memory."""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import memory
    from memory import working_memory as wm_mod
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "_DB_PATH", db_path)
    monkeypatch.setattr(memory, "_connection", None)
    monkeypatch.setattr(memory, "_calibration_version", 0)
    memory._pending_graduated_params.clear()
    memory._pending_order_context.clear()
    wm_mod.reset_working_memory_for_tests()
    memory.init_db()
    yield
    if memory._connection:
        memory._connection.close()
    monkeypatch.setattr(memory, "_connection", None)
    wm_mod.reset_working_memory_for_tests()


def _run(coro):
    return asyncio.run(coro)


# ── update_working_memory ───────────────────────────────────────

def test_update_happy_path():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={"section": "open_theses", "entry": "Long NVDA — chip cycle ramp"},
    ))
    assert res["success"] is True
    assert res["section"] == "open_theses"
    assert res["entry_id"] > 0
    assert res["section_size"] == 1


def test_update_missing_section_returns_error():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={"entry": "no section"},
    ))
    assert "error" in res
    assert "section required" in res["error"]
    assert "valid_sections" in res


def test_update_unknown_section_returns_error():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={"section": "feelings", "entry": "scared"},
    ))
    assert "error" in res
    assert "feelings" in res["error"]


def test_update_missing_entry_returns_error():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={"section": "open_theses"},
    ))
    assert "error" in res
    assert "entry required" in res["error"]


def test_update_empty_entry_returns_error():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={"section": "open_theses", "entry": "   "},
    ))
    assert "error" in res


def test_update_negative_expiry_returns_error():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={
            "section": "watching_for",
            "entry": "watch UNH",
            "expires_in_minutes": -5,
        },
    ))
    assert "error" in res
    assert ">= 0" in res["error"]


def test_update_non_numeric_expiry_returns_error():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={
            "section": "watching_for",
            "entry": "watch UNH",
            "expires_in_minutes": "soon",
        },
    ))
    assert "error" in res


def test_update_non_dict_metadata_returns_error():
    from tools.tools_working_memory import handle_update_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={
            "section": "watching_for",
            "entry": "watch UNH",
            "metadata": "not a dict",
        },
    ))
    assert "error" in res


def test_update_metadata_persists():
    from tools.tools_working_memory import handle_update_working_memory
    from memory.working_memory import get_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={
            "section": "watching_for",
            "entry": "UNH > 530 with vol",
            "metadata": {"symbol": "UNH", "level": 530.0, "confirm": "volume"},
        },
    ))
    assert res["success"] is True
    snap = get_working_memory().snapshot()
    entry = next(e for e in snap["watching_for"] if e["id"] == res["entry_id"])
    assert entry["metadata"]["symbol"] == "UNH"
    assert entry["metadata"]["level"] == 530.0


def test_update_explicit_expiry_passes_through():
    from tools.tools_working_memory import handle_update_working_memory
    from memory.working_memory import get_working_memory
    res = _run(handle_update_working_memory(
        executor=None,
        params={
            "section": "recent_verdicts",
            "entry": "BUY UNH calls",
            "expires_in_minutes": 120,
        },
    ))
    snap = get_working_memory().snapshot()
    entry = next(e for e in snap["recent_verdicts"] if e["id"] == res["entry_id"])
    delta = entry["expires_ts"] - entry["created_ts"]
    assert abs(delta - 7200.0) < 5.0


def test_update_reports_section_size_and_cap():
    from tools.tools_working_memory import handle_update_working_memory
    from memory.working_memory import SECTION_CAPS
    res1 = _run(handle_update_working_memory(
        executor=None,
        params={"section": "regime_notes", "entry": "high IV regime"},
    ))
    res2 = _run(handle_update_working_memory(
        executor=None,
        params={"section": "regime_notes", "entry": "tight breadth"},
    ))
    assert res2["section_size"] == 2
    assert res2["section_cap"] == SECTION_CAPS["regime_notes"]
    assert res1["entry_id"] != res2["entry_id"]


# ── clear_working_memory_entry ──────────────────────────────────

def test_clear_by_id():
    from tools.tools_working_memory import (
        handle_update_working_memory, handle_clear_working_memory_entry,
    )
    add = _run(handle_update_working_memory(
        executor=None,
        params={"section": "lessons_today", "entry": "do not chase"},
    ))
    eid = add["entry_id"]
    res = _run(handle_clear_working_memory_entry(
        executor=None,
        params={"section": "lessons_today", "entry_id": eid},
    ))
    assert res["success"] is True
    assert res["removed"] == 1


def test_clear_whole_section():
    from tools.tools_working_memory import (
        handle_update_working_memory, handle_clear_working_memory_entry,
    )
    _run(handle_update_working_memory(
        executor=None,
        params={"section": "lessons_today", "entry": "a"}))
    _run(handle_update_working_memory(
        executor=None,
        params={"section": "lessons_today", "entry": "b"}))
    res = _run(handle_clear_working_memory_entry(
        executor=None,
        params={"section": "lessons_today"},
    ))
    assert res["success"] is True
    assert res["removed"] == 2


def test_clear_unknown_section_returns_error():
    from tools.tools_working_memory import handle_clear_working_memory_entry
    res = _run(handle_clear_working_memory_entry(
        executor=None,
        params={"section": "feelings"},
    ))
    assert "error" in res


def test_clear_non_int_entry_id_returns_error():
    from tools.tools_working_memory import handle_clear_working_memory_entry
    res = _run(handle_clear_working_memory_entry(
        executor=None,
        params={"section": "open_theses", "entry_id": "abc"},
    ))
    assert "error" in res


def test_clear_unknown_id_is_noop_success():
    from tools.tools_working_memory import handle_clear_working_memory_entry
    res = _run(handle_clear_working_memory_entry(
        executor=None,
        params={"section": "open_theses", "entry_id": 99999},
    ))
    assert res["success"] is True
    assert res["removed"] == 0


# ── registration ────────────────────────────────────────────────

def test_handlers_registered_in_tools_executor():
    from tools.tools_executor import _REGISTRY
    assert "update_working_memory" in _REGISTRY
    assert "clear_working_memory_entry" in _REGISTRY


def test_executor_dispatches_to_handler():
    """Round-trip via the unified executor registry."""
    from tools.tools_executor import _REGISTRY
    handler = _REGISTRY["update_working_memory"]
    res = _run(handler(
        None,
        {"section": "regime_notes", "entry": "vix 18 → sell vol"},
    ))
    assert res["success"] is True
