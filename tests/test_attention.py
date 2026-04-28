"""Tests for core.runtime.attention — parsers, register, evaluate, render."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pytest


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Fresh temp DB per test; reset WorkingMemory singleton."""
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


@pytest.fixture
def db():
    from memory import get_db
    return get_db()


@dataclass
class _FakeQuote:
    last: Optional[float]
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: int = 0

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last


# ── parse_metadata ──────────────────────────────────────────────


def test_parse_metadata_structured_above():
    from core.runtime.attention import parse_metadata
    spec = parse_metadata({
        "symbol": "unh",
        "condition": "above",
        "threshold": 530.0,
        "confirm_with": ["volume", "composite_positive"],
    })
    assert spec == {
        "symbol": "UNH",
        "condition": "above",
        "threshold": 530.0,
        "confirm_with": ["volume", "composite_positive"],
    }


def test_parse_metadata_alias_break_above_to_crosses_above():
    from core.runtime.attention import parse_metadata
    spec = parse_metadata({"symbol": "TSLA", "condition": "break_above", "threshold": "300"})
    assert spec is not None
    assert spec["condition"] == "crosses_above"
    assert spec["threshold"] == 300.0


def test_parse_metadata_rejects_missing_symbol():
    from core.runtime.attention import parse_metadata
    assert parse_metadata({"condition": "above", "threshold": 1.0}) is None


def test_parse_metadata_rejects_unknown_condition():
    from core.runtime.attention import parse_metadata
    assert parse_metadata({"symbol": "X", "condition": "MOON", "threshold": 1.0}) is None


def test_parse_metadata_rejects_missing_threshold():
    from core.runtime.attention import parse_metadata
    assert parse_metadata({"symbol": "X", "condition": "above"}) is None


def test_parse_metadata_rejects_non_dict():
    from core.runtime.attention import parse_metadata
    assert parse_metadata("not a dict") is None
    assert parse_metadata(None) is None


# ── parse_text ──────────────────────────────────────────────────


def test_parse_text_trailer_above_with_confirm():
    from core.runtime.attention import parse_text
    spec = parse_text("watch UNH for break above 530 [trigger: UNH > 530 + vol, composite]")
    assert spec is not None
    assert spec["symbol"] == "UNH"
    assert spec["condition"] == "above"
    assert spec["threshold"] == 530.0
    assert spec["confirm_with"] == ["vol", "composite"]


def test_parse_text_trailer_below():
    from core.runtime.attention import parse_text
    spec = parse_text("[trigger: AAPL < 175.5]")
    assert spec is not None
    assert spec["symbol"] == "AAPL"
    assert spec["condition"] == "below"
    assert spec["threshold"] == 175.5
    assert spec["confirm_with"] == []


def test_parse_text_fallback_break_above_is_crosses_above():
    from core.runtime.attention import parse_text
    spec = parse_text("watching NVDA for a break above 950")
    assert spec is not None
    assert spec["symbol"] == "NVDA"
    assert spec["condition"] == "crosses_above"
    assert spec["threshold"] == 950.0


def test_parse_text_fallback_simple_above():
    from core.runtime.attention import parse_text
    spec = parse_text("if SPY closes above 520, lean bullish")
    assert spec is not None
    assert spec["condition"] == "above"
    assert spec["threshold"] == 520.0


def test_parse_text_returns_none_on_narrative():
    from core.runtime.attention import parse_text
    assert parse_text("just feeling cautious about tech today") is None


def test_parse_text_handles_empty_and_non_string():
    from core.runtime.attention import parse_text
    assert parse_text("") is None
    assert parse_text(None) is None
    assert parse_text(123) is None


# ── register_trigger / cap eviction ──────────────────────────────


def test_register_trigger_inserts_active_row(db):
    from core.runtime.attention import register_trigger
    rid = register_trigger(
        db, symbol="UNH", condition="above", threshold=530.0,
        confirm_with=["volume"], source_text="watch UNH > 530",
    )
    assert isinstance(rid, int)
    cur = db.execute(
        "SELECT symbol, condition, threshold, state, confirm_with_json, source_text "
        "FROM attention_triggers WHERE id=?",
        (rid,),
    )
    row = cur.fetchone()
    assert row[0] == "UNH"
    assert row[1] == "above"
    assert row[2] == 530.0
    assert row[3] == "active"
    assert "volume" in row[4]
    assert row[5] == "watch UNH > 530"


def test_register_trigger_rejects_bad_input(db):
    from core.runtime.attention import register_trigger
    assert register_trigger(db, symbol="", condition="above", threshold=1.0) is None
    assert register_trigger(db, symbol="X", condition="??", threshold=1.0) is None
    assert register_trigger(db, symbol="X", condition="above", threshold=None) is None


def test_register_trigger_evicts_oldest_past_cap(db):
    from core.runtime.attention import register_trigger, ACTIVE_CAP
    base = time.time() - 1000.0
    ids = []
    for i in range(ACTIVE_CAP):
        ids.append(register_trigger(
            db, symbol=f"S{i}", condition="above", threshold=float(i),
            now=base + i,
        ))
    # All active so far.
    cur = db.execute("SELECT COUNT(*) FROM attention_triggers WHERE state='active'")
    assert cur.fetchone()[0] == ACTIVE_CAP
    # One more — should evict the oldest.
    new_id = register_trigger(
        db, symbol="NEW", condition="above", threshold=99.0, now=base + 999,
    )
    assert new_id is not None
    cur = db.execute("SELECT COUNT(*) FROM attention_triggers WHERE state='active'")
    assert cur.fetchone()[0] == ACTIVE_CAP
    cur = db.execute(
        "SELECT state FROM attention_triggers WHERE id=?", (ids[0],),
    )
    assert cur.fetchone()[0] == "evicted"


# ── evaluate ────────────────────────────────────────────────────


def test_evaluate_above_fires_and_marks_state(db):
    from core.runtime.attention import register_trigger, evaluate
    rid = register_trigger(db, symbol="UNH", condition="above", threshold=530.0)
    fired = evaluate(db, quotes={"UNH": _FakeQuote(last=530.5)}, wake=False)
    assert len(fired) == 1
    assert fired[0]["symbol"] == "UNH"
    assert fired[0]["value"] == 530.5
    cur = db.execute("SELECT state, fired_ts FROM attention_triggers WHERE id=?", (rid,))
    state, fired_ts = cur.fetchone()
    assert state == "fired"
    assert fired_ts is not None


def test_evaluate_below_does_not_fire_when_above(db):
    from core.runtime.attention import register_trigger, evaluate
    register_trigger(db, symbol="AAPL", condition="below", threshold=170.0)
    fired = evaluate(db, quotes={"AAPL": _FakeQuote(last=180.0)}, wake=False)
    assert fired == []


def test_evaluate_crosses_above_requires_prior_below(db):
    from core.runtime.attention import register_trigger, evaluate
    register_trigger(db, symbol="NVDA", condition="crosses_above", threshold=950.0)
    # First call: above threshold but no prior — must NOT fire.
    fired = evaluate(db, quotes={"NVDA": _FakeQuote(last=955.0)}, wake=False)
    assert fired == []
    # Now last_value=955; if next obs is also above, still not a fresh cross.
    fired = evaluate(db, quotes={"NVDA": _FakeQuote(last=960.0)}, wake=False)
    assert fired == []


def test_evaluate_crosses_above_fires_on_edge(db):
    from core.runtime.attention import register_trigger, evaluate
    register_trigger(db, symbol="NVDA", condition="crosses_above", threshold=950.0)
    # Prior below threshold.
    evaluate(db, quotes={"NVDA": _FakeQuote(last=940.0)}, wake=False)
    # Now crosses up.
    fired = evaluate(db, quotes={"NVDA": _FakeQuote(last=951.0)}, wake=False)
    assert len(fired) == 1
    assert fired[0]["condition"] == "crosses_above"


def test_evaluate_composite_above_uses_composites_dict(db):
    from core.runtime.attention import register_trigger, evaluate
    register_trigger(db, symbol="TSLA", condition="composite_above", threshold=1.0)
    fired = evaluate(
        db, quotes={"TSLA": _FakeQuote(last=300.0)},
        composites={"TSLA": 1.5}, wake=False,
    )
    assert len(fired) == 1
    assert fired[0]["value"] == 1.5


def test_evaluate_skips_symbols_without_data(db):
    from core.runtime.attention import register_trigger, evaluate
    register_trigger(db, symbol="ZZZ", condition="above", threshold=10.0)
    fired = evaluate(db, quotes={}, wake=False)
    assert fired == []
    cur = db.execute("SELECT state FROM attention_triggers WHERE symbol='ZZZ'")
    assert cur.fetchone()[0] == "active"


def test_evaluate_signals_wake_bus(db):
    from core.runtime.attention import register_trigger, evaluate
    from core.wake_events import wake_bus
    register_trigger(db, symbol="UNH", condition="above", threshold=530.0)
    # Drain any existing signal and reset.
    wake_bus._event.clear()
    wake_bus._reason = None
    fired = evaluate(db, quotes={"UNH": _FakeQuote(last=540.0)}, wake=True)
    assert fired
    assert wake_bus._event.is_set()
    assert wake_bus._reason and "UNH" in wake_bus._reason


# ── sync_from_working_memory ────────────────────────────────────


def test_sync_registers_from_structured_metadata(db):
    from core.runtime.attention import sync_from_working_memory
    from memory.working_memory import get_working_memory
    wm = get_working_memory()
    wm.add(
        "watching_for",
        "watch UNH for break above 530",
        metadata={"symbol": "UNH", "condition": "above", "threshold": 530.0,
                  "confirm_with": ["volume"]},
    )
    n = sync_from_working_memory(db)
    assert n == 1
    cur = db.execute("SELECT symbol, condition, threshold FROM attention_triggers")
    row = cur.fetchone()
    assert tuple(row) == ("UNH", "above", 530.0)
    # Re-running should not double-register.
    assert sync_from_working_memory(db) == 0


def test_sync_falls_back_to_text_parse(db):
    from core.runtime.attention import sync_from_working_memory
    from memory.working_memory import get_working_memory
    wm = get_working_memory()
    wm.add("watching_for", "AAPL break above 175.5")
    n = sync_from_working_memory(db)
    assert n == 1
    cur = db.execute("SELECT symbol, condition, threshold FROM attention_triggers")
    assert tuple(cur.fetchone()) == ("AAPL", "crosses_above", 175.5)


def test_sync_skips_unparseable_narrative(db):
    from core.runtime.attention import sync_from_working_memory
    from memory.working_memory import get_working_memory
    wm = get_working_memory()
    wm.add("watching_for", "feeling cautious about banks today")
    assert sync_from_working_memory(db) == 0


def test_sync_skips_expired_entries(db):
    from core.runtime.attention import sync_from_working_memory
    from memory.working_memory import get_working_memory
    wm = get_working_memory()
    # Already-expired entry.
    wm.add(
        "watching_for", "old SPY above 600",
        metadata={"symbol": "SPY", "condition": "above", "threshold": 600.0},
        expires_in_minutes=0.0001,
    )
    time.sleep(0.05)
    assert sync_from_working_memory(db) == 0


# ── render_attention_block ──────────────────────────────────────


def test_render_empty_returns_empty_string(db):
    from core.runtime.attention import render_attention_block
    assert render_attention_block(db) == ""


def test_render_shows_active_watchers(db):
    from core.runtime.attention import register_trigger, render_attention_block
    register_trigger(
        db, symbol="UNH", condition="above", threshold=530.0,
        confirm_with=["volume"], source_text="watch UNH for break above 530",
    )
    out = render_attention_block(db)
    assert out.startswith("⚡ ATTENTION")
    assert "UNH" in out
    assert "watching" in out
    assert "Confirm with: volume" in out


def test_render_shows_recently_fired_with_note(db):
    from core.runtime.attention import register_trigger, evaluate, render_attention_block
    register_trigger(db, symbol="UNH", condition="above", threshold=530.0)
    evaluate(db, quotes={"UNH": _FakeQuote(last=531.25)}, wake=False)
    out = render_attention_block(db)
    assert "UNH" in out
    assert "531.25" in out


# ── tool integration: structured metadata flows end-to-end ──────


def test_tool_handler_metadata_round_trip(db):
    """update_working_memory(metadata={...}) → sync → trigger → fire."""
    import asyncio
    from tools.tools_working_memory import handle_update_working_memory
    from core.runtime.attention import sync_from_working_memory, evaluate

    class _StubExecutor:
        pass

    res = asyncio.run(handle_update_working_memory(_StubExecutor(), {
        "section": "watching_for",
        "entry": "watch UNH for a break above 530",
        "metadata": {"symbol": "UNH", "condition": "above", "threshold": 530.0,
                     "confirm_with": ["volume"]},
    }))
    assert res.get("success") is True
    assert sync_from_working_memory(db) == 1
    fired = evaluate(db, quotes={"UNH": _FakeQuote(last=530.05)}, wake=False)
    assert len(fired) == 1
    assert fired[0]["symbol"] == "UNH"
