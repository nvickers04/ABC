"""Tests for memory.working_memory — store, curator, renderer, persistence."""

from __future__ import annotations

import time

import pytest


# ── Fixtures ──
# Use the shared modern _isolated_db from tests/conftest.py (reset_state + init_db).
# The old fixture below was removed because it poked non-existent private attrs
# (_connection) and used sqlite_master after the memory layer migrated to Postgres.


@pytest.fixture
def db():
    from memory import get_db
    return get_db()


@pytest.fixture
def wm(db):
    from memory.working_memory import WorkingMemory
    return WorkingMemory(db)


# ── Schema ───────────────────────────────────────────────────────

def test_working_memory_table_exists(db):
    # Postgres-compatible table existence check (information_schema).
    # init_db in the autouse fixture guarantees the table is present.
    cur = db.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = 'working_memory'"
    )
    assert cur.fetchone() is not None


def test_working_memory_required_columns(db):
    cur = db.execute("PRAGMA table_info(working_memory)")
    cols = {row[1] for row in cur.fetchall()}
    for required in ("id", "section", "entry_text", "created_ts",
                     "expires_ts", "metadata_json"):
        assert required in cols


# ── add / validation ────────────────────────────────────────────

def test_add_returns_id_and_persists(wm, db):
    eid = wm.add("open_theses", "NVDA chip-cycle long; ride GB200 ramp")
    assert eid > 0
    cur = db.execute("SELECT id, section, entry_text FROM working_memory WHERE id = ?", (eid,))
    row = cur.fetchone()
    assert row is not None
    assert row[1] == "open_theses"
    assert "NVDA" in row[2]


def test_add_unknown_section_raises(wm):
    with pytest.raises(ValueError):
        wm.add("not_a_section", "anything")


def test_add_empty_entry_raises(wm):
    with pytest.raises(ValueError):
        wm.add("open_theses", "   ")


def test_add_persists_metadata(wm, db):
    eid = wm.add(
        "watching_for",
        "watch UNH > 530 with vol confirm",
        metadata={"symbol": "UNH", "level": 530.0},
    )
    cur = db.execute("SELECT metadata_json FROM working_memory WHERE id = ?", (eid,))
    row = cur.fetchone()
    assert row is not None
    import json
    meta = json.loads(row[0])
    assert meta["symbol"] == "UNH"
    assert meta["level"] == 530.0


# ── expiry ──────────────────────────────────────────────────────

def test_default_expiry_recent_verdicts_is_30min(wm):
    now = time.time()
    eid = wm.add("recent_verdicts", "PASS on TSLA — IV too rich", now_ts=now)
    snap = wm.snapshot()
    entry = next(e for e in snap["recent_verdicts"] if e["id"] == eid)
    # 30 minutes ± 1 second
    assert abs((entry["expires_ts"] - now) - 30 * 60) < 1.0


def test_default_expiry_watching_for_is_60min(wm):
    now = time.time()
    eid = wm.add("watching_for", "wait for SPY hold above 5800", now_ts=now)
    snap = wm.snapshot()
    entry = next(e for e in snap["watching_for"] if e["id"] == eid)
    assert abs((entry["expires_ts"] - now) - 60 * 60) < 1.0


def test_explicit_expiry_overrides_default(wm):
    now = time.time()
    eid = wm.add(
        "recent_verdicts", "BUY UNH long calls",
        expires_in_minutes=120, now_ts=now,
    )
    snap = wm.snapshot()
    entry = next(e for e in snap["recent_verdicts"] if e["id"] == eid)
    assert abs((entry["expires_ts"] - now) - 2 * 3600) < 1.0


def test_eod_default_for_open_theses(wm):
    now = time.time()
    eid = wm.add("open_theses", "long-vol thesis given Fed week", now_ts=now)
    snap = wm.snapshot()
    entry = next(e for e in snap["open_theses"] if e["id"] == eid)
    # EOD must be in the future, less than 24h from now
    delta = entry["expires_ts"] - now
    assert 0 < delta <= 24 * 3600 + 1


# ── curator ─────────────────────────────────────────────────────

def test_curate_drops_expired(wm, db):
    now = time.time()
    fresh = wm.add("recent_verdicts", "fresh", expires_in_minutes=60, now_ts=now)
    stale = wm.add("recent_verdicts", "stale", expires_in_minutes=1, now_ts=now)
    # Force the stale entry's expiry into the past in BOTH memory and DB.
    db.execute("UPDATE working_memory SET expires_ts = ? WHERE id = ?", (now - 10, stale))
    db.commit()
    for e in wm._entries["recent_verdicts"]:
        if e.id == stale:
            e.expires_ts = now - 10
    dropped = wm.curate(now_ts=now)
    assert dropped == 1
    snap = wm.snapshot()
    ids = {e["id"] for e in snap["recent_verdicts"]}
    assert fresh in ids
    assert stale not in ids
    cur = db.execute("SELECT COUNT(*) FROM working_memory WHERE id = ?", (stale,))
    assert cur.fetchone()[0] == 0


def test_curate_returns_zero_when_nothing_expired(wm):
    now = time.time()
    wm.add("watching_for", "x", expires_in_minutes=60, now_ts=now)
    assert wm.curate(now_ts=now) == 0


# ── caps and eviction ────────────────────────────────────────────

def test_cap_evicts_oldest_expires_first(wm, db):
    """recent_verdicts cap=12; 13th add evicts the soonest-to-expire."""
    now = time.time()
    # Fill 12: short expiries (5 min)
    short_ids = [
        wm.add("recent_verdicts", f"short {i}", expires_in_minutes=5, now_ts=now + i)
        for i in range(12)
    ]
    # 13th: long expiry (120 min)
    long_id = wm.add(
        "recent_verdicts", "long survivor",
        expires_in_minutes=120, now_ts=now + 13,
    )
    snap = wm.snapshot()
    ids_after = {e["id"] for e in snap["recent_verdicts"]}
    # Long entry survives; one of the short entries was evicted
    assert long_id in ids_after
    assert len(ids_after) == 12
    # The evicted short id should also be gone from disk
    evicted = set(short_ids) - ids_after
    assert len(evicted) == 1
    eid = next(iter(evicted))
    cur = db.execute("SELECT COUNT(*) FROM working_memory WHERE id = ?", (eid,))
    assert cur.fetchone()[0] == 0


def test_cap_independently_per_section(wm):
    """Filling open_theses to its cap shouldn't touch other sections."""
    for i in range(8):  # cap
        wm.add("open_theses", f"thesis {i}")
    wm.add("watching_for", "watch foo")
    snap = wm.snapshot()
    assert len(snap["open_theses"]) == 8
    assert len(snap["watching_for"]) == 1


# ── clear ───────────────────────────────────────────────────────

def test_clear_by_id(wm, db):
    eid = wm.add("regime_notes", "high-vol regime today")
    removed = wm.clear("regime_notes", entry_id=eid)
    assert removed == 1
    assert wm.snapshot()["regime_notes"] == []
    cur = db.execute("SELECT COUNT(*) FROM working_memory WHERE id = ?", (eid,))
    assert cur.fetchone()[0] == 0


def test_clear_section_wipes_all(wm, db):
    wm.add("lessons_today", "do not chase IV after open")
    wm.add("lessons_today", "size down on Fed days")
    removed = wm.clear("lessons_today")
    assert removed == 2
    assert wm.snapshot()["lessons_today"] == []
    cur = db.execute("SELECT COUNT(*) FROM working_memory WHERE section='lessons_today'")
    assert cur.fetchone()[0] == 0


def test_clear_unknown_id_is_noop(wm):
    wm.add("regime_notes", "x")
    assert wm.clear("regime_notes", entry_id=999_999) == 0
    assert len(wm.snapshot()["regime_notes"]) == 1


# ── render ──────────────────────────────────────────────────────

def test_render_empty_block(wm):
    out = wm.render()
    assert "WORKING MEMORY" in out
    assert "empty" in out.lower()


def test_render_omits_empty_sections(wm):
    wm.add("open_theses", "ride chip cycle")
    out = wm.render()
    assert "[open_theses]" in out
    assert "ride chip cycle" in out
    # Sections without entries must NOT appear
    assert "[recent_verdicts]" not in out
    assert "[watching_for]" not in out


def test_render_skips_expired_entries(wm, db):
    now = time.time()
    eid = wm.add("recent_verdicts", "stale", expires_in_minutes=10, now_ts=now)
    db.execute("UPDATE working_memory SET expires_ts = ? WHERE id = ?", (now - 1, eid))
    db.commit()
    # Force in-memory expiry to also be in the past
    for e in wm._entries["recent_verdicts"]:
        if e.id == eid:
            e.expires_ts = now - 1
    out = wm.render(now_ts=now)
    assert "stale" not in out


def test_render_shows_age(wm):
    now = time.time()
    wm.add("watching_for", "BTFD if NVDA holds 200", now_ts=now - 5 * 60)
    out = wm.render(now_ts=now)
    # Should show minutes-ago badge
    assert "5m ago" in out


# ── persistence: restore_today ──────────────────────────────────

def test_restore_today_loads_valid_entries(db):
    from memory.working_memory import WorkingMemory
    now = time.time()
    wm1 = WorkingMemory(db)
    wm1.add("open_theses", "thesis-A", now_ts=now)
    wm1.add("watching_for", "watch-B", expires_in_minutes=60, now_ts=now)
    # Simulate process restart: new instance, restore from disk
    wm2 = WorkingMemory(db)
    n = wm2.restore_today(now_ts=now + 60)  # 1 min later
    assert n == 2
    snap = wm2.snapshot()
    texts = {e["entry"] for s in snap.values() for e in s}
    assert "thesis-A" in texts
    assert "watch-B" in texts


def test_restore_today_purges_stale_rows(db):
    from memory.working_memory import WorkingMemory
    now = time.time()
    wm1 = WorkingMemory(db)
    eid_old = wm1.add("watching_for", "yesterday", now_ts=now - 26 * 3600)
    eid_now = wm1.add("watching_for", "today", expires_in_minutes=60, now_ts=now)
    wm2 = WorkingMemory(db)
    n = wm2.restore_today(now_ts=now)
    assert n == 1
    cur = db.execute("SELECT id FROM working_memory")
    remaining = {row[0] for row in cur.fetchall()}
    assert eid_now in remaining
    assert eid_old not in remaining


def test_restore_today_purges_already_expired(db):
    from memory.working_memory import WorkingMemory
    now = time.time()
    wm1 = WorkingMemory(db)
    eid = wm1.add("recent_verdicts", "long-stale", expires_in_minutes=1, now_ts=now)
    # Force expiry into the past
    db.execute("UPDATE working_memory SET expires_ts = ? WHERE id = ?", (now - 10, eid))
    db.commit()
    wm2 = WorkingMemory(db)
    n = wm2.restore_today(now_ts=now)
    assert n == 0
    cur = db.execute("SELECT COUNT(*) FROM working_memory")
    assert cur.fetchone()[0] == 0


# ── singleton ────────────────────────────────────────────────────

def test_singleton_returns_same_instance():
    from memory.working_memory import get_working_memory, reset_working_memory_for_tests
    a = get_working_memory()
    b = get_working_memory()
    assert a is b
    reset_working_memory_for_tests()
    c = get_working_memory()
    assert c is not a


# ── snapshot ────────────────────────────────────────────────────

def test_snapshot_includes_all_sections(wm):
    snap = wm.snapshot()
    from memory.working_memory import SECTIONS
    for s in SECTIONS:
        assert s in snap
        assert isinstance(snap[s], list)
