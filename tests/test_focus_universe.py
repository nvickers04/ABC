"""Tests for core.runtime.focus_universe."""

from __future__ import annotations

import sqlite3
import time

import pytest

from core.runtime.focus_universe import get_focus_symbols, merge_universes


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE attention_triggers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            condition TEXT NOT NULL,
            threshold REAL,
            confirm_with_json TEXT,
            source_entry_id INTEGER,
            source_text TEXT,
            created_ts REAL NOT NULL,
            fired_ts REAL,
            fire_value REAL,
            fire_note TEXT,
            last_value REAL,
            state TEXT NOT NULL DEFAULT 'active'
        )
        """
    )
    return conn


def _add(conn, sym: str, *, state: str = "active", offset: float = 0.0):
    conn.execute(
        "INSERT INTO attention_triggers "
        "(symbol, condition, threshold, created_ts, state) VALUES (?,?,?,?,?)",
        (sym, "above", 100.0, time.time() + offset, state),
    )
    conn.commit()


def test_returns_empty_when_no_triggers():
    conn = _make_conn()
    assert get_focus_symbols(conn) == []


def test_returns_only_active_triggers():
    conn = _make_conn()
    _add(conn, "AAPL", state="active")
    _add(conn, "MSFT", state="fired")
    _add(conn, "TSLA", state="evicted")
    out = get_focus_symbols(conn)
    assert out == ["AAPL"]


def test_dedupes_symbols():
    conn = _make_conn()
    _add(conn, "AAPL")
    _add(conn, "AAPL", offset=1.0)
    _add(conn, "MSFT", offset=2.0)
    out = get_focus_symbols(conn)
    # MSFT first (newest), then AAPL.  Each appears once.
    assert sorted(out) == ["AAPL", "MSFT"]
    assert len(out) == 2


def test_orders_newest_first():
    conn = _make_conn()
    _add(conn, "AAPL", offset=0.0)
    _add(conn, "MSFT", offset=10.0)
    _add(conn, "TSLA", offset=5.0)
    out = get_focus_symbols(conn)
    assert out == ["MSFT", "TSLA", "AAPL"]


def test_uppercases_symbols():
    conn = _make_conn()
    _add(conn, "aapl")
    out = get_focus_symbols(conn)
    assert out == ["AAPL"]


def test_respects_limit():
    conn = _make_conn()
    for i, sym in enumerate(["A", "B", "C", "D", "E"]):
        _add(conn, sym, offset=float(i))
    out = get_focus_symbols(conn, limit=2)
    # Newest first: E, D
    assert out == ["E", "D"]


def test_returns_empty_on_db_error():
    """Closed connection should return [] not raise."""
    conn = _make_conn()
    conn.close()
    assert get_focus_symbols(conn) == []


# ── merge_universes ──────────────────────────────────────────


def test_merge_focus_first_then_base():
    out = merge_universes(["AAPL", "MSFT"], ["NVDA"], include_base=True)
    assert out == ["NVDA", "AAPL", "MSFT"]


def test_merge_dedupes_across_lists():
    out = merge_universes(["AAPL", "MSFT", "NVDA"], ["NVDA", "TSLA"],
                          include_base=True)
    # focus first (NVDA, TSLA); then base sans dupes (AAPL, MSFT)
    assert out == ["NVDA", "TSLA", "AAPL", "MSFT"]


def test_merge_focus_only_when_base_excluded():
    out = merge_universes(["AAPL", "MSFT"], ["NVDA", "TSLA"], include_base=False)
    assert out == ["NVDA", "TSLA"]


def test_merge_focus_only_with_empty_focus_returns_empty():
    out = merge_universes(["AAPL", "MSFT"], [], include_base=False)
    assert out == []


def test_merge_normalizes_case_and_whitespace():
    out = merge_universes([" aapl "], ["msft"], include_base=True)
    assert out == ["MSFT", "AAPL"]


def test_merge_skips_non_strings():
    out = merge_universes(["AAPL", None, 42, ""], ["MSFT", ""], include_base=True)
    assert out == ["MSFT", "AAPL"]
