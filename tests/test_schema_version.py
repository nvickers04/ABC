"""Schema version / migration startup checks."""

from __future__ import annotations

import sqlite3
import time

import pytest

import memory
from memory import _apply_schema_migrations, get_schema_version


def _fresh_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


class TestGetSchemaVersion:
    def test_legacy_db_no_table_returns_one(self):
        conn = _fresh_conn()
        assert get_schema_version(conn) == 1

    def test_empty_table_returns_one(self):
        conn = _fresh_conn()
        conn.execute(
            "CREATE TABLE schema_version ("
            "version INTEGER PRIMARY KEY, applied_ts REAL NOT NULL, description TEXT)"
        )
        assert get_schema_version(conn) == 1

    def test_returns_max_stamped_version(self):
        conn = _fresh_conn()
        conn.execute(
            "CREATE TABLE schema_version ("
            "version INTEGER PRIMARY KEY, applied_ts REAL NOT NULL, description TEXT)"
        )
        conn.execute(
            "INSERT INTO schema_version VALUES (?, ?, ?)", (2, time.time(), "v2")
        )
        assert get_schema_version(conn) == 2


class TestApplySchemaMigrations:
    def test_future_version_raises(self):
        """A DB stamped newer than this build should fail loudly."""
        conn = _fresh_conn()
        conn.execute(
            "CREATE TABLE schema_version ("
            "version INTEGER PRIMARY KEY, applied_ts REAL NOT NULL, description TEXT)"
        )
        conn.execute(
            "INSERT INTO schema_version VALUES (?, ?, ?)",
            (memory.SCHEMA_VERSION + 5, time.time(), "from-the-future"),
        )
        with pytest.raises(RuntimeError, match="newer than this build"):
            _apply_schema_migrations(conn)

    def test_current_version_is_noop(self):
        conn = _fresh_conn()
        conn.execute(
            "CREATE TABLE schema_version ("
            "version INTEGER PRIMARY KEY, applied_ts REAL NOT NULL, description TEXT)"
        )
        conn.execute(
            "INSERT INTO schema_version VALUES (?, ?, ?)",
            (memory.SCHEMA_VERSION, time.time(), "current"),
        )
        # Should not raise and should not insert another row.
        _apply_schema_migrations(conn)
        rows = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        assert rows[0] == 1
