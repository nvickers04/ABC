"""Tests for core.runtime.heartbeat — read/write/staleness."""

from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import memory
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "_DB_PATH", db_path)
    monkeypatch.setattr(memory, "_connection", None)
    monkeypatch.setattr(memory, "_calibration_version", 0)
    memory._pending_graduated_params.clear()
    memory._pending_order_context.clear()
    memory.init_db()
    yield
    if memory._connection:
        memory._connection.close()
    monkeypatch.setattr(memory, "_connection", None)


def test_read_when_never_written_returns_zero():
    from core.runtime.heartbeat import read_heartbeat, is_daemon_alive
    assert read_heartbeat() == 0.0
    assert is_daemon_alive() is False


def test_write_then_read_round_trips():
    from core.runtime.heartbeat import write_heartbeat, read_heartbeat
    ts = 1_700_000_000.0
    written = write_heartbeat(now=ts)
    assert written == pytest.approx(ts)
    assert read_heartbeat() == pytest.approx(ts)


def test_is_daemon_alive_fresh_heartbeat():
    from core.runtime.heartbeat import write_heartbeat, is_daemon_alive
    now = time.time()
    write_heartbeat(now=now)
    assert is_daemon_alive(stale_after_s=60.0) is True


def test_is_daemon_alive_stale_heartbeat():
    from core.runtime.heartbeat import write_heartbeat, is_daemon_alive
    now = time.time()
    write_heartbeat(now=now - 120.0)  # 2 min ago
    assert is_daemon_alive(stale_after_s=60.0, now=now) is False


def test_heartbeat_age_s_infinite_when_unset():
    from core.runtime.heartbeat import heartbeat_age_s
    assert heartbeat_age_s() == float("inf")


def test_heartbeat_age_s_returns_seconds_since_write():
    from core.runtime.heartbeat import write_heartbeat, heartbeat_age_s
    now = time.time()
    write_heartbeat(now=now - 30.0)
    age = heartbeat_age_s(now=now)
    assert 29.5 <= age <= 30.5
