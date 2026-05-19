"""Tests for core.runtime.heartbeat — read/write/staleness."""

from __future__ import annotations

import time

import pytest


# Local _isolated_db removed — uses the single robust _isolated_db from tests/conftest.py
# (dotenv + reset_state + graceful skip on Postgres connection/permission errors).

def test_read_when_never_written_returns_zero():
    from core.runtime.heartbeat import read_heartbeat, is_research_host_alive
    assert read_heartbeat() == 0.0
    assert is_research_host_alive() is False


def test_write_then_read_round_trips():
    from core.runtime.heartbeat import write_heartbeat, read_heartbeat
    ts = 1_700_000_000.0
    written = write_heartbeat(now=ts)
    assert written == pytest.approx(ts)
    assert read_heartbeat() == pytest.approx(ts)


def test_research_host_alive_fresh_heartbeat():
    from core.runtime.heartbeat import write_heartbeat, is_research_host_alive
    now = time.time()
    write_heartbeat(now=now)
    assert is_research_host_alive(stale_after_s=60.0) is True


def test_research_host_alive_stale_heartbeat():
    from core.runtime.heartbeat import write_heartbeat, is_research_host_alive
    now = time.time()
    write_heartbeat(now=now - 120.0)  # 2 min ago
    assert is_research_host_alive(stale_after_s=60.0, now=now) is False


def test_heartbeat_age_s_infinite_when_unset():
    from core.runtime.heartbeat import heartbeat_age_s
    assert heartbeat_age_s() == float("inf")


def test_heartbeat_age_s_returns_seconds_since_write():
    from core.runtime.heartbeat import write_heartbeat, heartbeat_age_s
    now = time.time()
    write_heartbeat(now=now - 30.0)
    age = heartbeat_age_s(now=now)
    assert 29.5 <= age <= 30.5


def test_read_legacy_heartbeat_key():
    from core.runtime.heartbeat import (
        HEARTBEAT_KEY,
        LEGACY_HEARTBEAT_KEY,
        read_heartbeat,
    )
    from memory import get_research_config, set_research_config

    ts = 1_700_000_123.0
    set_research_config(HEARTBEAT_KEY, 0.0, reason="test reset")
    set_research_config(LEGACY_HEARTBEAT_KEY, ts, reason="legacy heartbeat")
    assert get_research_config(HEARTBEAT_KEY, 0.0) == 0.0
    assert read_heartbeat() == pytest.approx(ts)
