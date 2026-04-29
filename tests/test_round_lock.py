"""Tests for core.runtime.round_lock — file-based one-shot job lock."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from core.runtime.round_lock import acquire_lock, LockBusy, _lock_path


def test_acquires_when_no_lock_exists(tmp_path):
    with acquire_lock("test_a", lock_dir=tmp_path) as p:
        assert p.exists()
        contents = p.read_text(encoding="utf-8")
        assert str(os.getpid()) in contents
    assert not p.exists(), "lock file should be removed on context exit"


def test_releases_on_exception(tmp_path):
    p = _lock_path("test_b", tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with acquire_lock("test_b", lock_dir=tmp_path):
            assert p.exists()
            raise RuntimeError("boom")
    assert not p.exists(), "lock file should be removed even on exception"


def test_raises_lockbusy_when_fresh_lock_exists(tmp_path):
    p = _lock_path("test_c", tmp_path)
    p.write_text("99999 2026-04-29T00:00:00\n", encoding="utf-8")
    # Touch to ensure mtime is recent
    os.utime(p, None)

    with pytest.raises(LockBusy, match="held by pid"):
        with acquire_lock("test_c", lock_dir=tmp_path, stale_after_s=600.0):
            pytest.fail("should not have entered the with block")
    # Existing lock file is preserved (we did NOT steal it)
    assert p.exists()


def test_steals_stale_lock(tmp_path, caplog):
    p = _lock_path("test_d", tmp_path)
    p.write_text("99999 2026-04-29T00:00:00\n", encoding="utf-8")
    # Backdate mtime to 10 hours ago
    old = time.time() - 36_000
    os.utime(p, (old, old))

    caplog.set_level("WARNING", logger="core.runtime.round_lock")
    with acquire_lock("test_d", lock_dir=tmp_path, stale_after_s=600.0) as p2:
        assert p2.exists()
        # New lock should reflect our pid, not the stale one
        contents = p2.read_text(encoding="utf-8")
        assert str(os.getpid()) in contents
        assert "99999" not in contents
    assert any("stale" in r.message.lower() for r in caplog.records)


def test_corrupt_lock_file_treated_as_stealable(tmp_path):
    """A garbage lock file should not block acquisition forever."""
    p = _lock_path("test_e", tmp_path)
    p.write_text("garbage\n", encoding="utf-8")
    # Corrupt → _read_lock returns None → treated as no lock present → acquire OK
    with acquire_lock("test_e", lock_dir=tmp_path) as p2:
        assert p2.exists()
        assert str(os.getpid()) in p2.read_text(encoding="utf-8")


def test_serial_acquire_release_cycles(tmp_path):
    """Acquire/release/acquire in sequence should always succeed."""
    for _ in range(3):
        with acquire_lock("test_f", lock_dir=tmp_path) as p:
            assert p.exists()
        assert not p.exists()


def test_now_override_for_age_calculation(tmp_path):
    """The ``now`` parameter lets tests pin the clock for deterministic checks."""
    p = _lock_path("test_g", tmp_path)
    p.write_text("12345 2026-04-29T00:00:00\n", encoding="utf-8")
    base_mtime = 1_000_000.0
    os.utime(p, (base_mtime, base_mtime))

    # 30s old, stale_after_s=60 → busy
    with pytest.raises(LockBusy):
        with acquire_lock("test_g", lock_dir=tmp_path,
                          stale_after_s=60.0, now=base_mtime + 30):
            pass

    # 90s old, stale_after_s=60 → stealable
    with acquire_lock("test_g", lock_dir=tmp_path,
                      stale_after_s=60.0, now=base_mtime + 90):
        pass
