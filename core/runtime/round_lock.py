"""File-based advisory lock for one-shot scheduled jobs.

Used by ``score_round.py`` (and any other scheduled entry point) to
prevent two concurrent invocations from racing — Task Scheduler /
cron can overlap if a previous round runs long.

Design:
  * Lock file path: ``logs/.<name>.lock``
  * Contents: ``"<pid> <iso_timestamp>"``
  * Acquire:  refuses if file exists AND is younger than ``stale_after_s``.
              Otherwise (re)writes the file with current pid + timestamp.
  * Release:  deletes the file.

Stale-lock recovery: if a holder crashed the file persists.  The next
acquire sees the file is older than ``stale_after_s`` and steals it.
Default 600 s (10 minutes) — rounds historically take ~65 s, so this
gives ~10× headroom while still recovering from crashes within one
cadence cycle.

Not safe across machines (no NFS coordination) and not a true mutex —
two processes hitting the file in the same millisecond could both
acquire.  Sufficient for a single Windows host running Task Scheduler.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


DEFAULT_STALE_AFTER_S: float = 600.0
DEFAULT_LOCK_DIR = Path("logs")


class LockBusy(RuntimeError):
    """Raised when the lock is held by another fresh process."""


def _lock_path(name: str, lock_dir: Optional[Path] = None) -> Path:
    base = lock_dir if lock_dir is not None else DEFAULT_LOCK_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f".{name}.lock"


def _read_lock(path: Path) -> Optional[tuple[int, float]]:
    """Return (pid, mtime) if the lock file exists and is parseable, else None."""
    if not path.exists():
        return None
    try:
        contents = path.read_text(encoding="utf-8").strip()
        pid_str = contents.split()[0]
        pid = int(pid_str)
    except (OSError, ValueError, IndexError):
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return pid, mtime


def _write_lock(path: Path) -> None:
    pid = os.getpid()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    path.write_text(f"{pid} {stamp}\n", encoding="utf-8")


@contextmanager
def acquire_lock(
    name: str,
    *,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    lock_dir: Optional[Path] = None,
    now: Optional[float] = None,
) -> Iterator[Path]:
    """Context manager — acquires the named lock or raises ``LockBusy``.

    Usage:

        with acquire_lock("score_round"):
            do_one_round()

    On exit (normal or exception) the lock file is removed.
    """
    path = _lock_path(name, lock_dir)
    cur = float(now) if now is not None else time.time()

    existing = _read_lock(path)
    if existing is not None:
        pid, mtime = existing
        age = cur - mtime
        if age < stale_after_s:
            raise LockBusy(
                f"Lock '{name}' held by pid={pid}, age={age:.1f}s "
                f"(stale_after_s={stale_after_s})"
            )
        logger.warning(
            "Lock '%s' is stale (pid=%d, age=%.1fs >= %.1fs) — stealing",
            name, pid, age, stale_after_s,
        )

    _write_lock(path)
    try:
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Lock '%s' release failed: %s", name, e)
