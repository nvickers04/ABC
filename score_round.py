"""Run exactly one signal-scoring round and exit.

Designed to be invoked by Windows Task Scheduler / cron / any external
scheduler.  Replaces (eventually) the perpetual round-loop in
``research_daemon.py`` for the scoring path.

Usage:
    python score_round.py
    python score_round.py --verbose
    python score_round.py --cadence regular         # informational tag
    python score_round.py --no-template-evolution

Behavior:
  1. Acquires a file lock so two scheduled invocations never overlap.
     If the lock is held by a fresh process, exits 75 (EX_TEMPFAIL).
  2. Initializes the DB (idempotent — WAL on, tables created).
  3. Imports all signals to populate SIGNAL_REGISTRY.
  4. Executes one ``signals.scorer._scoring_round``.
  5. Writes the heartbeat (kept for observability + future capture daemon).
  6. Exits 0 on success, 1 on round failure, 75 on lock-busy.

Logging goes to ``logs/score_round.log`` (rotated daily) AND stdout.

This entry point exists alongside ``research_daemon.py`` during the
PR-B1 → PR-B2 migration window — both can run safely (the lock
prevents overlap and rounds are idempotent at the persistence layer).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# UTF-8 stdio (mirrors __main__.py + research_daemon.py)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


# ── Exit codes ───────────────────────────────────────────────────────
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_LOCK_BUSY = 75  # BSD EX_TEMPFAIL — Task Scheduler can retry

LOCK_NAME = "score_round"


def _setup_logging(verbose: bool = False) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)
    fh = TimedRotatingFileHandler(
        log_dir / "score_round.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)
    for lib in ("httpx", "httpcore", "urllib3", "asyncio",
                "ib_insync.wrapper", "ib_insync.ib", "ib_insync.client",
                "ib_insync.decoder", "ib_insync.connection"):
        logging.getLogger(lib).setLevel(logging.WARNING)


async def _run_one_round(*, verbose: bool, cadence_tag: str) -> int:
    """Execute exactly one scoring round.  Returns 0 on success, 1 on failure."""
    logger = logging.getLogger("score_round")

    # Lazy imports — startup-cost only paid when actually scoring.
    from memory import init_db, get_db
    from data.data_provider import DataProvider
    from signals.scorer import _scoring_round, _import_all_signals
    from signals.scorer import init_default_boundaries

    init_db()
    dp = DataProvider()
    conn = get_db()
    init_default_boundaries(conn)
    _import_all_signals()

    from signals.base import SIGNAL_REGISTRY
    logger.info(
        "score_round starting — cadence_tag=%s, signals=%d",
        cadence_tag, len(SIGNAL_REGISTRY),
    )

    t0 = time.time()
    try:
        credits = await _scoring_round(dp, conn, round_num=1)
    except Exception:
        logger.exception("score_round failed in _scoring_round")
        return EXIT_FAIL
    elapsed = time.time() - t0
    logger.info("score_round complete: %.1fs, ~%d credits", elapsed, credits)

    try:
        from core.wake_events import wake_bus
        wake_bus.signal("scorer_round_oneshot")
    except Exception:
        pass

    try:
        from core.runtime.heartbeat import write_heartbeat
        write_heartbeat()
    except Exception as e:
        logger.debug("heartbeat write skipped: %s", e)

    return EXIT_OK


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one signal-scoring round and exit")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument(
        "--cadence",
        choices=["regular", "extended", "overnight", "manual"],
        default="manual",
        help="Informational tag (logged); does not change behavior",
    )
    parser.add_argument(
        "--lock-stale-after",
        type=float,
        default=600.0,
        help="Seconds after which a held lock is considered crashed and stolen "
             "(default: 600)",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger("score_round")

    # Ensure cwd = repo root so relative paths (logs/, .env, memory.db) work.
    os.chdir(Path(__file__).resolve().parent)

    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception as e:
        logger.debug(".env load skipped: %s", e)

    from core.runtime.round_lock import acquire_lock, LockBusy

    try:
        with acquire_lock(LOCK_NAME, stale_after_s=args.lock_stale_after):
            rc = asyncio.run(
                _run_one_round(verbose=args.verbose, cadence_tag=args.cadence)
            )
    except LockBusy as e:
        logger.warning("score_round skipped: %s", e)
        sys.exit(EXIT_LOCK_BUSY)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(EXIT_FAIL)
    except Exception:
        logger.exception("score_round top-level failure")
        sys.exit(EXIT_FAIL)

    sys.exit(rc)


if __name__ == "__main__":
    main()
