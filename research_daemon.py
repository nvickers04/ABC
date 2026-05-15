"""
Research daemon — standalone entry point that owns the scorer and
template-evolution loops on its own clock.

Run alongside the trading agent in a second terminal:

    python research_daemon.py
    python research_daemon.py --no-evolution
    python research_daemon.py --verbose

Heartbeat is written each scoring round (signals.scorer round body)
into the ``research_config`` table under
``core.runtime.heartbeat.HEARTBEAT_KEY``.  When fresh (<60s by
default), ``__main__.py`` sees it and skips spawning its own
in-process scorer — so single-process dev mode still works when the
daemon isn't running.

The daemon does NOT touch:
  * the trader's working memory (only the agent writes there)
  * IBKR account / order tools (read-only data only)
  * Grok / xAI (no LLM calls)

It uses the session-aware cadence in ``core/runtime/cadence.py`` so
overnight is cheap and regular hours are responsive.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal as _signal
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


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
        log_dir / "research_daemon.log",
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
                "ib_insync.decoder", "ib_insync.connection",
                "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)


async def _run(*, verbose: bool, run_evolution: bool) -> None:
    logger = logging.getLogger("research_daemon")
    # Initialize DB once on this process so WAL is on and tables exist.
    from memory import init_db
    init_db()

    # Spawn template evolution thread if requested (it has its own
    # internal cadence; cheap to keep it on the same process).
    if run_evolution:
        try:
            from signals.template_evolution import run_template_evolution_threaded
            run_template_evolution_threaded()
            logger.info("Template evolution thread started (default on; matches trader split)")
        except Exception as e:
            logger.warning("Template evolution failed to start: %s", e)
    else:
        logger.info("Template evolution disabled (--no-evolution)")

    # Run the scorer on this loop directly with cadence mode on.
    from signals.scorer import run_research
    logger.info("Research daemon online — cadence mode enabled")
    await run_research(verbose=verbose, use_cadence=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ABC Research Daemon (no LLM)")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument(
        "--no-evolution",
        action="store_true",
        help="Don't spawn the template-evolution thread",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger("research_daemon")

    # Force UTF-8 stdio (mirrors __main__.py).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    # Make sure cwd is the repo root so relative paths (logs/, .env) work.
    os.chdir(Path(__file__).resolve().parent)

    # Load .env so MarketData / IBKR credentials are present.
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception as e:
        logger.debug(".env load skipped: %s", e)

    # Force-disable IBKR quote routing in this process. The daemon does NOT
    # connect to IBKR (only the trader does), so leaving IBKR_QUOTES_ENABLED=1
    # would cause data_provider.get_quote() to route to a never-connected
    # IBKR singleton and return None, making signals abstain.  The daemon's
    # real-time path is MarketData.app (trader plan, not delayed).
    os.environ["IBKR_QUOTES_ENABLED"] = "0"
    try:
        from core import config as _cfg
        _cfg.IBKR_QUOTES_ENABLED = False
        logger.info("IBKR quote routing disabled for daemon (MDA is the real-time source)")
    except Exception as e:
        logger.warning("Could not override IBKR_QUOTES_ENABLED: %s", e)

    # ── Mandatory Researcher Machine Boundaries (two-machine production) ─────
    # 1. MDA streaming must be healthy before any scoring starts.
    # 2. Hard daily cap RESEARCHER_DAILY_TOKEN_CAP (default 100k) — no unbounded research.
    #    Early shutdown + clear logging if approaching / exceeding limit.
    try:
        from core.config import (
            RESEARCHER_DAILY_TOKEN_CAP,
            RESEARCHER_MDA_HEALTH_CHECK_ENABLED,
        )
        from data.data_provider import get_data_provider
        from memory import get_research_config, set_research_config

        if RESEARCHER_MDA_HEALTH_CHECK_ENABLED:
            dp = get_data_provider()
            try:
                # Cheap health probe (liquid name, minimal credits)
                quote = dp.get_quote("SPY")
                if not quote or quote.get("last") is None:
                    raise RuntimeError("MDA health probe returned no usable quote")
                logger.info("MDA health check PASSED (SPY quote OK)")
            except Exception as mda_err:
                logger.critical("MDA HEALTH CHECK FAILED: %s", mda_err)
                logger.critical("Research daemon refusing to start — Market Data streaming is not healthy.")
                logger.critical("Fix MDA token / connectivity on this researcher host, then restart.")
                sys.exit(3)

        # Daily token / activity cap (persisted in research_config)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cap_key = f"researcher_daily_usage_{today}"
        current_usage = float(get_research_config(cap_key, 0.0))
        if current_usage >= RESEARCHER_DAILY_TOKEN_CAP:
            logger.critical(
                "RESEARCHER DAILY TOKEN CAP EXCEEDED: %.0f >= %d (key=%s). "
                "Shutting down for the remainder of the UTC day to prevent unbounded research.",
                current_usage, RESEARCHER_DAILY_TOKEN_CAP, cap_key
            )
            sys.exit(4)
        if current_usage > RESEARCHER_DAILY_TOKEN_CAP * 0.8:
            logger.warning(
                "RESEARCHER TOKEN USAGE APPROACHING CAP: %.0f / %d (%.1f%%). "
                "Will hard-stop at limit.",
                current_usage, RESEARCHER_DAILY_TOKEN_CAP, (current_usage / RESEARCHER_DAILY_TOKEN_CAP) * 100
            )
        logger.info("Researcher boundaries OK (daily usage=%.0f / cap=%d)", current_usage, RESEARCHER_DAILY_TOKEN_CAP)
    except SystemExit:
        raise
    except Exception as bound_err:
        logger.warning("Researcher boundary check encountered non-fatal issue: %s (proceeding with caution)", bound_err)

    print("=== Research daemon ===")
    print("This process: tiered signal scorer + template evolution (use --no-evolution to turn evolution off).")
    print("No Grok/LLM, no IBKR orders. Ctrl+C to stop.\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop(*_: object) -> None:
        logger.info("Stop signal received — shutting down")
        for task in asyncio.all_tasks(loop=loop):
            task.cancel()

    try:
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(_signal, sig_name, None)
            if sig is not None:
                try:
                    loop.add_signal_handler(sig, _stop)
                except (NotImplementedError, RuntimeError):
                    # Windows: add_signal_handler not supported on Proactor;
                    # KeyboardInterrupt still works.
                    pass
        loop.run_until_complete(_run(verbose=args.verbose,
                                     run_evolution=not args.no_evolution))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        try:
            loop.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
