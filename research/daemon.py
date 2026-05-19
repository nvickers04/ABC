"""
Research host process — scorer + template evolution (no Grok, no IBKR orders).

Run on the research machine:

    python -m research
    python -m research --verbose
    python -m research --no-evolution

Heartbeat is written each scoring round into ``research_config`` so the trader
can skip its in-process scorer when the daemon is healthy.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal as _signal
import sys
from datetime import datetime, timezone
from pathlib import Path


async def _run(*, verbose: bool, run_evolution: bool) -> None:
    logger = logging.getLogger("research.host")
    from memory import init_db

    init_db()

    if run_evolution:
        try:
            from signals.template_evolution import run_template_evolution_threaded

            run_template_evolution_threaded()
            logger.info("Template evolution thread started")
        except Exception as e:
            logger.warning("Template evolution failed to start: %s", e)
    else:
        logger.info("Template evolution disabled (--no-evolution)")

    from signals.scorer import run_research

    logger.info("Research daemon online — cadence mode enabled")
    await run_research(verbose=verbose, use_cadence=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ABC research daemon (no LLM)")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument(
        "--no-evolution",
        action="store_true",
        help="Don't spawn the template-evolution thread",
    )
    args = parser.parse_args()

    from core.log_setup import configure_root_logging

    configure_root_logging(
        "research.log",
        verbose=args.verbose,
        extra_quiet_loggers=("yfinance", "peewee"),
    )
    logger = logging.getLogger("research.host")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=True)
    except Exception as e:
        logger.debug(".env load skipped: %s", e)

    os.environ["IBKR_QUOTES_ENABLED"] = "0"
    try:
        from core import config as _cfg

        _cfg.IBKR_QUOTES_ENABLED = False
        logger.info("IBKR quote routing disabled (MDA is the real-time source)")
    except Exception as e:
        logger.warning("Could not override IBKR_QUOTES_ENABLED: %s", e)

    try:
        from core.config import (
            RESEARCHER_DAILY_TOKEN_CAP,
            RESEARCHER_MDA_HEALTH_CHECK_ENABLED,
        )
        from data.data_provider import get_data_provider
        from memory import get_research_config

        if RESEARCHER_MDA_HEALTH_CHECK_ENABLED:
            dp = get_data_provider()
            try:
                quote = dp.get_quote("SPY")
                if not quote or quote.get("last") is None:
                    raise RuntimeError("MDA health probe returned no usable quote")
                logger.info("MDA health check PASSED (SPY quote OK)")
            except Exception as mda_err:
                logger.critical("MDA HEALTH CHECK FAILED: %s", mda_err)
                logger.critical(
                    "Research daemon refusing to start — fix MDA on this host."
                )
                sys.exit(3)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cap_key = f"researcher_daily_usage_{today}"
        current_usage = float(get_research_config(cap_key, 0.0))
        if current_usage >= RESEARCHER_DAILY_TOKEN_CAP:
            logger.critical(
                "RESEARCHER DAILY TOKEN CAP EXCEEDED: %.0f >= %d",
                current_usage,
                RESEARCHER_DAILY_TOKEN_CAP,
            )
            sys.exit(4)
        if current_usage > RESEARCHER_DAILY_TOKEN_CAP * 0.8:
            logger.warning(
                "RESEARCHER TOKEN USAGE APPROACHING CAP: %.0f / %d",
                current_usage,
                RESEARCHER_DAILY_TOKEN_CAP,
            )
        logger.info(
            "Researcher boundaries OK (daily usage=%.0f / cap=%d)",
            current_usage,
            RESEARCHER_DAILY_TOKEN_CAP,
        )
    except SystemExit:
        raise
    except Exception as bound_err:
        logger.warning(
            "Researcher boundary check non-fatal: %s (proceeding)", bound_err
        )

    print("=== Research daemon ===")
    print("Scorer + template evolution. No Grok, no IBKR orders. Ctrl+C to stop.\n")

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
                    pass
        loop.run_until_complete(
            _run(verbose=args.verbose, run_evolution=not args.no_evolution)
        )
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        try:
            loop.close()
        except Exception:
            pass
