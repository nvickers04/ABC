"""
Research host process — scorer + template evolution (no Grok, no IBKR orders).

Run on the research machine (see ``python -m research --help``):

    python -m research
    python -m research -v
    python -m research --no-evolution

Heartbeat and status are written to Postgres each scoring round so the trader
can skip its in-process scorer when the research host is healthy.
"""

from __future__ import annotations

import asyncio
import os
import signal as _signal
import sys
from pathlib import Path

from core.log_context import (
    bind_research_host_context,
    ensure_utf8_stdio,
    get_logger,
    log_banner,
)


async def _run(*, verbose: bool, run_evolution: bool) -> None:
    logger = get_logger("research.host")
    bind_research_host_context()
    from memory import init_db

    from core.runtime.heartbeat import ResearchHostStatus, publish_research_host_heartbeat
    from core.runtime.research_host_runtime import (
        finalize_shutdown,
        is_research_host_process,
        shutdown_reason,
    )

    init_db()
    publish_research_host_heartbeat(status=ResearchHostStatus.STARTING)

    if run_evolution:
        try:
            from signals.template_evolution import run_template_evolution_threaded

            run_template_evolution_threaded()
            logger.info("template_evolution_started")
        except Exception as e:
            logger.warning("template_evolution_failed", error=str(e))
    else:
        logger.info("template_evolution_disabled")

    from signals.scorer import run_research

    logger.info("research_host_online", cadence=True, research_host=is_research_host_process())
    publish_research_host_heartbeat(status=ResearchHostStatus.RUNNING)

    try:
        await run_research(verbose=verbose, use_cadence=True)
    finally:
        if shutdown_reason() == "daily_token_cap":
            publish_research_host_heartbeat(status=ResearchHostStatus.CAP_STOPPED)
        finalize_shutdown()


def main() -> None:
    from core.entry_cli import parse_research_args

    args = parse_research_args()

    from core.log_setup import configure_root_logging

    configure_root_logging(
        "research.log",
        verbose=args.verbose,
        extra_quiet_loggers=("yfinance", "peewee"),
    )
    logger = get_logger("research.host")
    ensure_utf8_stdio()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=True)
    except Exception as e:
        logger.debug("dotenv_skipped", error=str(e))

    from core.runtime.research_host_runtime import mark_research_host_process

    mark_research_host_process()
    bind_research_host_context()

    os.environ["IBKR_QUOTES_ENABLED"] = "0"
    try:
        from core import config as _cfg

        _cfg.IBKR_QUOTES_ENABLED = False
        logger.info("ibkr_quotes_disabled", reason="mda_source")
    except Exception as e:
        logger.warning("ibkr_quotes_override_failed", error=str(e))

    try:
        from core.config import RESEARCHER_MDA_HEALTH_CHECK_ENABLED
        from core.runtime.research_host_runtime import check_startup_token_cap
        from data.data_provider import get_data_provider

        cap = check_startup_token_cap()
        if cap.exceeded:
            logger.critical(
                "researcher_token_cap_exceeded_at_startup",
                usage=cap.usage,
                cap=cap.cap,
            )
            sys.exit(4)
        if cap.warn:
            logger.warning(
                "researcher_token_cap_warning_at_startup",
                usage=cap.usage,
                cap=cap.cap,
                pct=round(cap.pct, 1),
            )
        logger.info(
            "researcher_boundaries_ok",
            usage=cap.usage,
            cap=cap.cap,
            pct=round(cap.pct, 1),
        )

        if RESEARCHER_MDA_HEALTH_CHECK_ENABLED:
            dp = get_data_provider()
            try:
                quote = dp.get_quote("SPY")
                if not quote or quote.get("last") is None:
                    raise RuntimeError("MDA health probe returned no usable quote")
                logger.info("mda_health_check_passed", symbol="SPY")
            except Exception as mda_err:
                logger.critical("mda_health_check_failed", error=str(mda_err))
                logger.critical("research_host_refusing_start")
                sys.exit(3)
    except SystemExit:
        raise
    except Exception as bound_err:
        logger.warning("researcher_boundary_check_non_fatal", error=str(bound_err))

    log_banner(
        logger,
        "=== Research host ===",
        lines=(
            "Scorer + template evolution. No Grok, no IBKR orders. Ctrl+C to stop.",
            "",
        ),
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop(*_: object) -> None:
        logger.info("research_host_stop_signal")
        try:
            from core.runtime.research_host_runtime import request_shutdown

            request_shutdown("signal")
        except Exception as e:
            logger.debug("request_shutdown_failed", error=str(e))
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
        logger.info("research_host_interrupted")
        try:
            from core.runtime.research_host_runtime import request_shutdown

            request_shutdown("keyboard_interrupt")
        except Exception:
            pass
    finally:
        try:
            loop.close()
        except Exception:
            pass
