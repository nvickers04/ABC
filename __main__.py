"""
Grok Trader — Entry Point

Usage:
    python __main__.py                  # Run agent (paper trading)
    python __main__.py --test           # Test Grok connection
    python __main__.py --verbose        # Debug logging
    python __main__.py --account live   # Live trading (use with caution)

Built for Grok (xAI) — dynamic liquidity, overnight holds OK, configurable risk.
"""

import asyncio
import argparse
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Force UTF-8 on stdio so non-ASCII chars in log messages (em-dash, ≤, ≥, ×, √,
# box-drawing, emoji) don't blow up on Windows cp1252 consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def setup_logging(verbose: bool = False):
    """Configure logging with console + rotating file handler."""
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

    file_handler = TimedRotatingFileHandler(
        log_dir / "agent.log",
        when="midnight",
        interval=1,
        backupCount=7,  # reduced; value preserved in DB snapshots/hypotheses
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    # Silence noisy libraries
    for lib in ("httpx", "httpcore", "grpc", "xai_sdk",
               "ib_insync.wrapper",
               "ib_insync.ib", "ib_insync.client", "ib_insync.decoder",
               "ib_insync.connection", "ib_insync.flexreport", "ib_insync.order",
               "asyncio", "urllib3", "charset_normalizer",
               "hpack", "h2", "nest_asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def validate_startup():
    """Validate critical secrets before launch."""
    from dotenv import load_dotenv
    load_dotenv(override=True)  # override=True so .env always wins over stale shell vars

    errors = []

    # Check Grok API key
    grok_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not grok_key:
        errors.append("Missing GROK_API_KEY or XAI_API_KEY")

    # Validate config invariants (risk %, RR, thresholds, LLM params).
    # Imported lazily so .env load above takes effect first.
    from core.config import validate_config
    errors.extend(validate_config())

    if errors:
        for err in errors:
            logger.error(f"Startup: {err}")
        logger.error("Fix the above errors and restart.")
        sys.exit(1)

    logger.info("Startup validation passed")


async def test_grok():
    """Test Grok LLM connection."""
    from core.grok_llm import get_grok_llm
    from xai_sdk.chat import user as sdk_user

    print("Testing Grok connection...")
    llm = get_grok_llm()

    chat = llm.client.chat.create(
        model=llm.model,
        messages=[sdk_user("Say 'connected' if you can read this.")],
        max_tokens=50,
    )
    response = await chat.sample()

    print(f"Response: {response.content}")
    print("✓ Grok connected successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Grok Trader (xAI)")
    parser.add_argument("--test", action="store_true", help="Test Grok connection")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--no-research", action="store_true",
                        help="Don't pre-start the background scorer "
                             "(agent can still start it via research_engine tool)")
    parser.add_argument("--no-evolution", action="store_true",
                        help="Don't pre-start template evolution "
                             "(agent can still start it via research_engine tool)")
    daemon_group = parser.add_mutually_exclusive_group()
    daemon_group.add_argument(
        "--require-daemon", action="store_true",
        help="Refuse to start unless research_daemon.py is alive (fresh heartbeat). "
             "Use in production to guarantee the agent never silently runs the scorer in-process.",
    )
    daemon_group.add_argument(
        "--force-in-process", action="store_true",
        help="Always run the scorer in-process, even if the daemon heartbeat is fresh. "
             "Useful for dev/debug when you want one process to own everything.",
    )
    parser.add_argument(
        "--account",
        choices=["paper", "live"],
        default="paper",
        help="Trading account type (default: paper)",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.test:
        asyncio.run(test_grok())
        return

    os.environ["IBKR_ACCOUNT_TYPE"] = args.account
    if args.account == "live":
        logger.warning("*** LIVE TRADING MODE — real money ***")
        if not os.environ.get("TRADING_MODE"):
            os.environ["TRADING_MODE"] = "live"

    validate_startup()

    print(f"\nStarting Grok Trader (account={args.account})...")
    print("Press Ctrl+C to stop\n")

    from core.agent import run_agent

    if not args.no_research:
        # Three modes:
        #   --require-daemon   → hard-fail if daemon isn't alive (production)
        #   --force-in-process → always run scorer in-process (dev/debug)
        #   default            → detect daemon via heartbeat; fall back if stale
        from core.runtime.heartbeat import is_daemon_alive, heartbeat_age_s
        daemon_alive = is_daemon_alive()

        if args.require_daemon and not daemon_alive:
            age = heartbeat_age_s()
            age_str = "never" if age == float("inf") else f"{age:.1f}s"
            msg = (f"--require-daemon set but research_daemon heartbeat is stale "
                   f"(age={age_str}). Start research_daemon.py first, or drop the flag.")
            print(msg)
            logger.error(msg)
            sys.exit(2)

        if daemon_alive and not args.force_in_process:
            age = heartbeat_age_s()
            print(f"Research daemon detected (last heartbeat {age:.1f}s ago) — "
                  f"agent will not spawn its own scorer.\n")
            logger.info("Research daemon alive (age=%.1fs); skipping in-process scorer", age)
        else:
            from signals.scorer import run_research_threaded
            if args.force_in_process and daemon_alive:
                print("--force-in-process set: running scorer in-process despite live daemon "
                      "(this will double-write — use only for dev/debug)\n")
                logger.warning("--force-in-process: in-process scorer running alongside live daemon")
            else:
                print("Research scorer running in-process (no daemon heartbeat)\n")
            run_research_threaded(verbose=args.verbose)

    tasks = [run_agent()]
    if not args.no_evolution:
        from signals.template_evolution import run_template_evolution_threaded
        run_template_evolution_threaded()
        print("Template evolution running (agent can pause/stop via research_engine tool)\n")

    async def _heartbeat():
        """Log a heartbeat every 60s with each known component's last activity.

        Surfaces silent stalls — if the agent is hung in build_state_context
        or chat.sample(), nothing else logs from core.agent and the operator
        cannot tell whether the trader is alive.  Heartbeat keeps that visible.
        """
        import time as _time
        from core.agent import TradingAgent  # type: ignore  # noqa
        hb_logger = logging.getLogger("heartbeat")
        # Find the agent instance by walking gc — cheap; runs once a minute.
        import gc
        STALL_THRESHOLD = 300  # seconds
        while True:
            try:
                agent_obj = next(
                    (o for o in gc.get_objects()
                     if type(o).__name__ == "TradingAgent"),
                    None,
                )
                if agent_obj is not None:
                    last_step = getattr(agent_obj, "_last_step", "?")
                    last_ts = getattr(agent_obj, "_last_step_ts", _time.time())
                    age = _time.time() - last_ts
                    cycle = getattr(agent_obj, "_cycle_id", 0)
                    session = getattr(agent_obj, "_current_session", "?")
                    if age > STALL_THRESHOLD:
                        hb_logger.warning(
                            "STALL_DETECTED cycle=%d session=%s step=%s age=%.0fs",
                            cycle, session, last_step, age,
                        )
                    else:
                        hb_logger.info(
                            "HEARTBEAT cycle=%d session=%s step=%s age=%.0fs",
                            cycle, session, last_step, age,
                        )
                else:
                    hb_logger.info("HEARTBEAT no_agent_yet")
            except Exception as e:
                hb_logger.debug("heartbeat error: %s", e)
            await asyncio.sleep(60)

    def _loop_exception_handler(loop, context):
        msg = context.get("message", "unknown")
        exc = context.get("exception")
        logger.error("ASYNC_LOOP_EXCEPTION: %s", msg, exc_info=exc)

    async def _run_all():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_loop_exception_handler)
        all_tasks = tasks + [_heartbeat()]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error("Task crashed: %s", r, exc_info=r)

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
