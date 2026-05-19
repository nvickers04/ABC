"""
Grok Trader — Entry Point

Usage:
    python __main__.py                  # Run agent (paper trading)
    python __main__.py --test           # Test Grok connection
    python __main__.py --verbose        # Debug logging
    python __main__.py --account live   # Live trading (use with caution)

Research (scoring + template evolution) lives in ``python -m research``. This process
only starts an in-process scorer when the research host heartbeat is absent/stale (unless
``TRADER_IN_PROCESS_SCORER=never`` / ``--require-research-host``, which forbid that fallback).

Built for Grok (xAI) — dynamic liquidity, overnight holds OK, configurable risk.
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Force UTF-8 on stdio so non-ASCII chars in log messages (em-dash, ≤, ≥, ×, √,
# box-drawing, emoji) don't blow up on Windows cp1252 consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with console + rotating file handler."""
    from core.log_setup import configure_root_logging

    configure_root_logging("agent.log", verbose=verbose)


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
    research_host_group = parser.add_mutually_exclusive_group()
    research_host_group.add_argument(
        "--require-daemon",
        "--require-research-host",
        action="store_true",
        dest="require_research_host",
        help="Refuse to start unless the research host is alive (fresh heartbeat). "
             "Alias: --require-daemon (legacy). Same effect as TRADER_IN_PROCESS_SCORER=never "
             "for this flag alone. Use in production so the trader never silently runs "
             "the scorer in-process.",
    )
    research_host_group.add_argument(
        "--force-in-process", action="store_true",
        help="Always run the scorer in-process, even if the research host heartbeat is fresh. "
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
        # Match validate_startup: load .env before reading GROK_API_KEY / XAI_API_KEY.
        from dotenv import load_dotenv

        load_dotenv(override=True)
        asyncio.run(test_grok())
        return

    os.environ["IBKR_ACCOUNT_TYPE"] = args.account
    if args.account == "live":
        logger.warning("*** LIVE TRADING MODE — real money ***")
        if not os.environ.get("TRADING_MODE"):
            os.environ["TRADING_MODE"] = "live"

    validate_startup()

    from core import config as _budget_cfg

    print(
        "\nLLM / API spend guardrails (this app’s estimates — xAI console may differ):\n"
        f"  • MAX_DAILY_LLM_COST ≈ ${_budget_cfg.MAX_DAILY_LLM_COST:.2f} tracked USD/day → agent halts\n"
        f"  • research() cap ≈ ${_budget_cfg.MAX_DAILY_MULTI_AGENT_RESEARCH_USD:.2f} tracked USD/day\n"
        "  • Daily token caps per bucket — see core/config.py / .env\n"
        "  Match these to your prepaid balance; export MAX_DAILY_LLM_COST=… if needed.\n"
    )
    logger.info(
        "Budget defaults: MAX_DAILY_LLM_COST=%s MAX_DAILY_MULTI_AGENT_RESEARCH_USD=%s",
        _budget_cfg.MAX_DAILY_LLM_COST,
        _budget_cfg.MAX_DAILY_MULTI_AGENT_RESEARCH_USD,
    )

    print(f"\n=== Grok Trader ({args.account}) ===")
    print("This process: Grok agent, IBKR, memory — not template evolution.")
    print("Template evolution runs only in python -m research (default there).")
    print("Press Ctrl+C to stop.\n")

    from core.agent import run_agent

    if not args.no_research:
        # --require-research-host / TRADER_IN_PROCESS_SCORER=never → hard-fail if no fresh heartbeat
        # --force-in-process → always run scorer in-process (dev/debug; wins over env gate)
        # default (auto) → fresh research host heartbeat → skip in-process scorer
        from core.runtime.heartbeat import heartbeat_age_s, is_research_host_alive
        from core.config import TRADER_IN_PROCESS_SCORER_NEVER

        research_host_alive = is_research_host_alive()
        require_research_host = bool(
            args.require_research_host or TRADER_IN_PROCESS_SCORER_NEVER
        )

        if args.force_in_process:
            from signals.scorer import run_research_threaded
            if research_host_alive:
                print(
                    "Scoring: in-process (--force-in-process; double-writes vs research host — "
                    "dev/debug only).\n"
                )
                logger.warning("--force-in-process: in-process scorer running alongside live research host")
            else:
                print(
                    "Scoring: in-process (--force-in-process; no research host heartbeat).\n"
                )
            run_research_threaded(verbose=args.verbose)
        elif require_research_host and not research_host_alive:
            age = heartbeat_age_s()
            age_str = "never" if age == float("inf") else f"{age:.1f}s"
            hint = ""
            if TRADER_IN_PROCESS_SCORER_NEVER and not args.require_research_host:
                hint = " (TRADER_IN_PROCESS_SCORER=never in .env)"
            msg = (
                f"Research scorer unavailable: no fresh research host heartbeat "
                f"(age={age_str}){hint}. Start ``python -m research`` on the research host, "
                f"or pass --force-in-process for dev only."
            )
            if args.require_research_host and not TRADER_IN_PROCESS_SCORER_NEVER:
                msg = (
                    f"--require-research-host set but heartbeat is stale "
                    f"(age={age_str}). Start python -m research first, or drop the flag."
                )
            print(msg)
            logger.error(msg)
            sys.exit(2)
        elif research_host_alive:
            age = heartbeat_age_s()
            print(
                f"Scoring: remote (research host heartbeat {age:.0f}s ago) — "
                "no in-process scorer here.\n"
            )
            logger.info("Research host alive (age=%.1fs); skipping in-process scorer", age)
        else:
            from signals.scorer import run_research_threaded
            print(
                "Scoring: in-process (no fresh research host heartbeat). "
                "Prefer running python -m research for production.\n"
            )
            run_research_threaded(verbose=args.verbose)
    else:
        print(
            "Scoring: not auto-started (--no-research). "
            "Use python -m research or research_engine tool for scorer only.\n"
        )

    tasks = [run_agent()]

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
