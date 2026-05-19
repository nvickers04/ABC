"""
Grok Trader — Entry Point

Run: ``python __main__.py`` (see ``python __main__.py --help``).

Research scoring and template evolution: ``python -m research``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Force UTF-8 on stdio so non-ASCII chars in log messages don't blow up on Windows.
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


def validate_startup() -> None:
    """Validate critical secrets before launch."""
    from dotenv import load_dotenv

    load_dotenv(override=True)

    errors: list[str] = []

    grok_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not grok_key:
        errors.append("Missing GROK_API_KEY or XAI_API_KEY")

    from core.config import validate_config

    errors.extend(validate_config())

    if errors:
        for err in errors:
            logger.error("Startup: %s", err)
        logger.error("Fix the above errors and restart.")
        sys.exit(1)

    logger.info("Startup validation passed")


async def test_grok() -> None:
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


def _configure_scorer(args) -> None:
    """Start or skip in-process scorer based on CLI flags and heartbeat."""
    if args.no_research:
        print(
            "Scoring: not auto-started (--no-research). "
            "Use python -m research or research_engine tool for scorer only.\n"
        )
        return

    from core.entry_cli import load_profit_config

    profit = load_profit_config(dotenv=False)
    trader_in_process_scorer_never = profit.risk.trader_in_process_scorer_never
    from core.runtime.heartbeat import (
        heartbeat_age_s,
        is_research_host_alive,
        is_research_host_operational,
        read_research_host_status,
    )
    from signals.scorer import run_research_threaded

    research_host_alive = is_research_host_operational()
    heartbeat_fresh = is_research_host_alive()
    require_research_host = bool(
        args.require_research_host or trader_in_process_scorer_never
    )

    if args.force_in_process:
        if research_host_alive:
            print(
                "Scoring: in-process (--force-in-process; double-writes vs research host — "
                "dev/debug only).\n"
            )
            logger.warning(
                "--force-in-process: in-process scorer running alongside live research host"
            )
        else:
            print(
                "Scoring: in-process (--force-in-process; no research host heartbeat).\n"
            )
        run_research_threaded(verbose=args.verbose)
        return

    if require_research_host and not research_host_alive:
        age = heartbeat_age_s()
        age_str = "never" if age == float("inf") else f"{age:.1f}s"
        status = read_research_host_status()
        hint = ""
        if trader_in_process_scorer_never and not args.require_research_host:
            hint = " (TRADER_IN_PROCESS_SCORER=never in .env)"
        msg = (
            f"Research scorer unavailable: research host not operational "
            f"(heartbeat_age={age_str}, status={status:.0f}){hint}. "
            f"Start ``python -m research`` on the research host, "
            f"or pass --force-in-process for dev only."
        )
        if args.require_research_host and not trader_in_process_scorer_never:
            msg = (
                f"--require-research-host set but heartbeat is stale "
                f"(age={age_str}). Start python -m research first, or drop the flag."
            )
        print(msg)
        logger.error(msg)
        sys.exit(2)

    if research_host_alive:
        age = heartbeat_age_s()
        print(
            f"Scoring: remote (research host operational, heartbeat {age:.0f}s ago) — "
            "no in-process scorer here.\n"
        )
        logger.info(
            "Research host operational (age=%.1fs status=%.0f); skipping in-process scorer",
            age,
            read_research_host_status(),
        )
        return

    if heartbeat_fresh and not research_host_alive:
        print(
            "Scoring: in-process (research host heartbeat fresh but not operational "
            "(shutting down or cap stopped)).\n"
        )
    else:
        print(
            "Scoring: in-process (no fresh research host heartbeat). "
            "Prefer running python -m research for production.\n"
        )
    run_research_threaded(verbose=args.verbose)


def main() -> None:
    """Main entry point."""
    from core.entry_cli import apply_trader_cli_to_environ, load_profit_config, parse_trader_args

    args = parse_trader_args()
    apply_trader_cli_to_environ(args)
    profit = load_profit_config(dotenv=not args.test)

    if getattr(args, "config_summary", False):
        profit.summary()
        return

    setup_logging(verbose=args.verbose)

    if args.test:
        asyncio.run(test_grok())
        return

    if args.account == "live" or profit.trading_mode == "live":
        logger.warning("*** LIVE TRADING MODE — real money ***")
    validate_startup()

    print(
        "\nLLM / API spend guardrails (this app’s estimates — xAI console may differ):\n"
        f"  • MAX_DAILY_LLM_COST ≈ ${profit.risk.max_daily_llm_cost:.2f} tracked USD/day → agent halts\n"
        f"  • research() cap ≈ ${profit.risk.max_daily_multi_agent_research_usd:.2f} tracked USD/day\n"
        "  • Daily token caps per bucket — run with --config-summary for full lever list\n"
        "  Match these to your prepaid balance; export MAX_DAILY_LLM_COST=… if needed.\n"
    )
    logger.info(
        "Budget defaults: MAX_DAILY_LLM_COST=%s MAX_DAILY_MULTI_AGENT_RESEARCH_USD=%s",
        profit.risk.max_daily_llm_cost,
        profit.risk.max_daily_multi_agent_research_usd,
    )

    print(f"\n=== Grok Trader ({args.account}) ===")
    print("This process: Grok agent, IBKR, memory — not template evolution.")
    print("Template evolution runs only in python -m research (default there).")
    print("Press Ctrl+C to stop.\n")

    _configure_scorer(args)

    from core.agent import run_agent

    tasks = [run_agent()]

    async def _heartbeat() -> None:
        """Log a heartbeat every 60s with each known component's last activity."""
        import gc
        import time as _time

        hb_logger = logging.getLogger("heartbeat")
        stall_threshold = 300
        while True:
            try:
                agent_obj = next(
                    (o for o in gc.get_objects() if type(o).__name__ == "TradingAgent"),
                    None,
                )
                if agent_obj is not None:
                    last_step = getattr(agent_obj, "_last_step", "?")
                    last_ts = getattr(agent_obj, "_last_step_ts", _time.time())
                    age = _time.time() - last_ts
                    cycle = getattr(agent_obj, "_cycle_id", 0)
                    session = getattr(agent_obj, "_current_session", "?")
                    if age > stall_threshold:
                        hb_logger.warning(
                            "STALL_DETECTED cycle=%d session=%s step=%s age=%.0fs",
                            cycle,
                            session,
                            last_step,
                            age,
                        )
                    else:
                        hb_logger.info(
                            "HEARTBEAT cycle=%d session=%s step=%s age=%.0fs",
                            cycle,
                            session,
                            last_step,
                            age,
                        )
                else:
                    hb_logger.info("HEARTBEAT no_agent_yet")
            except Exception as e:
                hb_logger.debug("heartbeat error: %s", e)
            await asyncio.sleep(60)

    def _loop_exception_handler(loop, context) -> None:
        msg = context.get("message", "unknown")
        exc = context.get("exception")
        logger.error("ASYNC_LOOP_EXCEPTION: %s", msg, exc_info=exc)

    async def _run_all() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_loop_exception_handler)
        results = await asyncio.gather(*tasks, _heartbeat(), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error("Task crashed: %s", r, exc_info=r)

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
