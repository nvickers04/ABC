"""Shared argparse definitions for trader and research entry points.

Keeps ``__main__.py`` and ``research/host.py`` flags, help text, and legacy aliases
in one place. See ``docs/entry-points.md`` for when to use each command.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from core.profile_ab_test import AB_MODES

# ── Help epilogs (shown after option list) ───────────────────────────────────

TRADER_EPILOG = """\
examples:
  python __main__.py --test
  python __main__.py --verbose
  python __main__.py --simulate conservative,balanced,aggressive --sim-start 2024-01-02 --sim-end 2024-03-01
  python __main__.py --require-research-host --verbose
  python __main__.py --live-optimize
  python __main__.py --ab-test conservative,balanced --ab-duration-days 30
  TRADER_IN_PROCESS_SCORER=never python __main__.py --verbose

scoring policy (pick at most one of --require-research-host / --force-in-process):
  default          Start in-process scorer only if research host heartbeat is stale
  --require-research-host
                   Exit if heartbeat is stale (production trader; split host)
  --force-in-process
                   Always run scorer in this process (dev only; may double-write)
  --no-research    Never auto-start scorer (use python -m research or research_engine)

environment:
  TRADER_IN_PROCESS_SCORER=never
                   Same hard gate as --require-research-host (also: 0, false, off, no)
  IBKR_ACCOUNT_TYPE / --account   paper (default) or live
  XAI_API_KEY or GROK_API_KEY     Required unless --test

See docs/entry-points.md and docs/simulation-and-optimization.md for split-host,
--simulate, and ProfitConfig profiles.
"""

RESEARCH_EPILOG = """\
examples:
  python -m research
  python -m research --verbose
  python -m research --profile balanced
  python -m research --no-evolution

notes:
  Writes research_host_heartbeat_ts each scoring round (trader reads this).
  Never run on the trader machine in production — use python __main__.py there.

See docs/entry-points.md.
"""


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Defaults in help lines plus raw epilog (preserves newlines)."""


def add_profit_profile_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--profit-profile`` / ``--profile`` (same as trader ``PROFIT_PROFILE``)."""
    parser.add_argument(
        "--profit-profile",
        "--profile",
        dest="profit_profile",
        default=None,
        metavar="PROFILE",
        help=(
            "ProfitConfig profile for this process (conservative | balanced | aggressive "
            "or evolved name in data/evolved_profiles.json). Sets PROFIT_PROFILE before load."
        ),
    )


def build_trader_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python __main__.py`` (Grok trader)."""
    parser = argparse.ArgumentParser(
        prog="python __main__.py",
        description=(
            "Grok trader (xAI + IBKR). Runs the agent loop; does not evolve templates. "
            "Research scoring defaults to the remote research host when its heartbeat is fresh."
        ),
        formatter_class=_HelpFormatter,
        epilog=TRADER_EPILOG,
    )

    general = parser.add_argument_group("general")
    general.add_argument(
        "--test",
        action="store_true",
        help="Test Grok API connectivity and exit (no IBKR, no agent loop).",
    )
    general.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging (console and logs/agent.log).",
    )
    general.add_argument(
        "--account",
        choices=["paper", "live"],
        default="paper",
        metavar="MODE",
        help="IBKR account mode (default: %(default)s). Live sets TRADING_MODE=live.",
    )
    general.add_argument(
        "--trading-mode",
        choices=["aggressive_paper", "paper", "live"],
        default=None,
        metavar="MODE",
        help="Override TRADING_MODE (aggressive_paper | paper | live).",
    )
    general.add_argument(
        "--risk-per-trade",
        type=float,
        default=None,
        metavar="PCT",
        help="Override RISK_PER_TRADE (percent of cash, e.g. 1.0 = 1%%).",
    )
    general.add_argument(
        "--max-daily-loss-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="Override MAX_DAILY_LOSS_PCT session loss flatten threshold.",
    )
    general.add_argument(
        "--max-llm-cost",
        type=float,
        default=None,
        metavar="USD",
        help="Override MAX_DAILY_LLM_COST (USD/day).",
    )
    general.add_argument(
        "--cash-only",
        action="store_true",
        default=None,
        help="Force CASH_ONLY=true (no margin / short stock).",
    )
    general.add_argument(
        "--allow-margin",
        action="store_true",
        help="Set CASH_ONLY=false (use only when you understand margin risk).",
    )
    general.add_argument(
        "--config-summary",
        action="store_true",
        help="Print ProfitConfig summary (all levers + P&L notes) and exit.",
    )
    add_profit_profile_argument(general)
    general.add_argument(
        "--live-optimize",
        action="store_true",
        help=(
            "Analyze last N days of real profit cycle logs and print the best "
            "PROFIT_PROFILE for tomorrow (lightweight; no backtest). Exits."
        ),
    )
    general.add_argument(
        "--live-optimize-days",
        type=int,
        default=7,
        metavar="N",
        help="Lookback for --live-optimize (default: %(default)s calendar days).",
    )
    general.add_argument(
        "--live-optimize-output",
        metavar="PATH",
        default=None,
        help="Write --live-optimize JSON to PATH (default: data/live_profile_suggestion.json).",
    )

    ab = parser.add_argument_group(
        "profile a/b test",
        description=(
            "Paper-only live comparison of two ProfitConfig profiles. One IBKR connection; "
            "profiles alternate per cycle (rotation) or per session day (session). "
            "Comparative P&L via profit cycle logs."
        ),
    )
    ab.add_argument(
        "--ab-test",
        metavar="PROFILE_A,PROFILE_B",
        default=None,
        help=(
            "Run paper A/B test between two profiles (comma-separated, exactly two). "
            "Exits normal single-profile trader mode."
        ),
    )
    ab.add_argument(
        "--ab-duration-days",
        type=int,
        default=30,
        metavar="N",
        help="Stop A/B test after N calendar days (default: %(default)s).",
    )
    ab.add_argument(
        "--ab-mode",
        choices=AB_MODES,
        default="rotation",
        help="rotation: switch profile each cycle; session: one profile per calendar day.",
    )

    sim = parser.add_argument_group(
        "simulation",
        description="Historical backtest using the real ReAct loop (no IBKR, no live xAI).",
    )
    sim.add_argument(
        "--simulate",
        nargs="?",
        const="balanced",
        metavar="PROFILES",
        help=(
            "Run backtest simulation (requires --sim-start/--sim-end). "
            "PROFILES: one name or comma-separated list, e.g. "
            "conservative,balanced,aggressive (default with bare --simulate: balanced)."
        ),
    )
    sim.add_argument(
        "--sim-start",
        metavar="YYYY-MM-DD",
        default=None,
        help="Backtest start date (inclusive).",
    )
    sim.add_argument(
        "--sim-end",
        metavar="YYYY-MM-DD",
        default=None,
        help="Backtest end date (inclusive).",
    )
    sim.add_argument(
        "--sim-cash",
        type=float,
        default=100_000.0,
        metavar="USD",
        help="Initial simulated equity (default: %(default)s).",
    )
    from core.profile_optimization import DEFAULT_CYCLES_PER_DAY

    sim.add_argument(
        "--sim-cycles-per-day",
        "--cycles-per-day",
        type=int,
        default=DEFAULT_CYCLES_PER_DAY,
        dest="sim_cycles_per_day",
        metavar="N",
        help=(
            "Agent cycles per trading day, max 4 (default: %(default)s). "
            "Alias: --cycles-per-day."
        ),
    )
    sim.add_argument(
        "--sim-csv",
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "Write per-trade CSV (timestamps, ticker, entry/exit, realized R:R, profile). "
            "Default path: data/sim_exports/backtest_<profiles>_<start>_<end>.csv"
        ),
    )

    scoring = parser.add_argument_group(
        "scoring",
        description=(
            "How the trader relates to signals/scorer.py. At most one of "
            "--require-research-host and --force-in-process."
        ),
    )
    host_policy = scoring.add_mutually_exclusive_group()
    host_policy.add_argument(
        "--require-research-host",
        "--require-daemon",
        action="store_true",
        dest="require_research_host",
        help=(
            "Refuse to start if the research host heartbeat is stale. "
            "Legacy alias: --require-daemon."
        ),
    )
    host_policy.add_argument(
        "--force-in-process",
        action="store_true",
        help=(
            "Always run the scorer in this process even when the research host is alive "
            "(dev/debug only; risks duplicate writes with python -m research)."
        ),
    )
    scoring.add_argument(
        "--no-research",
        action="store_true",
        help=(
            "Do not auto-start the background scorer. Use python -m research or the "
            "research_engine tool for scoring."
        ),
    )

    return parser


def parse_simulate_profiles(simulate_arg: str | None) -> list[str]:
    """Parse ``--simulate`` value into validated profile names (deduped, stable order)."""
    from core.profit_profiles import normalize_profit_profile

    raw = (simulate_arg or "balanced").strip()
    if not raw:
        raw = "balanced"
    seen: set[str] = set()
    profiles: list[str] = []
    for part in raw.split(","):
        name = normalize_profit_profile(part.strip())
        if name not in seen:
            seen.add(name)
            profiles.append(name)
    return profiles


def apply_profit_profile_cli_to_environ(args: argparse.Namespace) -> None:
    """Set ``PROFIT_PROFILE`` when ``--profit-profile`` or ``--profile`` is passed."""
    import os

    from core.profit_config_state import get_active_profile_label
    from core.profit_profiles import PROFIT_PROFILE_ENV, normalize_profit_profile

    profile = getattr(args, "profit_profile", None)
    if profile is not None:
        previous = get_active_profile_label()
        new_profile = normalize_profit_profile(profile)
        os.environ[PROFIT_PROFILE_ENV] = new_profile
        try:
            from core.profile_rollback import on_profile_applied

            on_profile_applied(previous, str(new_profile))
        except Exception:
            pass


def apply_trader_cli_to_environ(args: argparse.Namespace) -> None:
    """Push recognized CLI overrides into ``os.environ`` before config reload."""
    import os

    apply_profit_profile_cli_to_environ(args)
    if getattr(args, "trading_mode", None):
        os.environ["TRADING_MODE"] = str(args.trading_mode)
    if getattr(args, "account", None) == "live":
        os.environ["IBKR_ACCOUNT_TYPE"] = "live"
        if not os.environ.get("TRADING_MODE"):
            os.environ["TRADING_MODE"] = "live"
    elif getattr(args, "account", None) == "paper":
        os.environ["IBKR_ACCOUNT_TYPE"] = "paper"
    if getattr(args, "risk_per_trade", None) is not None:
        os.environ["RISK_PER_TRADE"] = str(args.risk_per_trade)
    if getattr(args, "max_daily_loss_pct", None) is not None:
        os.environ["MAX_DAILY_LOSS_PCT"] = str(args.max_daily_loss_pct)
    if getattr(args, "max_llm_cost", None) is not None:
        os.environ["MAX_DAILY_LLM_COST"] = str(args.max_llm_cost)
    if getattr(args, "cash_only", None) is True:
        os.environ["CASH_ONLY"] = "true"
    if getattr(args, "allow_margin", False):
        os.environ["CASH_ONLY"] = "false"
    if getattr(args, "require_research_host", False):
        os.environ["TRADER_IN_PROCESS_SCORER"] = "never"


def run_ab_test_cli_entry(args: argparse.Namespace) -> int:
    """Run paper A/B profile test loop."""
    from core.profile_ab_test import run_ab_test_cli

    return run_ab_test_cli(args)


def run_live_optimize_cli(args: argparse.Namespace) -> int:
    """Run nightly live-log profile suggestion and print report."""
    import json
    from pathlib import Path

    from core.live_profile_optimize import format_live_optimize_report, run_live_optimize

    days = max(1, int(getattr(args, "live_optimize_days", 7) or 7))
    result = run_live_optimize(days=days)
    print(format_live_optimize_report(result))

    out = getattr(args, "live_optimize_output", None)
    if out is None:
        out = "data/live_profile_suggestion.json"
    path = Path(out)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON: {path}")
    return 0


def parse_trader_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse trader CLI arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    return build_trader_parser().parse_args(argv)


def build_research_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m research`` (research host)."""
    parser = argparse.ArgumentParser(
        prog="python -m research",
        description=(
            "Research host: market scoring + optional template evolution. "
            "No Grok agent, no IBKR orders. Writes heartbeat for the trader."
        ),
        formatter_class=_HelpFormatter,
        epilog=RESEARCH_EPILOG,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging (logs/research.log).",
    )
    parser.add_argument(
        "--no-evolution",
        action="store_true",
        help="Do not start the template-evolution background thread.",
    )
    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Print ProfitConfig summary (all levers + P&L notes) and exit.",
    )
    add_profit_profile_argument(parser)

    return parser


def load_profit_config(*, dotenv: bool = True):
    """Return the process :class:`ProfitConfig` singleton (reloads from env / CLI)."""
    from core.central_profit_config import get_profit_config

    return get_profit_config().reload(dotenv=dotenv)


def parse_research_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse research host CLI arguments."""
    return build_research_parser().parse_args(argv)
