"""Shared argparse definitions for trader and research entry points.

Keeps ``__main__.py`` and ``research/host.py`` flags, help text, and legacy aliases
in one place. See ``docs/entry-points.md`` for when to use each command.
"""

from __future__ import annotations

import argparse
from typing import Sequence

# ── Help epilogs (shown after option list) ───────────────────────────────────

TRADER_EPILOG = """\
examples:
  python __main__.py --test
  python __main__.py --verbose
  python __main__.py --require-research-host --verbose
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

See docs/entry-points.md for split-host vs single-machine workflows.
"""

RESEARCH_EPILOG = """\
examples:
  python -m research
  python -m research --verbose
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

    return parser


def parse_research_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse research host CLI arguments."""
    return build_research_parser().parse_args(argv)
