#!/usr/bin/env python3
"""Health checks for research host and trader operating mode.

Runs Postgres, research heartbeat, researcher token cap, MDA quote probe,
and role-specific checks. Colored output; exit codes:

  0 — healthy
  1 — degraded (warnings only)
  2 — unhealthy (one or more failures)

Usage (from repo root):

  python scripts/health.py                 # researcher + trader sections
  python scripts/health.py researcher
  python scripts/health.py trader
  python scripts/health.py trader --ibkr-client-id 11
  python scripts/health.py all --no-color
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
for _p in (_ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from health_common import (  # noqa: E402
    Reporter,
    check_ibkr_async,
    check_required_tables,
    load_dotenv_repo,
    run_platform_checks,
)


def _check_researcher(*, use_color: bool | None) -> int:
    rep = Reporter("ABC Research Host Health", use_color=use_color)
    if run_platform_checks(
        rep,
        role="researcher",
        include_mda=True,
        include_scoring=True,
        include_evolution=True,
    ):
        check_required_tables(rep)
    rep.print_report()
    return rep.exit_code()


def _check_trader(*, use_color: bool | None, ibkr_client_id: int | None) -> int:
    rep = Reporter("ABC Trader Health", use_color=use_color)
    if run_platform_checks(
        rep,
        role="trader",
        include_mda=True,
        include_trader_context=True,
    ):
        check_required_tables(rep)

    if ibkr_client_id is not None:
        asyncio.run(check_ibkr_async(rep, client_id=ibkr_client_id))
    else:
        rep.skip("IBKR gateway", "add --ibkr-client-id N to test TWS/Gateway")

    rep.print_report()
    return rep.exit_code()


def main() -> None:
    parser = argparse.ArgumentParser(description="ABC health checks")
    parser.add_argument(
        "target",
        nargs="?",
        choices=("researcher", "trader", "all"),
        default="all",
        help="Which check to run (default: all)",
    )
    parser.add_argument(
        "--ibkr-client-id",
        type=int,
        default=None,
        help="Trader only: connect to IBKR and verify gateway (optional)",
    )
    parser.add_argument("--no-color", action="store_true", help="Plain text output")
    args = parser.parse_args()

    load_dotenv_repo(_ROOT)
    use_color = False if args.no_color else None
    code = 0
    if args.target in ("researcher", "all"):
        code = max(code, _check_researcher(use_color=use_color))
    if args.target in ("trader", "all"):
        code = max(code, _check_trader(use_color=use_color, ibkr_client_id=args.ibkr_client_id))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
