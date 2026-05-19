#!/usr/bin/env python3
"""Health checks for research host and trader operating mode.

Runs Postgres, research heartbeat, researcher token cap, MDA quote probe,
ProfitConfig observability (safety, cycle logs, alerts), and role-specific checks.

Usage (from repo root):

  python scripts/health.py
  python scripts/health.py trader --json
  python scripts/health.py --status-api-url http://127.0.0.1:8790/status

APIs: ``python -m api`` (8787), ``python -m infra.status_api`` (8790)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
for _p in (_ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from health_common import (  # noqa: E402
    Reporter,
    apply_health_report_to_reporter,
    check_ibkr_async,
    check_required_tables,
    load_dotenv_repo,
    run_platform_checks,
)


def _exit_from_status(overall: str) -> int:
    if overall == "unhealthy":
        return 2
    if overall == "degraded":
        return 1
    return 0


def _fetch_status_api(url: str) -> dict:
    import httpx

    with httpx.Client(timeout=8.0) as client:
        resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="ABC health checks")
    parser.add_argument(
        "target",
        nargs="?",
        choices=("researcher", "trader", "all"),
        default="all",
    )
    parser.add_argument("--ibkr-client-id", type=int, default=None)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--json", action="store_true", help="Full JSON health report")
    parser.add_argument("--profit-api-url", default=None)
    parser.add_argument(
        "--status-api-url",
        default=None,
        help="Use remote status API instead of local report builder",
    )
    parser.add_argument("--skip-profit-api", action="store_true")
    args = parser.parse_args()

    load_dotenv_repo(_ROOT)
    role = args.target
    use_color = False if args.no_color else None

    status_url = args.status_api_url or os.getenv("STATUS_API_URL", "").strip()
    if args.json or status_url:
        url = status_url or "http://127.0.0.1:8790/status"
        if status_url or args.json:
            try:
                if status_url:
                    report = _fetch_status_api(url)
                else:
                    from core.observability.health_report import build_health_report

                    report = build_health_report(role=role)
            except Exception as e:
                if args.json:
                    print(json.dumps({"error": str(e)}))
                else:
                    rep = Reporter("ABC Health", use_color=use_color)
                    rep.fail("Status API", str(e))
                    rep.print_report()
                raise SystemExit(2) from e
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                rep = Reporter("ABC Health (status API)", use_color=use_color)
                apply_health_report_to_reporter(rep, report)
                rep.print_report()
            raise SystemExit(_exit_from_status(str(report.get("overall_status", "healthy"))))

    code = 0
    if role in ("researcher", "all"):
        rep = Reporter("ABC Research Host Health", use_color=use_color)
        if run_platform_checks(
            rep,
            role="researcher",
            include_mda=True,
            include_scoring=True,
            include_evolution=True,
            include_profit_api=not args.skip_profit_api,
            profit_api_url=args.profit_api_url,
        ):
            check_required_tables(rep)
        rep.print_report()
        code = max(code, rep.exit_code())

    if role in ("trader", "all"):
        rep = Reporter("ABC Trader Health", use_color=use_color)
        if run_platform_checks(
            rep,
            role="trader",
            include_mda=True,
            include_trader_context=True,
            include_profit_api=not args.skip_profit_api,
            profit_api_url=args.profit_api_url,
        ):
            check_required_tables(rep)
        if args.ibkr_client_id is not None:
            asyncio.run(check_ibkr_async(rep, client_id=args.ibkr_client_id))
        else:
            rep.skip("IBKR gateway", "add --ibkr-client-id N to test TWS/Gateway")
        rep.print_report()
        code = max(code, rep.exit_code())

    raise SystemExit(code)


if __name__ == "__main__":
    main()
