#!/usr/bin/env python3
"""Daily profitability summary: dashboard + optimizer → tomorrow's ProfitConfig.

Runs:
  1. Dashboard (today's / N-day cycle logs)
  2. Live profile optimization (real logs, no backtest)
  3. Optional simulation grid optimizer (``--quick`` by default)
  4. Merged recommendation for tomorrow's ``PROFIT_PROFILE``

Notifications (optional env):
  DAILY_SUMMARY_SLACK_WEBHOOK or ALERT_WEBHOOK_URL
  DAILY_SUMMARY_EMAIL_TO, DAILY_SUMMARY_SMTP_HOST, DAILY_SUMMARY_SMTP_PORT,
  DAILY_SUMMARY_SMTP_USER, DAILY_SUMMARY_SMTP_PASSWORD, DAILY_SUMMARY_EMAIL_FROM

Examples::

    python scripts/daily_summary.py
    python scripts/daily_summary.py --days 7 --skip-sim
    python scripts/daily_summary.py --json-only -o data/daily_summary.json
    python scripts/daily_summary.py --notify

Schedule (cron / Task Scheduler after market close):

    python scripts/daily_summary.py --notify >> logs/daily_summary.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily ABC profitability summary")
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Dashboard anchor date when --dashboard-days 1 (default: today)",
    )
    parser.add_argument(
        "--dashboard-days",
        type=int,
        default=1,
        help="Cycle log window for dashboard section (default: 1)",
    )
    parser.add_argument(
        "--live-days",
        type=int,
        default=7,
        help="Lookback for live profile ranking (default: 7)",
    )
    parser.add_argument(
        "--sim-days",
        type=int,
        default=7,
        help="Simulation optimizer calendar lookback (default: 7)",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Skip historical simulation optimizer (faster)",
    )
    parser.add_argument(
        "--no-quick",
        action="store_true",
        help="Simulation: include perturbation candidates (slower)",
    )
    parser.add_argument(
        "--baseline",
        default="balanced",
        choices=("conservative", "balanced", "aggressive"),
        help="Simulation baseline profile (default: balanced)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="JSON output path (default: data/daily_summary_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print JSON to stdout only (no console report)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Slack/email when env vars are configured",
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO / ".env", override=True)
    except ImportError:
        pass

    from core.daily_summary import (
        deliver_notifications,
        format_daily_summary_report,
        run_daily_summary,
        save_daily_summary_json,
    )

    report = run_daily_summary(
        dashboard_days=max(1, args.dashboard_days),
        dashboard_date=args.date,
        live_lookback_days=max(1, args.live_days),
        sim_days=max(1, args.sim_days),
        run_sim=not args.skip_sim,
        sim_quick=not args.no_quick,
        sim_baseline=args.baseline,
    )

    out = save_daily_summary_json(
        report,
        Path(args.output) if args.output else None,
    )
    report["output_path"] = str(out)

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_daily_summary_report(report))
        print(f"\nWrote JSON: {out}")

    if args.notify:
        notes = deliver_notifications(report)
        if notes:
            print("Notifications:", ", ".join(notes))
        else:
            print(
                "Notifications: none configured "
                "(set DAILY_SUMMARY_SLACK_WEBHOOK or DAILY_SUMMARY_EMAIL_TO)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
