#!/usr/bin/env python3
"""Poll status API and emit alerts (logs + optional webhook).

Used as a Docker sidecar or cron job:

  STATUS_API_URL=http://127.0.0.1:8790/status python scripts/alert_watch.py

Exit codes:
  0 — healthy or degraded (warnings only)
  1 — unhealthy (critical alerts)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env", override=False)
    except Exception:
        pass


def _fetch_report(url: str, timeout: float) -> dict[str, Any]:
    import httpx

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


def _post_webhook(webhook: str, payload: dict[str, Any]) -> None:
    import httpx

    with httpx.Client(timeout=10.0) as client:
        client.post(webhook, json=payload)


def _emit(report: dict[str, Any], *, webhook: str | None) -> int:
    overall = report.get("overall_status", "healthy")
    alerts = report.get("alerts") or []
    critical = [a for a in alerts if a.get("severity") == "critical"]
    warns = [a for a in alerts if a.get("severity") == "warn"]

    print(
        f"alert_watch: overall={overall} profile={report.get('active_profile')} "
        f"critical={len(critical)} warn={len(warns)}"
    )
    for a in critical + warns:
        print(f"  [{a.get('severity')}] {a.get('code')}: {a.get('message')} — {a.get('detail', '')}")

    if webhook and (critical or warns):
        try:
            _post_webhook(
                webhook,
                {
                    "text": f"ABC {overall}: {len(critical)} critical, {len(warns)} warn",
                    "report": {
                        "overall_status": overall,
                        "active_profile": report.get("active_profile"),
                        "alerts": critical + warns,
                    },
                },
            )
        except Exception as e:
            print(f"alert_watch: webhook failed: {e}", file=sys.stderr)

    if overall == "unhealthy":
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll ABC status API for alerts")
    parser.add_argument(
        "--url",
        default=os.getenv("STATUS_API_URL", "http://127.0.0.1:8790/status"),
        help="Status API URL (default STATUS_API_URL)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=float(os.getenv("ALERT_POLL_SECONDS", "0")),
        help="Poll every N seconds (0 = run once)",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--webhook",
        default=os.getenv("ALERT_WEBHOOK_URL", "").strip() or None,
        help="Optional JSON webhook (ALERT_WEBHOOK_URL)",
    )
    args = parser.parse_args()
    _load_env()

    if args.poll > 0:
        while True:
            try:
                report = _fetch_report(args.url, args.timeout)
                code = _emit(report, webhook=args.webhook)
                if code != 0:
                    sys.exit(code)
            except Exception as e:
                print(f"alert_watch: fetch failed: {e}", file=sys.stderr)
                sys.exit(1)
            time.sleep(args.poll)

    report = _fetch_report(args.url, args.timeout)
    raise SystemExit(_emit(report, webhook=args.webhook))


if __name__ == "__main__":
    main()
