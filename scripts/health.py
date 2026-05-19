#!/usr/bin/env python3
"""Health checks for research daemon and trader operating mode.

  python scripts/health.py researcher
  python scripts/health.py trader
  python scripts/health.py all          # default
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _check_researcher() -> int:
    from core.config import RESEARCHER_DAILY_TOKEN_CAP
    from core.runtime.heartbeat import heartbeat_age_s, is_daemon_alive
    from memory import get_db, get_research_config

    print("=== ABC Researcher Health ===\n")
    alive = is_daemon_alive()
    age = heartbeat_age_s()
    print(f"Daemon heartbeat alive:     {alive}")
    print(f"Heartbeat age:              {round(age, 1)} seconds")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage_key = f"researcher_daily_usage_{today}"
    usage = float(get_research_config(usage_key, 0.0))
    pct = (usage / RESEARCHER_DAILY_TOKEN_CAP) * 100 if RESEARCHER_DAILY_TOKEN_CAP > 0 else 0
    print(f"Today's researcher usage:   {usage:,.0f} / {RESEARCHER_DAILY_TOKEN_CAP:,} ({pct:.1f}%)")

    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT MAX(ts) as last_ts, COUNT(*) as count
            FROM signal_scores
            WHERE DATE(ts, 'unixepoch') = DATE('now')
            """
        ).fetchone()
        if row and row["last_ts"]:
            last = datetime.fromtimestamp(row["last_ts"], tz=timezone.utc)
            print(f"Last scoring round:         {last} ({row['count']} scores today)")
        else:
            print("Last scoring round:         No activity today")
    except Exception as e:
        print(f"Could not query signal_scores: {e}")

    try:
        last_evo = get_research_config("last_template_evolution_round", 0)
        if last_evo > 0:
            print(f"Last evolution round:       {datetime.fromtimestamp(last_evo, tz=timezone.utc)}")
        else:
            print("Last evolution round:       No record")
    except Exception:
        print("Last evolution round:       Unable to read")

    if not alive or age > 600:
        status = "DOWN or STALLED"
        code = 1
    elif pct >= 100:
        status = "CAP EXCEEDED"
        code = 1
    elif pct > 85:
        status = "HEALTHY (approaching cap)"
        code = 0
    else:
        status = "HEALTHY"
        code = 0

    print(f"\nOverall: {status}\n")
    return code


def _check_trader() -> int:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)

    from core.runtime.heartbeat import heartbeat_age_s, is_daemon_alive
    from core.runtime.local_memory_fallback import LOCAL_MEMORY_FILE
    from core.runtime.operating_context import get_operating_context

    print("=== ABC Trader Status ===\n")
    ctx = get_operating_context()
    try:
        researcher_alive = is_daemon_alive()
        hb_age = heartbeat_age_s()
    except Exception:
        researcher_alive = False
        hb_age = float("inf")

    mode = "INDEPENDENT" if (not researcher_alive or ctx.is_independent_mode) else "FULL"
    print(f"Mode:                   {mode}")
    print(f"Researcher available:   {ctx.quality.researcher_available}")
    print(f"Memory source:          {ctx.quality.memory_source}")
    print(f"Risk multiplier:        {ctx.risk_multiplier}")
    print(f"WM completeness:        {ctx.quality.working_memory_completeness * 100:.0f}%")
    print(f"Overall quality:        {ctx.quality.overall_quality}")
    if hb_age < float("inf"):
        print(f"Researcher heartbeat:   {round(hb_age, 1)}s ago")
    else:
        print("Researcher heartbeat:   unreachable")

    if LOCAL_MEMORY_FILE.exists():
        try:
            size = LOCAL_MEMORY_FILE.stat().st_size
            data = json.loads(LOCAL_MEMORY_FILE.read_text())
            total = sum(len(v) for v in data.values())
            print(f"Local memory file:      {size} bytes, ~{total} entries")
        except Exception:
            print(f"Local memory file:      exists ({LOCAL_MEMORY_FILE})")
    else:
        print("Local memory file:      not created")

    print()
    if mode == "INDEPENDENT":
        print("→ Conservative Independent Mode (reduced new risk).")
    else:
        print("→ Full researcher access.")
    print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ABC health checks")
    parser.add_argument(
        "target",
        nargs="?",
        choices=("researcher", "trader", "all"),
        default="all",
        help="Which check to run (default: all)",
    )
    args = parser.parse_args()
    code = 0
    if args.target in ("researcher", "all"):
        code = max(code, _check_researcher())
    if args.target in ("trader", "all"):
        if args.target == "all":
            print()
        code = max(code, _check_trader())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
