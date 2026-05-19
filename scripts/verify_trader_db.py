#!/usr/bin/env python3
"""Verify Postgres connectivity, schema, heartbeat, and core tables.

Run on the trader or research machine with DATABASE_URL (or PG*) in ``.env``.
Set DATABASE_APP_ROLE=abc_app when using the shared DDL owner (see
docs/operations/postgres.md).

Exit codes:

  0 — all checks passed
  1 — warnings only (e.g. stale heartbeat on trader-only probe)
  2 — failure (config, connection, or schema)

Usage (from repo root):

    python scripts/verify_trader_db.py
    python scripts/verify_trader_db.py --no-color
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for _p in (ROOT, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from health_common import (  # noqa: E402
    Reporter,
    check_database_url_config,
    check_postgres_init,
    check_postgres_ping,
    check_required_tables,
    check_research_heartbeat,
    check_token_cap,
    load_dotenv_repo,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Postgres for ABC trader/research")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--role",
        choices=("trader", "researcher"),
        default="trader",
        help="Heartbeat strictness (default: trader)",
    )
    args = parser.parse_args()

    load_dotenv_repo(ROOT)
    rep = Reporter("ABC Postgres Verification", use_color=False if args.no_color else None)

    if not check_database_url_config(rep):
        rep.print_report()
        return rep.exit_code()

    if not check_postgres_ping(rep):
        rep.print_report()
        return rep.exit_code()

    if not check_postgres_init(rep):
        rep.print_report()
        return rep.exit_code()

    check_required_tables(rep)
    check_research_heartbeat(rep, role=args.role)
    check_token_cap(rep)

    rep.print_report()
    return rep.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
