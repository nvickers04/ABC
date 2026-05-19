#!/usr/bin/env python3
"""Lightweight healthchecks for Docker HEALTHCHECK (no colored tables).

Designed to run inside containers every 60s without burning MDA credits.
Startup gates (MDA probe, token cap) remain in ``research/host.py`` and
``scripts/health.py``.

Usage (in Dockerfile / compose):

  python scripts/docker_healthcheck.py research
  python scripts/docker_healthcheck.py trader

Exit codes: 0 healthy, 1 unhealthy.

Environment:

  DOCKER_HEALTHCHECK_MDA=1     Research only: also probe SPY quote (costly).
  DOCKER_HEALTHCHECK_REQUIRE_RESEARCH=1   Trader: fail if research host not operational.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env", override=False)
    except Exception:
        pass


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def check_research() -> int:
    """Postgres + fresh research heartbeat (scorer in this container writes it)."""
    _load_env()
    try:
        from memory import get_db

        get_db().execute("SELECT 1").fetchone()
    except Exception as e:
        print(f"research_health: postgres failed: {e}", file=sys.stderr)
        return 1

    try:
        from core.runtime.heartbeat import is_research_host_operational

        if not is_research_host_operational():
            print(
                "research_health: heartbeat stale or host not operational",
                file=sys.stderr,
            )
            return 1
    except Exception as e:
        print(f"research_health: heartbeat check failed: {e}", file=sys.stderr)
        return 1

    if _truthy("DOCKER_HEALTHCHECK_MDA"):
        try:
            from data.data_provider import get_data_provider

            q = get_data_provider().get_quote("SPY")
            if not q or q.last is None:
                print("research_health: MDA SPY probe failed", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"research_health: MDA probe error: {e}", file=sys.stderr)
            return 1

    print("research_health: ok")
    return 0


def check_trader() -> int:
    """Config validation + Postgres; optional research-host gate."""
    _load_env()
    try:
        from core.config import validate_config

        errs = validate_config()
        if errs:
            print(f"trader_health: config invalid: {errs}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"trader_health: config check failed: {e}", file=sys.stderr)
        return 1

    try:
        from memory import get_db

        get_db().execute("SELECT 1").fetchone()
    except Exception as e:
        print(f"trader_health: postgres failed: {e}", file=sys.stderr)
        return 1

    if _truthy("DOCKER_HEALTHCHECK_REQUIRE_RESEARCH"):
        try:
            from core.runtime.heartbeat import is_research_host_operational

            if not is_research_host_operational():
                print(
                    "trader_health: research host not operational",
                    file=sys.stderr,
                )
                return 1
        except Exception as e:
            print(f"trader_health: research host check failed: {e}", file=sys.stderr)
            return 1

    print("trader_health: ok")
    return 0


def main() -> None:
    role = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if role == "research":
        raise SystemExit(check_research())
    if role == "trader":
        raise SystemExit(check_trader())
    print("usage: docker_healthcheck.py research|trader", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
