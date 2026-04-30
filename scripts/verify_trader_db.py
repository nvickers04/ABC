#!/usr/bin/env python3
"""Verify Postgres connectivity and schema init using credentials in `.env`.

Run on the trader machine with DATABASE_URL or PG* set for `trader_user`.
Exit code 0 and prints OK when memory.init_db() succeeds.

Usage (from repo root):

    python scripts/verify_trader_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import memory

memory.reset_state()
memory.init_db()
print("OK memory.init_db() completed")
