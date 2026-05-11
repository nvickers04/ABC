"""One-off: print research heartbeat and recent signal-engine activity."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
# Avoid indefinite hang when VPN / firewall blocks Postgres.
os.environ.setdefault("PGCONNECT_TIMEOUT", "15")

import memory  # noqa: E402

memory.reset_state()
memory.init_db()
db = memory.get_db()
now = time.time()

row = db.execute(
    "SELECT key, value, updated_ts, reason FROM research_config WHERE key = ?",
    ("daemon_heartbeat_ts",),
).fetchone()
print("=== research daemon heartbeat (daemon_heartbeat_ts) ===")
if row:
    hb = float(row["value"])
    age = now - hb
    print("value_unix:", hb)
    print("updated_ts:", row["updated_ts"])
    print("reason:", row["reason"])
    print("age_seconds:", round(age, 1))
    print("is_fresh_under_5min:", age < 300)
else:
    print("NO ROW — daemon may not have written to this database yet")

print()
print("=== recent research_config rows (last 12 by updated_ts) ===")
rows = db.execute(
    "SELECT key, value, updated_ts, reason FROM research_config "
    "ORDER BY updated_ts DESC LIMIT 12"
).fetchall()
for r in rows:
    print(dict(r))

print()
checks: list[tuple[str, str, tuple | None]] = [
    (
        "signal_scores (24h)",
        "SELECT COUNT(*) AS n, MAX(ts) AS max_ts FROM signal_scores WHERE ts > ?",
        (now - 86400,),
    ),
    (
        "composite_scores (24h)",
        "SELECT COUNT(*) AS n, MAX(ts) AS max_ts FROM composite_scores WHERE ts > ?",
        (now - 86400,),
    ),
    (
        "signal_weights (all rows)",
        "SELECT COUNT(*) AS n, MAX(updated_ts) AS max_ts FROM signal_weights",
        None,
    ),
    (
        "signal_symbol_ic (all rows)",
        "SELECT COUNT(*) AS n, MAX(last_updated_ts) AS max_ts FROM signal_symbol_ic",
        None,
    ),
]
for label, q, params in checks:
    if params is None:
        r = db.execute(q).fetchone()
    else:
        r = db.execute(q, params).fetchone()
    print(f"{label}: {dict(r) if r else None}")

print()
sess = db.execute("SELECT current_user, session_user").fetchone()
print("session:", dict(sess))
