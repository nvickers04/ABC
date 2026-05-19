#!/usr/bin/env python3
"""
Quick health check for the ABC Research Daemon (researcher machine).
Run this from either the research or trader machine as long as DB is reachable.
"""

from datetime import datetime, timezone
from core.runtime.heartbeat import is_daemon_alive, heartbeat_age_s, read_heartbeat
from memory import get_db, get_research_config
from core.config import RESEARCHER_DAILY_TOKEN_CAP

print("=== ABC Researcher Health Check ===\n")

# 1. Heartbeat
alive = is_daemon_alive()
age = heartbeat_age_s()
print(f"Daemon heartbeat alive:     {alive}")
print(f"Heartbeat age:              {round(age, 1)} seconds")

# 2. Daily usage vs cap
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
usage_key = f"researcher_daily_usage_{today}"
usage = float(get_research_config(usage_key, 0.0))
pct = (usage / RESEARCHER_DAILY_TOKEN_CAP) * 100 if RESEARCHER_DAILY_TOKEN_CAP > 0 else 0
print(f"Today's researcher usage:   {usage:,.0f} / {RESEARCHER_DAILY_TOKEN_CAP:,} ({pct:.1f}%)")

# 3. Last scoring activity
conn = get_db()
try:
    row = conn.execute("""
        SELECT MAX(ts) as last_ts, COUNT(*) as count
        FROM signal_scores
        WHERE DATE(ts, 'unixepoch') = DATE('now')
    """).fetchone()
    if row and row["last_ts"]:
        last = datetime.fromtimestamp(row["last_ts"], tz=timezone.utc)
        print(f"Last scoring round:         {last} ({row['count']} scores today)")
    else:
        print("Last scoring round:         No activity today")
except Exception as e:
    print(f"Could not query signal_scores: {e}")

# 4. Template evolution status
try:
    last_evo = get_research_config("last_template_evolution_round", 0)
    if last_evo > 0:
        evo_time = datetime.fromtimestamp(last_evo, tz=timezone.utc)
        print(f"Last evolution round:       {evo_time}")
    else:
        print("Last evolution round:       No record (or not yet run)")
except Exception:
    print("Last evolution round:       Unable to read")

print()

# Final verdict
if not alive or age > 600:
    status = "DOWN or STALLED"
elif pct >= 100:
    status = "CAP EXCEEDED — daemon likely shut down for the day"
elif pct > 85:
    status = "HEALTHY but approaching daily cap"
else:
    status = "HEALTHY"

print(f"Overall Status: {status}")
print("=====================================")
