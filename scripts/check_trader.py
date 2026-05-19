#!/usr/bin/env python3
"""
Quick status check for the ABC Trader (Independent Mode support).

Run this anytime (even with market closed) to see the current operating mode,
risk posture, and memory situation.
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from core.runtime.operating_context import get_operating_context
from core.runtime.heartbeat import is_daemon_alive, heartbeat_age_s
from core.runtime.local_memory_fallback import LOCAL_MEMORY_FILE

def main():
    print("=== ABC Trader Status ===\n")

    ctx = get_operating_context()

    try:
        researcher_alive = is_daemon_alive()
        hb_age = heartbeat_age_s()
    except Exception:
        researcher_alive = False
        hb_age = float("inf")

    mode = "INDEPENDENT" if (not researcher_alive or ctx.is_independent_mode) else "FULL"
    print(f"Mode: {mode}")
    print(f"Researcher available : {ctx.quality.researcher_available}")
    print(f"Memory source        : {ctx.quality.memory_source}")
    print(f"Risk multiplier      : {ctx.risk_multiplier}  (1.0 = normal, lower = more conservative)")
    print(f"Working Memory fill  : {ctx.quality.working_memory_completeness * 100:.0f}%")
    print(f"Overall quality      : {ctx.quality.overall_quality}")

    if hb_age < float("inf"):
        print(f"Researcher heartbeat : {round(hb_age, 1)}s ago")
    else:
        print("Researcher heartbeat : unreachable")

    # Show if local memory file exists and its rough size
    if LOCAL_MEMORY_FILE.exists():
        try:
            size = LOCAL_MEMORY_FILE.stat().st_size
            data = json.loads(LOCAL_MEMORY_FILE.read_text())
            total_entries = sum(len(v) for v in data.values())
            print(f"Local memory file    : {size} bytes, ~{total_entries} entries")
        except Exception:
            print(f"Local memory file    : exists ({LOCAL_MEMORY_FILE})")
    else:
        print("Local memory file    : not yet created")

    print()
    if mode == "INDEPENDENT":
        print("→ Trader is in conservative Independent Mode.")
        print("   New risk is reduced. Call context_quality() from inside the agent for details.")
    else:
        print("→ Trader has full researcher access.")

    print("\n=== End ===")

if __name__ == "__main__":
    main()

