# Independent Mode & Working Memory Policy

This document defines how the trader chooses between **Postgres working memory**
and the **local JSON fallback**, what happens on restart, and what happens when
the researcher comes back. It is the operational contract for
`get_active_working_memory()` and `OperatingContext`.

## Two separate signals (do not conflate)

| Signal | Meaning | Typical cause |
|--------|---------|----------------|
| **`researcher_available`** | Research daemon heartbeat is fresh | `research_daemon.py` scoring on research host |
| **`memory_source`** | Where WM reads/writes go | `postgres` vs `local_fallback` |

**Independent Mode** (for trading posture and prompts) means the researcher
feed is unavailable or WM has fallen back to local storage — see
`OperatingContext.is_independent_mode`.

**Postgres unavailable** is a different failure: the trader cannot persist
trades, signals, or shared state normally. Local WM alone is **not** sufficient
to run production trading; it only preserves the agent's short-term theses and
notes.

### Split-host production

- Research machine: Postgres + `research_daemon.py`
- Trader machine: `python __main__.py --verbose --require-daemon`

The trader uses the **same** Postgres for heartbeat and WM. A stale heartbeat
means **no fresh research**, not “no database.” Prefer fixing the daemon or DB
connection before treating the system as a generic “DB down” event.

## Authority: single writer per mode

Do **not** dual-write working memory to Postgres and local JSON in normal
operation. One store is authoritative at a time.

| Mode | WM authority | Postgres `working_memory` | `data/local_working_memory.json` |
|------|----------------|---------------------------|----------------------------------|
| **Full** (heartbeat OK, postgres WM reachable) | Postgres | Read/write | Ignored |
| **Independent** (heartbeat stale / down) | Local JSON | Not written | Read/write |
| **Postgres WM error** | Local JSON (emergency) | Unavailable | Read/write; context flips to `local_fallback` |

All agent and tool paths must use **`get_active_working_memory()`** in
`core/runtime/working_memory_access.py` — never call `get_working_memory()` and
`get_local_working_memory()` directly except inside that module or tests.

## Trader restart

On each process start:

1. `OperatingContext.sync_researcher_from_heartbeat()`
2. Active store = result of `get_active_working_memory()`
3. **Postgres path:** `get_working_memory()` runs `restore_today()` (today's rows)
4. **Local path:** JSON file loaded from `data/local_working_memory.json`

**Do not** sync or merge both stores on restart. Pick one authority from the
heartbeat + WM availability checks above.

## Recovery when the researcher returns (Option A — current)

When heartbeat becomes healthy, `set_researcher_available()`:

1. Sets `memory_source` back to `postgres`
2. Refreshes QualityMatrix policy
3. Logs a **recovery summary** (local vs postgres entry counts) — **no automatic merge**

**Policy:** Postgres wins for new writes. Entries created in local JSON during
the outage remain in `data/local_working_memory.json` as an archive until they
expire or an operator merges them manually.

Future **Option B** (not implemented): one-shot `reconcile_working_memory_on_recovery()`
from local → Postgres on reconnect, with cap/expiry rules and tests.

## What is not duplicated in local JSON

Local fallback is **only** the five WM sections (`open_theses`, `recent_verdicts`,
`watching_for`, `regime_notes`, `lessons_today`). It does **not** replace:

- Signal scores / briefing tables
- Attention triggers (Postgres)
- Trade history / `trade_feedback`
- QualityMatrix provenance tables

Those remain on Postgres; Independent Mode reduces **trust** in research-facing
tools via QualityMatrix, not by copying the research DB locally.

## Operator checklist

| Situation | Action |
|-----------|--------|
| Heartbeat stale on trader | Fix research daemon or network; check `scripts/check_researcher.py` |
| Long independent session | After recovery, read logs for `WM recovery:` line; review local file if theses matter |
| Local file growing | Normal during outages; gitignored; delete only if you accept losing archived notes |
| Tests hang on Postgres | Quality/WM unit tests override `_isolated_db`; see `tests/quality_test_support.py` |

## Code map

| Piece | Role |
|-------|------|
| `core/runtime/operating_context.py` | Mode flags, heartbeat sync, recovery log hook |
| `core/runtime/working_memory_access.py` | Routing + recovery summaries |
| `core/runtime/local_memory_fallback.py` | JSON-backed WM (API-compatible with Postgres WM) |
| `memory/working_memory.py` | Canonical Postgres WM |
| `core/quality/quality_matrix.py` | Risk/policy when context quality drops |

## Related docs

- `docs/PLAN_COGNITIVE_ARCHITECTURE.md` — WM sections and tool design
- `docs/plain-english-glossary.md` — cycle loop, contracts, blast radius
- `.cursor/rules/single-trader-process.mdc` — split-host trader vs daemon
