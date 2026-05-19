# Independent Mode & working memory

How the trader chooses between **Postgres working memory** and the **local JSON
fallback**, what happens on restart, and what happens when the researcher returns.

Operational contract for `get_active_working_memory()` and `OperatingContext`.

**Split-host context:** [deployment.md](deployment.md) · **Start commands:** [../entry-points.md](../entry-points.md)

---

## Two separate signals (do not conflate)

| Signal | Meaning | Typical cause |
|--------|---------|----------------|
| **`researcher_available`** | Research host heartbeat is fresh | `python -m research` on research host |
| **`memory_source`** | Where WM reads/writes go | `postgres` vs `local_fallback` |

**Independent Mode** (trading posture + prompts) means the researcher feed is
unavailable or WM has fallen back to local storage (`OperatingContext.is_independent_mode`).

**Postgres unavailable** is different: the trader cannot persist trades, signals,
or shared state. Local WM alone is not enough to trade safely — it only preserves
short-term theses and notes.

On split-host production, a **stale heartbeat** means no fresh research, not “no
database.” Fix the daemon or network before assuming Postgres is down.

---

## Authority: single writer per mode

Do **not** dual-write working memory to Postgres and local JSON in normal operation.

| Mode | WM authority | Postgres `working_memory` | `data/local_working_memory.json` |
|------|----------------|---------------------------|----------------------------------|
| **Full** (heartbeat OK) | Postgres | Read/write | Ignored |
| **Independent** (heartbeat stale) | Local JSON | Not written | Read/write |
| **Postgres WM error** | Local JSON (emergency) | Unavailable | Read/write |

All paths use **`get_active_working_memory()`** in `core/runtime/working_memory_access.py`.

---

## Trader restart

1. `OperatingContext.sync_researcher_from_heartbeat()`
2. `get_active_working_memory()` picks the store
3. Postgres: `restore_today()` on startup; local: load JSON file

**Do not** sync both stores on restart.

---

## Recovery when the researcher returns (Option A)

When heartbeat is healthy, `set_researcher_available()`:

1. Sets `memory_source` back to `postgres`
2. Refreshes QualityMatrix policy
3. Logs **WM recovery:** summary (counts) — **no automatic merge**

Postgres wins for new writes. Local entries during the outage stay in
`data/local_working_memory.json` until they expire or you merge manually.

---

## What local JSON does not replace

Only the five WM sections: `open_theses`, `recent_verdicts`, `watching_for`,
`regime_notes`, `lessons_today`. Signals, attention, trades, and provenance
remain on Postgres; QualityMatrix reduces trust in research tools when degraded.

---

## Operator checklist

| Situation | Action |
|-----------|--------|
| Heartbeat stale | `python scripts/health.py researcher`; fix daemon |
| After long independent run | Grep logs for `WM recovery:` |
| Local file growth | Normal; gitignored |

---

## Code map

| Module | Role |
|--------|------|
| `core/runtime/operating_context.py` | Mode flags, heartbeat sync |
| `core/runtime/working_memory_access.py` | Routing + recovery log |
| `core/runtime/local_memory_fallback.py` | JSON WM store |
| `memory/working_memory.py` | Postgres WM |
| `core/quality/quality_matrix.py` | Host policy when quality drops |

---

## Related

- [deployment.md](deployment.md) — two-machine layout
- [../engineering.md](../engineering.md) (glossary)
- `memory/working_memory.py` — WM sections, caps, and persistence
