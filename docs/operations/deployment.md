# Split-host deployment

Production layout: **one research machine** (scoring + evolution + Postgres host)
and **one trader machine** (Grok ReAct + IBKR), sharing a single **PostgreSQL**
database.

**Canonical start commands:** [../entry-points.md](../entry-points.md)

## Architecture (principles)

1. **Two-process decoupling** — `python -m research` on research host only; `__main__.py` on trader with `TRADER_IN_PROCESS_SCORER=never` and `--require-research-host` in production.
2. **Thin tools, thick brain** — `core/agent.py` decides; `tools/` execute.
3. **Cash-only, deterministic LLM** — `TotalCashValue` sizing; `temperature=0`, `seed=42` (QualityMatrix may tighten further).
4. **Key modules:** `core/quality/quality_matrix.py`, `core/runtime/operating_context.py`, `memory/`, `signals/scorer.py`.

Researcher writes → Postgres → trader reads each cycle. No direct IPC. When the researcher is down: [independent-mode.md](independent-mode.md).

```
Research host                          Trader host
─────────────────                      ─────────────────
python -m research                     python __main__.py --require-research-host
  scorer, templates, heartbeat           Grok + ToolExecutor + IBKR
  MDA / signal pipeline                  reads signals, WM, hypotheses from DB
         │                                        │
         └──────────── Postgres (shared) ────────┘
```

Postgres setup: [postgres.md](postgres.md).  
When the researcher is down: [independent-mode.md](independent-mode.md).

---

## What runs where

| Process | Host | Never on other host |
|---------|------|---------------------|
| `python -m research` | Research | Do **not** run `__main__.py` here (starts Grok → spend) |
| `python __main__.py` | Trader | Do **not** run `python -m research` here (duplicate scorer) |

Both machines need the same `DATABASE_URL` (or `PG*` vars). The memory layer
is **PostgreSQL only** — no SQLite fallback in current builds.

### Research host

- Runs scorer + template evolution; **does not** call Grok.
- Primary market data: MarketData.app (token cap enforced).
- `IBKR_QUOTES_ENABLED=0` internally — no IBKR market-data lines on this box.
- MDA health check at startup; research host exits if unhealthy.
- Daily cap: `RESEARCHER_DAILY_TOKEN_CAP` (default 100000) in `research_config`.

### Trader host

- Grok ReAct loop + IBKR execution only.
- **Production:** `TRADER_IN_PROCESS_SCORER=never` and `--require-research-host` so the
  trader never starts a second in-process scorer when the research host heartbeat is stale.
- `--force-in-process` is **dev/single-machine only**.
- IBKR streaming (stocks + options) lives here only.

---

## Launch commands

| Host | Command |
|------|---------|
| Research | `python -m research` or `scripts/run_research.ps1` |
| Trader (production) | `TRADER_IN_PROCESS_SCORER=never` then `python __main__.py --require-research-host --verbose` |

### Research host

Docker (recommended):

```powershell
cd C:\path\to\ABC
docker compose -f infra/runtime/docker-compose.research.yml --env-file .env up -d --build
```

Direct:

```powershell
python -m research
```

Windows Task Scheduler: run `scripts/run_research.ps1` or
`python -m research` from the repo root — **not** `__main__.py`.

### Trader host

Docker (recommended):

```powershell
docker compose -f infra/runtime/docker-compose.trader.yml --env-file .env up -d --build
```

Direct (production):

```powershell
$env:TRADER_IN_PROCESS_SCORER = "never"
python __main__.py --require-research-host --verbose
```

Paper soak before live: see [launch-checklist.md](launch-checklist.md).

---

## Network and security

- Prefer **Tailscale** (or private LAN) between hosts; do not expose port 5432 to the internet.
- Restrict firewall to trader IP / Tailscale range.
- Both machines should run the **same app commit** when sharing one database.

---

## Health checks

| Check | Command | Healthy |
|-------|---------|---------|
| Researcher | `python scripts/health.py researcher` | Heartbeat fresh, usage under cap |
| Trader mode | `python scripts/health.py trader` | Mode FULL or expected INDEPENDENT |
| Heartbeat (quick) | `python -c "from core.runtime.heartbeat import is_research_host_alive, heartbeat_age_s; print(is_research_host_alive(), heartbeat_age_s())"` | `True`, age low |
| DB | `python scripts/verify_trader_db.py` | Exit 0 (trader host) |

Research token usage query:

```sql
SELECT key, value FROM research_config WHERE key LIKE 'researcher_daily_usage_%';
```

---

## Common mistakes

| Mistake | Why it hurts |
|---------|----------------|
| `__main__.py` on research host | Grok spend every cycle |
| Bare `python __main__.py` on trader (split-host) | In-process scorer duplicates research host |
| Different DB or stale commit on one host | Stale signals, schema drift |
| Dual-write WM to Postgres + local JSON | Drift — see [independent-mode.md](independent-mode.md) |

---

## Related

- [postgres.md](postgres.md) — install, `abc_app` role, backups
- [independent-mode.md](independent-mode.md) — WM authority when researcher is down
- [launch-checklist.md](launch-checklist.md) — pre-live gates
- [../data-sources.md](../data-sources.md) — MDA, IBKR, yfinance, Grok
