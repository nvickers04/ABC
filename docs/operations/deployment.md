# Split-host deployment

Production layout: **one research machine** (scoring + evolution + Postgres host)
and **one trader machine** (Grok ReAct + IBKR), sharing a single **PostgreSQL**
database.

**See also:** [../entry-points.md](../entry-points.md) · [postgres.md](postgres.md) · [independent-mode.md](independent-mode.md) · [../plain-english-glossary.md](../plain-english-glossary.md)

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

Docker (two-machine production):

```powershell
cd C:\path\to\ABC
docker network create postgres_default
docker compose -f infra/runtime/docker-compose.research.yml `
  -f infra/runtime/examples/docker-compose.prod.research.yml `
  --env-file .env up -d --build
```

Env template: `infra/runtime/env/docker.research.env.example` → merge into repo `.env`.

Direct:

```powershell
python -m research
```

Windows Task Scheduler: run `scripts/run_research.ps1` or
`python -m research` from the repo root — **not** `__main__.py`.

### Trader host

Docker (two-machine production):

```powershell
docker compose -f infra/runtime/docker-compose.trader.yml `
  -f infra/runtime/examples/docker-compose.prod.trader.yml `
  --env-file .env up -d --build
```

Env template: `infra/runtime/env/docker.trader.env.example` → merge into repo `.env`.

Direct (production):

```powershell
$env:TRADER_IN_PROCESS_SCORER = "never"
python __main__.py --require-research-host --verbose
```

Paper soak before live: see [launch-checklist.md](launch-checklist.md).

---

## Docker layouts

Reference: [infra/runtime/README.md](../../infra/runtime/README.md).

### Two-machine production (default)

| Machine | Compose files | Must not run |
|---------|---------------|--------------|
| Research + Postgres | `infra/postgres/docker-compose.yml` + `docker-compose.research.yml` + `examples/docker-compose.prod.research.yml` | Trader compose |
| Trader | `docker-compose.trader.yml` + `examples/docker-compose.prod.trader.yml` | Research compose |

Shared `postgres_default` network on the Postgres host; research/trader `DATABASE_URL` uses Tailscale IP or `abc-postgres:5432` when on the same Docker network.

### Single-machine development

Run Postgres once, then stack **both** services with the dev override (allows in-process scorer fallback, no `--require-research-host`):

```powershell
docker network create postgres_default
docker compose -f infra/postgres/docker-compose.yml --env-file infra/postgres/.env up -d

docker compose -f infra/runtime/docker-compose.research.yml `
  -f infra/runtime/examples/docker-compose.dev.yml `
  --env-file .env up -d --build

docker compose -f infra/runtime/docker-compose.trader.yml `
  -f infra/runtime/examples/docker-compose.dev.yml `
  --env-file .env up -d --build
```

Do **not** use the dev override in production split-host (it disables `TRADER_IN_PROCESS_SCORER=never`).

### Container healthchecks vs operator scripts

| Layer | Command | Purpose |
|-------|---------|---------|
| Docker `HEALTHCHECK` | `python scripts/docker_healthcheck.py research\|trader` | Fast: Postgres + heartbeat / config (no MDA by default) |
| Operator | `python scripts/health.py researcher\|trader` | Full: MDA, token cap, tables, optional IBKR |

**Environment overrides (compose / `.env`):**

| Variable | Research | Trader |
|----------|----------|--------|
| `DATABASE_URL` | Required | Required (same DB) |
| `DOCKER_HEALTHCHECK_MDA` | `0` (default) / `1` to probe SPY each interval | — |
| `DOCKER_HEALTHCHECK_REQUIRE_RESEARCH` | — | `1` in prod override (fail health if research host down) |
| `TRADER_IN_PROCESS_SCORER` | — | `never` (prod) / `auto` (dev override) |
| `IBKR_HOST` | — | `host.docker.internal` (default in compose) |

Inspect: `docker inspect --format='{{.State.Health.Status}}' abc-research-host`

---

## Network and security

- Prefer **Tailscale** (or private LAN) between hosts; do not expose port 5432 to the internet.
- Restrict firewall to trader IP / Tailscale range.
- Both machines should run the **same app commit** when sharing one database.

---

## Health checks

| Check | Command | Healthy |
|-------|---------|---------|
| Research host | `python scripts/health.py researcher` | Exit **0** (or **1** warn); heartbeat fresh, token under cap, MDA OK |
| Trader | `python scripts/health.py trader` | Exit **0**–**1**; FULL or expected INDEPENDENT |
| Trader + IBKR | `python scripts/health.py trader --ibkr-client-id 11` | IBKR line OK |
| Postgres | `python scripts/verify_trader_db.py` | Exit **0** |
| Heartbeat (quick) | `python -c "from core.runtime.heartbeat import is_research_host_alive, heartbeat_age_s; print(is_research_host_alive(), heartbeat_age_s())"` | `True`, age low |

Exit **2** = failure. See [plain-english-glossary.md](../plain-english-glossary.md#health-scripts-exit-codes).

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

- [infra/runtime/README.md](../../infra/runtime/README.md) — Dockerfiles, overrides, healthchecks
- [postgres.md](postgres.md) — install, `abc_app` role, backups
- [independent-mode.md](independent-mode.md) — WM authority when researcher is down
- [launch-checklist.md](launch-checklist.md) — pre-live gates
- [../data-sources.md](../data-sources.md) — MDA, IBKR, yfinance, Grok
