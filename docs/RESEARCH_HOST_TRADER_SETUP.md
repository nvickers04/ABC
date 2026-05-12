# Research Host + Trader Machine Setup

This runbook is for the split deployment:
- **This machine** = research daemon + DB host
- **Other machine** = trading agent

Detailed host bootstrap commands live in `docs/POSTGRES_HOST_SETUP.md`.

## Current reality (important)

The memory layer now targets PostgreSQL only.
Both machines must point at the same Postgres DB to share research/trading state.

## Docker / Task Scheduler (avoid accidental Grok spend)

On the **research host**, only one of these should run continuously:

| Correct (research only) | Wrong (full trader + Grok) |
|-------------------------|----------------------------|
| `python research_daemon.py` | `python __main__.py` |
| `.\scripts\run_research_daemon.ps1` | Any Docker `CMD` / scheduler action pointing at `__main__.py` |

`__main__.py` starts **TradingAgent** and will call **Grok** every cycle (input tokens add up fast).  
`research_daemon.py` does **not** call Grok (see header comment in that file).

**Docker (recommended on the research machine):**

```powershell
cd C:\path\to\ABC
docker compose -f infra/runtime/docker-compose.research.yml --env-file .env up -d --build
```

Ensure `.env` has a `DATABASE_URL` (or `PG*`) that the container can reach (`host.docker.internal` for Postgres on the same PC under Docker Desktop, or your DB host / Tailscale IP).

**Windows Task Scheduler:** set **Program** to `powershell.exe` and **Arguments** to  
`-ExecutionPolicy Bypass -File "C:\path\to\ABC\scripts\run_research_daemon.ps1"`  
(or call `python` with an explicit full path to `research_daemon.py` — not `__main__.py`).

## Host machine (today)

### 1) Install and verify Docker
- Install Docker Desktop (AMD64 on most Intel/AMD Windows PCs)
- Verify:
  - `docker --version`
  - `docker compose version`

### 2) Keep research running with current code
- Pull latest repo changes
- Set Postgres environment variables (`DATABASE_URL` or `PG*` vars) and
  `DATABASE_APP_ROLE=abc_app` after running the shared-owner migration — see
  `docs/POSTGRES_HOST_SETUP.md` §7.
- Start your research daemon as normal
- Confirm signal pipeline is live (IC logs, composite updates)

### 3) Prepare secure network path for tomorrow
- Install Tailscale (recommended) on both machines
- Verify they can ping each other on Tailscale IPs
- Do **not** expose DB port 5432 publicly to the internet

### 4) Back up local SQLite now
- Back up `memory/abc.db` before any DB migration work

## Trader machine (tomorrow)

### 1) Install prerequisites
- Docker Desktop
- Tailscale
- Git + Python environment for this repo

### 2) Sync repo safely (required if history was rewritten)
- If you have local uncommitted work, stash it first:
  - `git stash push -m "pre-main-sync"`
- Then hard-sync to remote `main`:
  - `git fetch origin`
  - `git checkout main`
  - `git reset --hard origin/main`
- Confirm latest commit:
  - `git log --oneline -1`

### 3) Validate runtime before shared DB cutover
- Start trader once against shared Postgres to confirm environment health
- Stop it after sanity check

## Shared DB cutover phases

### Phase A: Infrastructure
- Run Postgres in Docker on host machine (private network only)
- Create DB and separate users:
  - `research_user`
  - `trader_user`
- Restrict inbound DB access to trader machine Tailscale IP

### Phase B: App support
- Ensure Postgres env vars are set on both machines
- Run app startup once to apply schema/init against Postgres

### Phase C: Data migration + switch
- Export/import relevant SQLite tables into Postgres
- Point both machines to the same DB connection string
- Start research first, then trader

## Safety checklist before going live on shared DB
- Both machines on same app commit hash
- Research writes are fresh (timestamps advancing)
- Trader sees non-stale research snapshots
- Backups configured (at least daily dump)
- Rollback plan documented (switch both back to SQLite if needed)

## Tomorrow quick checklist (copy/paste)
- [ ] Install Docker Desktop on trader machine
- [ ] Install Tailscale on trader machine
- [ ] `git fetch origin && git checkout main && git reset --hard origin/main`
- [ ] Verify local startup on trader machine
- [ ] Confirm host and trader can reach each other over Tailscale
- [ ] Proceed with Postgres setup + code cutover
