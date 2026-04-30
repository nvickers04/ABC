# Research Host + Trader Machine Setup

This runbook is for the split deployment:
- **This machine** = research daemon + DB host
- **Other machine** = trading agent

## Current reality (important)

The codebase currently defaults to local SQLite at `memory/abc.db`.
That means each machine has its own DB unless/until Postgres support is wired in.

Use this doc to prepare both machines now and complete cutover once shared DB support is enabled.

## Host machine (today)

### 1) Install and verify Docker
- Install Docker Desktop (AMD64 on most Intel/AMD Windows PCs)
- Verify:
  - `docker --version`
  - `docker compose version`

### 2) Keep research running with current code
- Pull latest repo changes
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
- Start trader with local SQLite once to confirm environment health
- Stop it after sanity check

## Shared DB cutover phases

### Phase A: Infrastructure
- Run Postgres in Docker on host machine (private network only)
- Create DB and separate users:
  - `research_user`
  - `trader_user`
- Restrict inbound DB access to trader machine Tailscale IP

### Phase B: App support
- Add Postgres backend support in memory layer
- Keep SQLite fallback for local testing
- Run schema/init against Postgres

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
