# PostgreSQL Host Setup (Research Machine)

This machine is your DB host and research box.

## Target subsystem
- Persistence and infra setup only (`infra/postgres/*`, docs)

## Likely adjacent subsystem impact
- `memory/__init__.py` (future Postgres cutover work)
- research/trader runtime startup scripts (future connection-string switch)

## Verification steps
- `docker compose ps`
- `docker compose logs postgres --tail 50`
- `docker exec abc-postgres psql -U postgres -d abc_shared -c "\du"`

## Rollback approach
- Stop and remove the DB stack:
  - `docker compose -f infra/postgres/docker-compose.yml down`
- If you must reset all DB data:
  - `docker compose -f infra/postgres/docker-compose.yml down -v`
- Current builds require PostgreSQL; there is no automatic SQLite fallback.
- If Postgres is unavailable, stop trading/research processes and restore service from a known-good Postgres backup before resuming.

---

## 1) Create local secrets file

Copy `infra/postgres/.env.example` to `infra/postgres/.env`, then set strong passwords.

PowerShell:

`Copy-Item "infra/postgres/.env.example" "infra/postgres/.env"`

## 2) Start Postgres

From repo root:

`docker compose -f infra/postgres/docker-compose.yml up -d`

## 3) Verify health

- `docker compose -f infra/postgres/docker-compose.yml ps`
- `docker compose -f infra/postgres/docker-compose.yml logs postgres --tail 50`
- `docker exec abc-postgres psql -U postgres -d abc_shared -c "SELECT current_database();"`
- `docker exec abc-postgres psql -U postgres -d abc_shared -c "\du"`

Expected roles:
- `postgres`
- `research_user`
- `trader_user`
- `abc_app` (NOLOGIN — optional until shared DDL setup; see §7)

## 4) Security guardrails

- Keep port `5432` private (LAN/VPN/Tailscale only).
- Do not expose DB directly to internet.
- Restrict host firewall to your trader machine IP (or Tailscale network).

## 5) Tomorrow: trader machine connection info

Use the DB host machine private IP (or Tailscale IP) and trader credentials.

Example URL format:

`postgresql://trader_user:<TRADER_PASSWORD>@<HOST_IP>:5432/abc_shared`

Research machine URL format:

`postgresql://research_user:<RESEARCH_PASSWORD>@127.0.0.1:5432/abc_shared`

Note: the memory layer now requires PostgreSQL connection env vars (`DATABASE_URL` or `PG*`).

## 6) Daily backups (recommended)

This repo includes a local backup script:

`scripts/backup_postgres.ps1`

It will:
- run `pg_dump -Fc` from the running `abc-postgres` container
- write a timestamped `.dump` file to `backups/postgres/`
- delete backups older than 14 days (default)

Run once manually:

`powershell -ExecutionPolicy Bypass -File "scripts/backup_postgres.ps1"`

Create a daily Windows Task Scheduler job (example 03:15):

`schtasks /Create /TN "ABC-Postgres-DailyBackup" /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"C:\Users\nvick\Documents\GitHub\ABC\scripts\backup_postgres.ps1\"" /SC DAILY /ST 03:15 /F`

## 7) Shared DDL role `abc_app` (trader + research `init_db`)

PostgreSQL only allows the **table owner** (or a superuser) to run `CREATE INDEX` and similar DDL. Grants are not enough. The trader runs `memory.init_db()` too (e.g. `scripts/verify_trader_db.py`), so objects cannot remain owned solely by `research_user`.

**Layout:** NOLOGIN role `abc_app` owns application tables. Both login roles are members. On connect, the app runs `SET ROLE abc_app` when `DATABASE_APP_ROLE=abc_app` is set in the **project** `.env` on the research host **and** the trader host.

**New databases:** `infra/postgres/init/02-abc-app-role.sh` creates `abc_app` on first container init.

**Existing databases** (tables already owned by `research_user`):

1. Apply role + grants (safe online):

`Get-Content "infra/postgres/admin/create_abc_app_role.sql" -Raw | docker exec -i abc-postgres psql -U postgres -d abc_shared -v ON_ERROR_STOP=1`

2. **Pause** the research daemon and trader DB connections — otherwise `REASSIGN OWNED` can wait indefinitely on locks.

3. Reassign ownership:

`Get-Content "infra/postgres/admin/reassign_owned_to_abc_app.sql" -Raw | docker exec -i abc-postgres psql -U postgres -d abc_shared -v ON_ERROR_STOP=1`

4. Add to **both** project `.env` files: `DATABASE_APP_ROLE=abc_app`

5. Trader: `python scripts/verify_trader_db.py` should print OK and exit 0.
