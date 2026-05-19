# PostgreSQL setup (shared database)

The research and trader machines share one Postgres database for signals,
working memory, trades, heartbeat, and `research_config`.

**Deployment context:** [deployment.md](deployment.md) · **Start commands:** [../entry-points.md](../entry-points.md)

---

## 1) Create local secrets file

Copy `infra/postgres/.env.example` to `infra/postgres/.env` and set strong passwords.

```powershell
Copy-Item "infra/postgres/.env.example" "infra/postgres/.env"
```

## 2) Start Postgres

From repo root:

```powershell
docker compose -f infra/postgres/docker-compose.yml up -d
```

## 3) Verify health

```powershell
docker compose -f infra/postgres/docker-compose.yml ps
docker compose -f infra/postgres/docker-compose.yml logs postgres --tail 50
docker exec abc-postgres psql -U postgres -d abc_shared -c "SELECT current_database();"
docker exec abc-postgres psql -U postgres -d abc_shared -c "\du"
```

Expected roles: `postgres`, `research_user`, `trader_user`, and optionally `abc_app` (NOLOGIN until §5).

## 4) Security

- Keep port `5432` on a private network (LAN / Tailscale only).
- Do not expose the database to the public internet.
- Restrict the host firewall to the trader machine IP.

## 5) Connection strings

Trader machine (use host private or Tailscale IP):

```text
postgresql://trader_user:<PASSWORD>@<HOST_IP>:5432/abc_shared
```

Research machine (local):

```text
postgresql://research_user:<PASSWORD>@127.0.0.1:5432/abc_shared
```

Set `DATABASE_URL` or `PG*` in each machine’s project `.env`. The app requires Postgres; there is no SQLite fallback.

## 6) Daily backups

```powershell
powershell -ExecutionPolicy Bypass -File "scripts/backup_postgres.ps1"
```

Writes timestamped dumps under `backups/postgres/` (14-day retention by default).

Example scheduled task (adjust path):

```text
schtasks /Create /TN "ABC-Postgres-DailyBackup" /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"C:\path\to\ABC\scripts\backup_postgres.ps1\"" /SC DAILY /ST 03:15 /F
```

## 7) Shared DDL role `abc_app`

PostgreSQL only allows the **table owner** (or superuser) to run `CREATE INDEX` and similar DDL. Both the trader and research processes call `memory.init_db()`, so application objects should be owned by a shared role.

**Layout:** NOLOGIN role `abc_app` owns application tables. Both login roles are members. Set `DATABASE_APP_ROLE=abc_app` in **both** project `.env` files.

**New databases:** `infra/postgres/init/02-abc-app-role.sh` runs on first container init.

**Existing databases** (tables already owned by `research_user`):

1. Apply role + grants:

```powershell
Get-Content "infra/postgres/admin/create_abc_app_role.sql" -Raw | docker exec -i abc-postgres psql -U postgres -d abc_shared -v ON_ERROR_STOP=1
```

2. Pause the research host (`python -m research`) and trader DB connections (avoid lock waits on `REASSIGN OWNED`).

3. Reassign ownership:

```powershell
Get-Content "infra/postgres/admin/reassign_owned_to_abc_app.sql" -Raw | docker exec -i abc-postgres psql -U postgres -d abc_shared -v ON_ERROR_STOP=1
```

4. Add `DATABASE_APP_ROLE=abc_app` to both `.env` files.

5. On trader: `python scripts/verify_trader_db.py` should print OK and exit 0.

## Rollback

Stop stack:

```powershell
docker compose -f infra/postgres/docker-compose.yml down
```

Reset all data (destructive):

```powershell
docker compose -f infra/postgres/docker-compose.yml down -v
```

If Postgres is unavailable, stop research and trader processes and restore from a known-good backup before resuming.
