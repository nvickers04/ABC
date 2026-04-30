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
- App runtime can continue using SQLite at `memory/abc.db`.

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
