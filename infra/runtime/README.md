# Runtime Docker images

| File | Purpose |
|------|---------|
| `Dockerfile.research` | Research host — `python -m research` only |
| `Dockerfile.trader` | Trader — `python __main__.py` |
| `docker-compose.research.yml` | Research service base |
| `docker-compose.trader.yml` | Trader service base |
| `examples/docker-compose.dev.yml` | Single-machine dev override |
| `examples/docker-compose.prod.research.yml` | Two-host research override |
| `examples/docker-compose.prod.trader.yml` | Two-host trader override |
| `docker-compose.status.yml` | Status API + alert watcher (port 8790) |
| `../status_api/Dockerfile` | Observability API image |
| `env/paper.env.example` | Paper trading profile (`TRADING_MODE`, `IBKR_PORT`, …) |
| `env/live.env.example` | Live trading profile |
| `env/dev.env.example` | Single-machine dev profile |
| `env/docker.research.env.example` | Research-host `.env` template |
| `env/docker.trader.env.example` | Trader-host `.env` template |
| `examples/docker-compose.env.paper.yml` | Compose override for paper IBKR |
| `examples/docker-compose.env.live.yml` | Compose override for live IBKR |
| `../deploy.py` | One-click deploy (`--role`, `--env`, `--registry`) |

## Healthchecks

Containers use `scripts/docker_healthcheck.py` (not full `scripts/health.py`) to avoid
MDA spend on every interval:

| Role | Default check | Optional |
|------|---------------|----------|
| Research | Postgres `SELECT 1` + operational heartbeat | `DOCKER_HEALTHCHECK_MDA=1` → SPY quote |
| Trader | `validate_config()` + Postgres | `DOCKER_HEALTHCHECK_REQUIRE_RESEARCH=1` → research host operational |

Full operator checks: `python scripts/health.py researcher|trader` on the host.

## Status API and alerting

Attach the status stack to trader or research compose:

```powershell
docker compose -f infra/runtime/docker-compose.trader.yml `
  -f infra/runtime/docker-compose.status.yml `
  --env-file .env up -d
```

| Endpoint | Purpose |
|----------|---------|
| `GET http://localhost:8790/status` | Full JSON health (ProfitConfig, alerts, daily summary) |
| `GET /health/ready` | 503 when `overall_status` is `unhealthy` |
| `GET /status/text` | Plain-text operator summary |
| `GET /dashboard` | HTML ops dashboard (HTTP Basic; requires `DASHBOARD_PASSWORD`) |
| `GET /dashboard/data` | Same payload as JSON |

On the research host set `ABC_HEALTH_ROLE=researcher` in compose or `.env`.

Set `DASHBOARD_PASSWORD` in `.env` for the web dashboard. Mounts `logs/` and `data/` read-only
so cycle logs and optimizer JSON are visible inside the container.

`abc-alert-watch` polls `/status` every 60s and logs critical/warn alerts; optional
`ALERT_WEBHOOK_URL` for Slack/generic webhooks.

Local: `python -m infra.status_api` (port `STATUS_API_PORT`, default 8790).

## Network

Postgres compose should attach to network `postgres_default` (see `infra/postgres/docker-compose.yml`).
Create once: `docker network create postgres_default`.

See [docs/operations/deployment.md](../../docs/operations/deployment.md) for split-host vs dev layout.

## One-click deploy

From repo root (after copying `infra/postgres/.env.example` → `infra/postgres/.env` and merging secrets into `.env`):

```powershell
# Research machine + Postgres (paper)
python infra/deploy.py --role research --env paper --with-postgres --build

# Trader machine (paper, split-host production)
python infra/deploy.py --role trader --env paper --build

# Live trader (after paper soak)
python infra/deploy.py --role trader --env live --build

# Single-machine dev
python infra/deploy.py --role all --env dev --with-postgres --build

# Pull images built by GitHub Actions (GHCR)
python infra/deploy.py --role trader --env paper --registry ghcr.io/YOUR_ORG/YOUR_REPO --tag latest --pull
```

Profiles load `infra/runtime/env/{paper,live,dev}.env.example` (or gitignored `{env}.env`), merge with repo `.env`, and write `.env.deploy` for compose. Optional local override: copy `paper.env.example` → `infra/runtime/env/paper.env`.

CI builds and pushes `ghcr.io/<repo>/trader`, `/research`, and `/status-api` on pushes to `main` (see `.github/workflows/ci.yml`).
