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
| `env/docker.research.env.example` | Env vars for research `.env` |
| `env/docker.trader.env.example` | Env vars for trader `.env` |

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

On the research host set `ABC_HEALTH_ROLE=researcher` in compose or `.env`.

`abc-alert-watch` polls `/status` every 60s and logs critical/warn alerts; optional
`ALERT_WEBHOOK_URL` for Slack/generic webhooks.

Local: `python -m infra.status_api` (port `STATUS_API_PORT`, default 8790).

## Network

Postgres compose should attach to network `postgres_default` (see `infra/postgres/docker-compose.yml`).
Create once: `docker network create postgres_default`.

See [docs/operations/deployment.md](../../docs/operations/deployment.md) for split-host vs dev layout.
