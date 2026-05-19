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

## Network

Postgres compose should attach to network `postgres_default` (see `infra/postgres/docker-compose.yml`).
Create once: `docker network create postgres_default`.

See [docs/operations/deployment.md](../../docs/operations/deployment.md) for split-host vs dev layout.
