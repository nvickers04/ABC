# Entry points

**Single source of truth** for how to start each process. Implementation paths are
listed for navigation only — always use the commands below, not duplicate root scripts.

---

## Production (split host)

Run **one research process** on the research machine and **one trader process** on
the trader machine. Both use the same Postgres (`DATABASE_URL` / `PG*`).

| Machine | Start this | Never run here |
|---------|------------|----------------|
| **Research** | `python -m research` | `python __main__.py` (starts Grok → cost) |
| **Trader** | `python __main__.py --require-research-host` | `python -m research` (duplicate scorer) |

Set on the trader before start:

```powershell
$env:TRADER_IN_PROCESS_SCORER = "never"
python __main__.py --require-research-host --verbose
```

Research host alternatives:

```powershell
# From repo root (same as python -m research)
.\scripts\run_research.ps1

# Docker
docker compose -f infra/runtime/docker-compose.research.yml --env-file .env up -d --build
```

Trader Docker:

```powershell
docker compose -f infra/runtime/docker-compose.trader.yml --env-file .env up -d --build
```

Full layout: [operations/deployment.md](operations/deployment.md).  
Pre-live gates: [operations/launch-checklist.md](operations/launch-checklist.md).

---

## Development (single machine)

| Goal | Command |
|------|---------|
| Research scoring only | `python -m research` |
| Trader only (uses in-process scorer if heartbeat stale) | `python __main__.py` |
| Trader, never start local scorer | `TRADER_IN_PROCESS_SCORER=never python __main__.py --require-research-host` (research must be running) |
| Force in-process scorer on trader (no research host) | `python __main__.py --force-in-process` |
| Test Grok connection | `python __main__.py --test` |

Typical local loop: terminal 1 → `python -m research`; terminal 2 → `python __main__.py --verbose`.

### Heartbeat (research host liveness)

| Item | Value |
|------|--------|
| Postgres key (canonical) | `research_host_heartbeat_ts` in `research_config` |
| Legacy key (still read) | `daemon_heartbeat_ts` |
| Written by | `signals/scorer.py` each round via `core.runtime.heartbeat.write_heartbeat` |
| Checked by | Trader boot + each cycle (`is_research_host_alive`) |
| CLI flag | `--require-research-host` (legacy alias `--require-daemon`) — refuse trader start if heartbeat stale (`TRADER_IN_PROCESS_SCORER=never` is the same policy via env) |

---

## Ops scripts (from repo root)

| Script | Purpose |
|--------|---------|
| `python scripts/health.py researcher` | Heartbeat, daily usage, last scoring round |
| `python scripts/health.py trader` | FULL vs INDEPENDENT mode, WM source, risk multiplier |
| `python scripts/health.py` | Both checks |
| `python scripts/verify_trader_db.py` | Postgres reachability + migrations (trader host) |
| `python scripts/smoke_tools.py --client-id <id>` | Paper-safe tool sweep |
| `python scripts/smoke_tools.py --client-id <id> --all` | Safe + broker-mutating tools |
| `scripts/run_research.ps1` | Windows: same as `python -m research` |
| `scripts/backup_postgres.ps1` | DB backup (Windows) |
| `scripts/watch_research.ps1` | List `python -m research` PIDs + tail log |

---

## Where the code lives

| What you run | Module / file |
|--------------|----------------|
| `python -m research` | `research/__main__.py` → `research/host.py` |
| `python __main__.py` | `__main__.py` → `core/agent.py` (`TradingAgent`) |
| Research log file | `logs/research.log` (was `research_daemon.log` before consolidation) |
| Trader log file | `logs/agent.log` |

There is **no** `research_daemon.py` at the repository root.

---

## Removed commands (do not use)

These paths were deleted; grep the repo if docs or scripts still mention them:

- `python research_daemon.py`
- `research/daemon.py` (renamed to `research/host.py`)
- `python scripts/check_researcher.py` / `check_trader.py` → use `scripts/health.py`
- `python scripts/smoke_trader_tools.py` (and `smoke_all_tools`, `smoke_order_tools`) → use `scripts/smoke_tools.py`

See also [engineering.md](engineering.md) (file-move rules).
