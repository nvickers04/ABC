# Entry points

**Single source of truth** for how to start each process. Implementation paths are
listed for navigation only — always use the commands below, not duplicate root scripts.

**See also:** [plain-english-glossary.md](plain-english-glossary.md) · [README.md](README.md) · [operations/deployment.md](operations/deployment.md)

Run **`python __main__.py --help`** or **`python -m research --help`** for full flag
lists, defaults, and copy-paste examples. Shared definitions live in `core/entry_cli.py`.

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
.\scripts\run_research.ps1 -Verbose

# Docker (production — see infra/runtime/examples/ for dev override)
docker compose -f infra/runtime/docker-compose.research.yml `
  -f infra/runtime/examples/docker-compose.prod.research.yml `
  --env-file .env up -d --build
```

Trader Docker:

```powershell
docker compose -f infra/runtime/docker-compose.trader.yml `
  -f infra/runtime/examples/docker-compose.prod.trader.yml `
  --env-file .env up -d --build
```

Full layout: [operations/deployment.md](operations/deployment.md).  
Pre-live gates: [operations/launch-checklist.md](operations/launch-checklist.md).

---

## CLI reference

### Trader — `python __main__.py`

| Flag | Short | Purpose |
|------|-------|---------|
| `--test` | | Test Grok API only; no IBKR, no agent |
| `--verbose` | `-v` | DEBUG logging → `logs/agent.log` |
| `--account MODE` | | `paper` (default) or `live` |
| `--require-research-host` | | Exit if research host heartbeat is stale (production) |
| `--require-daemon` | | Legacy alias for `--require-research-host` |
| `--force-in-process` | | Always run scorer in this process (dev; may double-write) |
| `--no-research` | | Never auto-start scorer |

**Mutually exclusive:** `--require-research-host` (or legacy `--require-daemon`) and
`--force-in-process` cannot be combined.

**Profitability (trader — no IBKR loop):**

| Flag | Purpose |
|------|---------|
| `--config-summary` | Print master ProfitConfig levers and exit |
| `--profile NAME` | Set `PROFIT_PROFILE` before load (`--profit-profile` alias; built-in or evolved) |
| `--simulate [PROFILES]` | Historical backtest; requires `--sim-start` / `--sim-end` |
| `--sim-cash`, `--sim-cycles-per-day`, `--sim-csv` | Backtest equity, cycles/day (default 1), optional CSV |
| `--live-optimize` | Suggest profile from cycle logs (see `--live-optimize-days`, `--live-optimize-output`) |

Full examples and env vars: [simulation-and-optimization.md](simulation-and-optimization.md).

**Environment (trader):**

| Variable | Effect |
|----------|--------|
| `TRADING_MODE` | `paper` (default), `aggressive_paper`, or `live` — must match `IBKR_ACCOUNT_TYPE` (validated at startup) |
| `RISK_PER_TRADE` | Percent of cash per trade (e.g. `1.0` = 1%); live mode capped at 2% |
| `DATABASE_URL` | Postgres DSN (`postgresql://…`); validated when set |
| `TRADER_IN_PROCESS_SCORER=never` | Same hard gate as `--require-research-host` (also `0`, `false`, `off`, `no`) |
| `XAI_API_KEY` / `GROK_API_KEY` | Required unless `--test` |
| `IBKR_ACCOUNT_TYPE` | Overridden by `--account` when passed |

Startup check: `python -c "from core.config import validate_config; print(validate_config() or 'OK')"`
Settings implementation: `core/settings.py` (Pydantic), re-exported via `core/config.py`.

### Research host — `python -m research`

| Flag | Short | Purpose |
|------|-------|---------|
| `--verbose` | `-v` | DEBUG logging → `logs/research.log` |
| `--no-evolution` | | Skip template-evolution background thread |
| `--config-summary` | | Print master ProfitConfig levers and exit |
| `--profile NAME` | | Set `PROFIT_PROFILE` before load (alias: `--profit-profile`; same names as trader) |
| `--profit-profile NAME` | | Same as `--profile` |

---

## Development (single machine)

| Goal | Command |
|------|---------|
| Research scoring only | `python -m research` |
| Trader only (uses in-process scorer if heartbeat stale) | `python __main__.py` |
| Trader, never start local scorer | `TRADER_IN_PROCESS_SCORER=never python __main__.py --require-research-host` (research must be running) |
| Force in-process scorer on trader (no research host) | `python __main__.py --force-in-process` |
| Test Grok connection | `python __main__.py --test` |
| Trader without auto-scorer | `python __main__.py --no-research` |
| Research without template evolution | `python -m research --no-evolution` |

Typical local loop: terminal 1 → `python -m research --verbose`; terminal 2 →
`python __main__.py --verbose`.

### Heartbeat (research host liveness)

| Item | Value |
|------|--------|
| Postgres key (canonical) | `research_host_heartbeat_ts` in `research_config` |
| Legacy key (still read) | `daemon_heartbeat_ts` |
| Written by | `signals/scorer.py` each round via `core.runtime.heartbeat.write_heartbeat` |
| Checked by | Trader boot + each cycle (`is_research_host_alive`) |
| CLI flag | `--require-research-host` (legacy `--require-daemon`) — refuse trader start if heartbeat stale |

---

## Ops scripts (from repo root)

| Script | Purpose |
|--------|---------|
| `python scripts/health.py researcher` | Postgres, heartbeat, token cap, MDA, scoring/evolution (exit 0/1/2) |
| `python scripts/health.py trader` | Same + operating mode; optional `--ibkr-client-id` for TWS probe |
| `python scripts/health.py` | Both sections |
| `python scripts/verify_trader_db.py` | Postgres ping, `init_db`, core tables, heartbeat, token cap |
| `python scripts/smoke_tools.py --platform-only` | Platform checks only (no tools) |
| `python scripts/smoke_tools.py --preflight --client-id <id>` | Platform + IBKR + open_orders summary |
| `python scripts/smoke_tools.py --client-id <id>` | Platform checks + paper-safe tool sweep |
| `python scripts/smoke_tools.py --client-id <id> --all` | Safe + broker-mutating tools |
| `scripts/run_research.ps1` | Windows: same as `python -m research` (`-Verbose`, `-NoEvolution`) |
| `scripts/backup_postgres.ps1` | DB backup (Windows) |
| `scripts/watch_research.ps1` | List `python -m research` PIDs + tail log |
| `python scripts/optimize_profiles.py` | Grid/genetic search over ProfitConfig profiles (historical sim) |
| `python scripts/dashboard.py` | Terminal/HTML summary of `logs/profit_cycles_*.json` |
| `python -m api` | Profit API — `GET /profit_summary`, `GET /status` (port 8787) |
| `python -m infra.status_api` | Status API — `GET /status`, `GET /health/ready` (port 8790) |
| `python scripts/health.py --json` | Full JSON health report (ProfitConfig + alerts) |
| `python scripts/alert_watch.py` | Poll status API; optional `ALERT_WEBHOOK_URL` |
| `python scripts/daily_summary.py` | Daily dashboard + optimizer → tomorrow's `PROFIT_PROFILE` |
| `python scripts/evolve_strategy.py` | Optional: MULTI_AGENT_MODEL reviews 7d logs → review-only `.patch` for prompts/tools |

**Live profile rollback:** when `TRADING_MODE=live`, a new trial profile that draws down ≥10% from its adoption peak is auto-reverted to the previous known-good profile (see `core/profile_rollback.py`, `ABC_PROFILE_ROLLBACK_DRAWDOWN_PCT`).

**Exit codes** (health, verify_trader_db, smoke platform checks): **0** healthy · **1** warnings · **2** failures. Use `--no-color` for plain logs. See [plain-english-glossary.md](plain-english-glossary.md#health-scripts-exit-codes).

---

## Where the code lives

| What you run | Module / file |
|--------------|----------------|
| `python -m research` | `research/__main__.py` → `research/host.py` |
| `python __main__.py` | `__main__.py` → `core/agent.py` (`TradingAgent`) |
| CLI definitions | `core/entry_cli.py` |
| Research log file | `logs/research.log` |
| Trader log file | `logs/agent.log` |

There is **no** `research_daemon.py` at the repository root.

---

## Removed commands (do not use)

These paths were deleted; grep the repo if docs or scripts still mention them:

- `python research_daemon.py`
- `research/daemon.py` (renamed to `research/host.py`)
- `python scripts/check_researcher.py` / `check_trader.py` → use `scripts/health.py`
- `python scripts/smoke_trader_tools.py` (and `smoke_all_tools`, `smoke_order_tools`) → use `scripts/smoke_tools.py`

**See also:** [plain-english-glossary.md](plain-english-glossary.md) · [codebase-layout.md](codebase-layout.md) · [engineering.md](engineering.md) · [operations/deployment.md](operations/deployment.md)
