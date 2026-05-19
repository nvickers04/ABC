# ABC — Grok Autonomous Trader

**Ultra-lean version — Grok does ALL screening, analysis, and decisions.**

Cash-only, configurable risk, dynamic liquidity, overnight holds OK.  
Pure model autonomy + thin tools + iron-clad risk. No margin, no shorts.

## Architecture

```
Grok (ReAct Brain)  →  Tools (thin wrappers)  →  IBKR Execution
     ↑                                              │
     └──────────── broker queries ←─────────────┘
```

- **1-minute cycles** — Grok observes state, calls tools, decides WAIT or TRADE
- **Cash-only** — uses TotalCashValue for sizing (never AvailableFunds which includes margin), no short selling
- **MarketData.app** — real-time quotes, candles, options chains, IV, fundamentals via REST API
- **IBKR execution** — orders placed through Interactive Brokers (paper on port 7497)
- **temperature=0.0, seed=42** — deterministic, reproducible
- **Configurable risk** — `RISK_PER_TRADE` in `.env` (default 1.0% of cash)
- **No auto-close** — Grok decides hold time (intraday, overnight, multi-day)

## Structure

```
├── core/              # Agent loop, config, runtime, quality policy
│   ├── agent.py       # ReAct cycle
│   ├── quality/       # QualityMatrix (risk, tool gates, provenance)
│   └── runtime/       # Safety, scheduler, operating context, WM routing
├── tools/             # Thin tool wrappers (account, orders, research, …)
├── memory/            # Postgres persistence, working memory, migrations
├── data/              # Market data, broker gateway, cost tracker
├── execution/         # IBKR orders and queries
├── research/          # Universe config, simulator; daemon in research/daemon.py
├── signals/           # Scorers, combiner, templates (used by research host)
├── docs/              # Ops and engineering (see docs/README.md)
├── scripts/           # run_research.ps1, health.py, smoke_tools.py, verify_trader_db.py
├── __main__.py        # Trader: python __main__.py  (Grok + IBKR)
└── pyproject.toml     # Pytest config; research host: python -m research
```

**Documentation:** [docs/README.md](docs/README.md) · **[Entry points](docs/entry-points.md)** (what to run on each host)

## Entry points (summary)

| Role | Production command | Code |
|------|------------------|------|
| **Research host** | `python -m research` | `research/daemon.py` |
| **Trader** | `python __main__.py --require-daemon` | `__main__.py` → `core/agent.py` |
| **Health** | `python scripts/health.py` | `scripts/health.py` |

Details, dev single-machine modes, Docker, and scripts: **[docs/entry-points.md](docs/entry-points.md)**.

## Quick Start

```bash
git clone https://github.com/nvickers04/ABC.git
cd ABC
pip install -r requirements.txt
cp .env.template .env
# Edit .env: GROK_API_KEY, IBKR credentials, DATABASE_URL (Postgres)
```

Start **TWS or IB Gateway** (paper port 7497) before the trader.

### Local development (two terminals, recommended)

Matches production: research host scores; trader reads Postgres and runs Grok.

```bash
# Terminal 1 — research host (no Grok, no IBKR orders)
python -m research

# Terminal 2 — trader (paper)
python __main__.py --verbose
```

Health: `python scripts/health.py`

### Production (two machines)

See **[docs/entry-points.md](docs/entry-points.md)** — research: `python -m research`; trader: `python __main__.py --require-daemon` with `TRADER_IN_PROCESS_SCORER=never`.

### Trader-only shortcuts

```bash
python __main__.py --test          # Grok connection only
python __main__.py                 # trader (in-process scorer if no research heartbeat)
python __main__.py --account live  # live — use launch checklist first
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROK_API_KEY` | xAI API key (or `XAI_API_KEY`) |
| `TRADING_MODE` | **`aggressive_paper`** · **`paper`** (default) · **`live`** — controls risk, prompt, and IBKR port |
| `IBKR_HOST` | IBKR TWS/Gateway host (default: 127.0.0.1) |
| `IBKR_PORT` | IBKR port — auto-set by TRADING_MODE (7497=paper, 7496=live) |
| `IBKR_CLIENT_ID` | Client ID (default: 1) |
| `RISK_PER_TRADE` | % of cash balance per trade (override mode default) |
| `CASH_ONLY` | `true` (default) — enforce cash-only, no shorts |

## Trading Modes

Set `TRADING_MODE` in `.env`:

```bash
# Normal paper trading (default)
TRADING_MODE=paper python __main__.py

# Stress-test mode — forces complex options, tests every order type
TRADING_MODE=aggressive_paper python __main__.py

# Live trading — real money, strict rules, port 7496
TRADING_MODE=live python __main__.py
```

| Setting | `aggressive_paper` | `paper` | `live` |
|---------|-------------------|---------|--------|
| Risk/trade | 5% | 1% | 1% |
| Min R:R | 1.5:1 | 2:1 | 2.5:1 |
| Min confidence | 50% | 65% | 70% |
| IBKR port | 7497 (paper) | 7497 (paper) | 7496 (live) |
| Complex options | Preferred when gate open | Normal | Conservative |
| Auto-suggest spreads | Yes | No | No |
| Tickers evaluated | 3–5 per cycle | 1–2 | 1 |
| 42-symbol scan | Every cycle | Every cycle | Every cycle |

## Risk Rules

- Max **RISK_PER_TRADE%** of cash balance per trade (mode-dependent, overridable in `.env`)
- **Cash-only** — no margin, no short selling (use long puts for bearish views)
- **15%** daily loss → emergency flatten
- **$50** daily LLM cost ceiling → halt
- Turn limit: nudge at turn 8, hard max at turn 10 per cycle
- Rolling context summary every 5 turns to keep reasoning sharp

**Production**: [docs/operations/launch-checklist.md](docs/operations/launch-checklist.md) · [docs/operations/deployment.md](docs/operations/deployment.md)

## Model

The trader uses **Grok** (xAI) via the native SDK (`xai-sdk >= 1.8.0`). Exact API
model slugs are defined in **`core/grok_llm.py`** only (`REASONING_MODEL` for the
ReAct loop, `MULTI_AGENT_MODEL` for the optional `research()` tool). Documentation
refers generically to Grok so you do not chase version strings across the repo when
xAI renames models.

## Tools

Tool interfaces are normalized for Grok tool-calling usage:

- ATR accepts string resolutions (for example `daily`, `D`, `5`) without failing on `int()` parsing.
- `bracket_order` accepts `stop_loss` + `take_profit` (entry price inferred from explicit input or live quote).
- Options tools normalize rights/side aliases (`C`/`CALL`, `P`/`PUT`) internally.
- Options pre-validation checks expiration/strike existence before IBKR placement and returns clean contract-not-found errors.
- Tool outputs are standardized to: `success`, `data`, `error`, `is_realtime`, `data_warning`.
- Cash-only order failures return explicit `CASH-ONLY: insufficient cash` with available cash shown.

## Two-machine production

Research host runs `python -m research` only (never `__main__.py`).
Trader host runs `python __main__.py --require-daemon` with `TRADER_IN_PROCESS_SCORER=never`.
Both share one **PostgreSQL** database.

Full runbooks: **[docs/operations/deployment.md](docs/operations/deployment.md)** · [postgres](docs/operations/postgres.md) · [independent mode](docs/operations/independent-mode.md)

Quick health: `python scripts/health.py`
