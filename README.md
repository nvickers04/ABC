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
├── core/           # The brain
│   ├── agent.py    # Pure ReAct loop
│   ├── grok_llm.py # xAI API wrapper
│   └── config.py   # System prompt + risk constants (reads .env)
├── tools/          # Thin tool wrappers (account, orders, research, options, etc.)
├── data/           # Market data client, data provider, cost tracker, broker gateway
├── execution/      # IBKR core, orders, options, queries
├── research/       # Autoresearch-style strategy evolution package
├── __main__.py     # Entry point
├── .env.template   # Copy to .env and fill in
├── requirements.txt
└── pyproject.toml
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/nvickers04/ABC.git
cd ABC
pip install -r requirements.txt

# 2. Configure
cp .env.template .env
# Edit .env: fill GROK_API_KEY + IBKR credentials
# Adjust RISK_PER_TRADE (default 1.0% — set 0.5 for ultra-conservative, 2.0 for testing)

# 3. Start TWS or IB Gateway (paper trading on port 7497)

# 4. Run (paper mode by default)
python __main__.py

# Test Grok connection first:
python __main__.py --test

# Verbose logging:
python __main__.py --verbose

# Live trading (when ready):
python __main__.py --account live
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

**Production**: See `docs/PRODUCTION_LAUNCH_CHECKLIST.md` (pre-flight, Docker, live gate, monitoring, rollback). Use `--require-daemon` + separate research host for best isolation. Full Docker trader support added in `infra/runtime/`.

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

## Two-Machine Production Deployment (Mandatory for Live)

The system is designed for **strict separation** between a dedicated **Researcher machine** and a **Trader machine** sharing a Postgres database.

### Researcher Machine (Market Data + Signals + Evolution)
- Runs **only** `research_daemon.py` (never `__main__.py`).
- Always starts the scorer + template evolution automatically.
- Primary data source: MarketData.app (real-time, high credit burn).
- **Hard boundaries enforced**:
  - MDA streaming health check at startup (SPY quote probe). Daemon exits with code 3 if unhealthy.
  - Hard daily cap `RESEARCHER_DAILY_TOKEN_CAP=100000` (default). Tracks activity per UTC day in `research_config`. Approaching 85% → warning; hitting cap → critical shutdown (exit 4). No unbounded research allowed.
- Docker (recommended):
  ```bash
  docker compose -f infra/runtime/docker-compose.research.yml --env-file .env up -d --build
  ```
- Manual: `python research_daemon.py` (or with `--no-evolution` only for debugging).
- Environment: Set `IBKR_QUOTES_ENABLED=0` (enforced internally). Never run trader code here.

### Trader Machine (Grok ReAct + IBKR Execution)
- Runs only `__main__.py`.
- **Completely independent** — reads everything (signals, hypotheses, briefings, working memory) from shared Postgres.
- **Default is fully decoupled**: `TRADER_IN_PROCESS_SCORER=never` (or in Docker compose). Trader **refuses** to start an in-process scorer and hard-exits if no fresh research_daemon heartbeat.
- Use `--require-daemon` (or the env var) for production. `--force-in-process` is **dev only**.
- Docker (recommended):
  ```bash
  docker compose -f infra/runtime/docker-compose.trader.yml --env-file .env up -d --build
  ```
- Manual production launch:
  ```bash
  TRADER_IN_PROCESS_SCORER=never python __main__.py --require-daemon
  ```
- IBKR streaming (stocks + options) is managed here with line-budget protection, LRU eviction, explicit `cancelMktData`, and reconnection handling. Researcher never holds IBKR market data lines.

### Exact Launch Commands (Production)

**Researcher host** (one terminal / systemd / Docker):
```bash
# Docker (preferred)
docker compose -f infra/runtime/docker-compose.research.yml --env-file .env up -d --build

# Or direct
python research_daemon.py
```

**Trader host** (separate machine / container):
```bash
# Docker (preferred — defaults to fully decoupled)
docker compose -f infra/runtime/docker-compose.trader.yml --env-file .env up -d --build

# Or direct (production)
TRADER_IN_PROCESS_SCORER=never python __main__.py --require-daemon --verbose
```

**Monitoring token / MDA usage on researcher**:
- Watch logs for "RESEARCHER DAILY TOKEN USAGE APPROACHING CAP" and "MDA HEALTH CHECK".
- Query: `SELECT key, value FROM research_config WHERE key LIKE 'researcher_daily_usage_%';`
- The daemon will not start new heavy rounds once the 100k cap is reached.

This architecture guarantees the researcher can run 24/7 on one host (with its own MDA token) while the trader runs on another (with IBKR + Grok) with zero risk of double-writes or silent in-process scorer fallback.

## Risk Rules

- Max **RISK_PER_TRADE%** of cash balance per trade (mode-dependent, overridable in `.env`)
- **Cash-only** — no margin, no short selling (use long puts for bearish views)
- **15%** daily loss → emergency flatten
- **$50** daily LLM cost ceiling → halt
- Turn limit: nudge at turn 8, hard max at turn 10 per cycle
- Rolling context summary every 5 turns to keep reasoning sharp

## Model
