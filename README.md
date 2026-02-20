# ABC — Grok 4.20 Autonomous Trader

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
- **Cash-only** — uses CashBalance / AvailableFunds, no margin, no short selling
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

## Environment Variables (required)

| Variable | Description |
|----------|-------------|
| `GROK_API_KEY` | xAI API key (or `XAI_API_KEY`) |
| `IBKR_HOST` | IBKR TWS/Gateway host (default: 127.0.0.1) |
| `IBKR_PORT` | IBKR port — 7497=paper, 7496=live |
| `IBKR_CLIENT_ID` | Client ID (default: 1) |
| `PAPER_MODE` | `True` (default) or `False` for live |
| `PAPER_AGGRESSIVE` | `true` for stress-test mode (5% risk, 1.5:1 R:R, 50% confidence, forces complex options) |
| `RISK_PER_TRADE` | % of cash balance per trade (default: 1.0) |
| `CASH_ONLY` | `true` (default) — enforce cash-only, no shorts |

## Stress Test Mode (Paper)

```bash
# Super aggressive paper mode — forces complex options orders, tests every order type
PAPER_AGGRESSIVE=true python __main__.py
```

When `PAPER_AGGRESSIVE=true`:
- Risk per trade: **5%** of cash (vs 1% normal)
- Min R:R: **1.5:1** (vs 2:1 normal)
- Min confidence: **50%** (vs 65% normal)
- Forces complex options: spreads, iron condors, calendars, straddles, diagonals
- Auto-suggests spread upgrades when simple long calls/puts are selected
- Evaluates 3–5 setups per cycle (vs 1 normal)
- Tests all order types: bracket, trailing stop, adaptive, midprice, VWAP
- 42-symbol watchlist scanned every cycle

## Risk Rules

- Max **RISK_PER_TRADE%** of cash balance risk per trade (default 1.0%, configurable in `.env`)
- **Cash-only** — no margin, no short selling (use long puts for bearish views)
- Min **2:1** reward-to-risk ratio (1.5:1 in PAPER_AGGRESSIVE mode)
- Min **65%** confidence required (50% in PAPER_AGGRESSIVE mode)
- **15%** daily loss → emergency flatten
- **$50** daily LLM cost ceiling → halt
- Turn limit: nudge at turn 8, hard max at turn 10 per cycle
- Rolling context summary every 5 turns to keep reasoning sharp

## Model

Currently using `grok-4-1-fast-reasoning` via xAI's OpenAI-compatible API.  
Change `DEFAULT_MODEL` in `core/grok_llm.py` when Grok 4.20 is available.
