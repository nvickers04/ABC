# ABC Minimal Grok 4.2 Trader — Pure ReAct (my ideal setup)

Built for Grok 4.2 — dynamic liquidity, overnight holds OK, 0.5% risk max.  
Alpha Arena winning philosophy: pure model autonomy + thin tools + iron-clad risk.

## Architecture

```
Grok (ReAct Brain)  →  Tools (thin wrappers)  →  IBKR Execution
     ↑                                              │
     └──────────── state + tool results ←───────────┘
```

- **5-minute cycles** — Grok observes state, calls tools, decides WAIT or TRADE
- **temperature=0.0, seed=42** — deterministic, reproducible
- **0.5% max risk per trade** — enforced in code AND prompt
- **No auto-close** — Grok decides hold time (intraday, overnight, multi-day)
- **No screening layer** — Grok uses raw data tools to find opportunities itself

## Structure

```
├── core/           # The brain
│   ├── agent.py    # Pure ReAct loop
│   ├── grok_llm.py # xAI API wrapper
│   └── config.py   # System prompt + risk constants
├── tools/          # Thin tool wrappers (account, orders, research, options, etc.)
├── data/           # Market data client, data provider, cost tracker, live state
├── execution/      # IBKR core, orders, options, queries
├── config/         # trading.yaml, agent_prompt.txt, ibkr_config.ini
├── __main__.py     # Entry point
├── .env.template   # Copy to .env and fill in
├── requirements.txt
└── pyproject.toml
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/nvickers04/ABC-Application.git
cd ABC-Application
git checkout minimal-grok42-trader
pip install -r requirements.txt

# 2. Configure
cp .env.template .env
# Edit .env: fill GROK_API_KEY + IBKR credentials

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

## Risk Rules (hard-coded)

- Max **0.5%** portfolio equity risk per trade (Python guard blocks violations)
- Min **3:1** reward-to-risk ratio
- Min **75%** confidence required
- **15%** daily loss → emergency flatten
- **$50** daily LLM cost ceiling → halt
- WAIT 95%+ of the time — doing nothing is winning

## Model

Currently using `grok-4-1-fast-reasoning` via xAI's OpenAI-compatible API.  
Change `DEFAULT_MODEL` in `core/grok_llm.py` when Grok 4.2 is available.
