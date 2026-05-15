# ABC Grok Autonomous Trader — Architecture

**Version**: Production 2026  
**Principle**: Ultra-lean, two-machine decoupled, Grok does all thinking, thin tools, iron-clad risk, cash-only.

## High-Level Diagram

```
┌──────────────────────────────┐          Shared Postgres (single source of truth)
│   Researcher Machine         │◄───────────────────────────────────────┐
│                              │                                        │
│  research_daemon.py          │                                        │
│   • MDA health gate          │                                        │
│   • Hard 100k daily token    │                                        │
│     / activity cap           │                                        │
│   • signals/scorer (all 50+) │                                        │
│   • template_evolution       │                                        │
│   • Writes: signal_scores,   │                                        │
│     hypotheses, briefings,   │                                        │
│     research_config          │                                        │
└──────────────────────────────┘                                        │
                                                                        │
┌──────────────────────────────┐          ┌──────────────────────────┐ │
│   Trader Machine             │          │  Interactive Brokers     │ │
│                              │          │  (TWS/Gateway)           │ │
│  __main__.py                 │◄────────►│  • Real-time streaming   │ │
│   • TradingAgent (ReAct)     │  IBKR    │  • Order execution       │ │
│   • Grok (grok-4.3, temp=0)  │  API     │  • Options, spreads      │ │
│   • ToolExecutor (thin)      │          └──────────────────────────┘ │
│   • SafetyController         │                                        │
│   • Attention / Intuition    │                                        │
│   • Working Memory (curated) │                                        │
│   • Reads everything from DB │                                        │
└──────────────────────────────┘                                        │
                                                                        │
                              MarketData.app (MDA) ◄───────────────────┘
                              (real-time quotes, chains, IV for researcher)
```

## Core Principles (Never Violated)

1. **Two-Machine Decoupling** (strict)
   - Researcher: `research_daemon.py` runs **always and automatically**. Never runs trader code.
   - Trader: `__main__.py` runs **completely independently**. Default = `TRADER_IN_PROCESS_SCORER=never`. Only reads from Postgres.
   - `--require-daemon` or the env var is production. `--force-in-process` is dev-only.

2. **Researcher Hard Boundaries**
   - MDA streaming must be healthy (health probe at startup).
   - Hard daily cap `RESEARCHER_DAILY_TOKEN_CAP` (default 100000). Early warning + shutdown if exceeded.

3. **Determinism**
   - `LLM_TEMPERATURE = 0.0`
   - `LLM_SEED = 42`
   - All Grok calls use these.

4. **Cash-Only, No Margin, No Shorts**
   - All sizing uses `TotalCashValue`.
   - Long puts only for bearish views.

5. **Thin Tools + Thick Brain**
   - All tools in `tools/` are thin wrappers.
   - Grok (ReAct) decides everything in `core/agent.py`.

6. **Safety Layers**
   - `SafetyController` (daily loss, intraday drawdown, LLM cost).
   - Per-turn checks in ReAct loop.
   - Researcher 100k cap + MDA health.
   - IBKR line budget + subscription management.

## Key Modules

- `core/agent.py`: The ReAct brain. `TradingAgent.run_cycle()` + inner turn loop.
- `research_daemon.py`: Always-on researcher process.
- `signals/scorer.py`: Computes all signals + forward returns + IC.
- `core/runtime/`: `safety.py`, `attention.py`, `intuition.py`, `working_memory.py`, `heartbeat.py`.
- `data/`: `data_provider.py` (MDA + IBKR), `cost_tracker.py`, `marketdata_client.py`, `ibkr_quote_source.py`.
- `execution/`: Full IBKR (orders, options spreads, queries, streaming).
- `tools/`: Thin handlers for every action Grok can call.
- `memory/`: Postgres layer (working_memory, hypotheses, execution snapshots, research_config).

## Data Flow (Handoff)

Researcher writes → Postgres → Trader reads every cycle via `StateContextBuilder` + tools (`briefing`, `prior_research`, `open_hypotheses`, `signal_breakdown`).

No direct process communication. Heartbeat in `research_config` + "researcher_daily_usage_*" keys for caps.

## Production Launch (Exact Commands)

See README.md → "Two-Machine Production Deployment" section for the authoritative commands.

Researcher must always be up first. Trader refuses to start without fresh heartbeat when in production mode.

## Diagrams & Rationale

The architecture deliberately splits heavy research (MDA burn, signal computation, evolution) from the low-latency, high-stakes trading loop (Grok decisions + IBKR execution). This prevents one bad research day from affecting trading capital or LLM spend, and allows independent scaling/monitoring of the two machines.

All state that needs to survive restarts lives in Postgres (the only shared resource).

This document + the code comments + the Production Launch Checklist in docs/ together form the complete operational view.

---

*Generated as part of 100% production readiness pass. Update when architecture changes.*