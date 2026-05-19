# Data sources and process ownership

Which external providers we use, what each cannot do, and which process owns each feed.

**See also:** [entry-points.md](entry-points.md) · [operations/deployment.md](operations/deployment.md) · [plain-english-glossary.md](plain-english-glossary.md)

## Overview

```
┌──────────────────────────────┐    ┌──────────────────────────────┐
│  python -m research          │    │  __main__.py (trader)         │
│  MDA + yfinance              │    │  IBKR + Grok (xAI)            │
│  Writes signal_scores (50+)  │    │  Writes auction_imbalance     │
│  Never IBKR or LLM           │    │  + orders, WM, attention      │
└──────────────┬───────────────┘    └──────────────┬───────────────┘
               └──────────── Postgres (shared) ─────┘
```

| Question | Answer |
|----------|--------|
| Research host on IBKR? | **No** — line cap, competes with trader for `client_id` |
| Trader on MDA? | Rarely — reads pre-scored rows from DB |
| Who owns `auction_imbalance`? | **Trader** — needs IBKR generic tick 225 |
| Who runs the LLM? | **Trader only** — cost, latency, determinism |
| Research without trader? | OK for IC backfill / signal testing |
| Trader without research host? | **Dev only** — stale composites in production |

IC needs live forward-return pairs; **run the research host whenever the trader runs** in production.

---

## MarketData.app (MDA)

**Code:** `data/marketdata_client.py`, `data/data_provider.py` · **Process:** research host (primary)  
**Auth:** `MARKETDATA_TOKEN` · **Plan:** ~100k credits/day

**Used for:** bulk quotes, historical candles (1m–D), option chains with Greeks, EOD quotes.

**Not used for:** real-time streaming (use IBKR), auction imbalance, Level 2, fundamentals, news.

**Cost:** option chains dominate (~1 credit/strike). Universe rounds ~120 credits; 30s cadence stays within budget.

**Signals (~40):** momentum, mean_reversion, iv_rank, option_flow, market_breadth, etc. (see `signals/` registry).

---

## Interactive Brokers (IBKR)

**Code:** `execution/ibkr_*.py`, `data/ibkr_quote_source.py` · **Library:** `ib_insync`  
**Process:** **trader only** · **Ports:** 7497 paper, 7496 live

**Used for:** orders, positions, real-time L1, **auction imbalance** (generic tick `"225"`).

**Not on research host:** 90/100 line budget for trader; stateful subscriptions; historical throttling; options Greeks cost extra vs MDA.

**Signals:** `auction_imbalance` only — trader writes rows to `signal_scores` during the session.

**Ops:** TWS/Gateway must run; midnight ET restart handled in `execution/ibkr_core.py`.

---

## Yahoo Finance (yfinance)

**Code:** `data/data_provider.py` · **Process:** research host · **Auth:** none (public scrape)

**Used for:** fundamentals, insider flow, institutional ownership, earnings calendar, news, short interest.

**Not for:** intraday prices or options Greeks (stale/wrong — use MDA).

**Signals:** `debt_health`, `valuation`, `insider_flow`, `news_sentiment`, `event_proximity`, etc.

---

## Grok (xAI)

**Code:** `core/grok_llm.py`, `core/agent.py` · **Process:** trader only  
**Auth:** `XAI_API_KEY` / `GROK_API_KEY` · **Models:** `REASONING_MODEL`, `MULTI_AGENT_MODEL` in `grok_llm.py`

The LLM is the trader brain each cycle: state → tools → decisions. **Zero LLM calls on the research host.**

Prompt order is intentional: ATTENTION → INTUITION → WORKING MEMORY → state → cost → briefing.

---

## Code pointers

| Area | Path |
|------|------|
| MDA client | `data/marketdata_client.py` |
| IBKR core | `execution/ibkr_core.py` |
| IBKR quotes | `data/ibkr_quote_source.py` |
| Research host | `python -m research` (`research/host.py`) |
| Trader | `__main__.py` |
| Combiner / IC | `signals/combiner.py`, `signals/scorer.py` |
| Heartbeat | `core/runtime/heartbeat.py` |

---

## Related

- [plain-english-glossary.md](plain-english-glossary.md) — research host, MDA, IBKR, Grok
- [operations/independent-mode.md](operations/independent-mode.md) — when research host is down
