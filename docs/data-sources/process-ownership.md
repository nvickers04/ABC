# Process Ownership

This is the **single source of truth** for "which process owns which
data source, and why."  Read this before debating architecture.

## The two processes

```
┌──────────────────────────────┐    ┌──────────────────────────────┐
│  research_daemon.py          │    │  __main__.py  (the trader)    │
│  ────────────────────        │    │  ────────────────────         │
│  Owns:                       │    │  Owns:                        │
│    • MDA (all)               │    │    • IBKR (all)               │
│    • yfinance (all)          │    │    • Grok / xAI               │
│  Writes:                     │    │  Writes:                      │
│    • signal_scores           │    │    • signal_scores            │
│      (50 of 51 signals)      │    │      (auction_imbalance only) │
│    • composite_scores        │    │    • orders, fills            │
│    • signal_symbol_ic        │    │    • working_memory           │
│  Cadence:                    │    │    • attention_triggers       │
│    regular  30s              │    │  Cadence:                     │
│    extended 300s             │    │    LLM-driven                 │
│    overnight 1800s           │    │  Uptime: ~16h/weekday         │
│  Uptime: matches trader      │    │  Cost driver: LLM tokens      │
└──────────────────────────────┘    └──────────────────────────────┘
                  ┃                                   ┃
                  ┃        ┌──────────────────┐       ┃
                  ┗━━━━━━━━┫   SQLite (WAL)   ┣━━━━━━━┛
                           │  shared write    │
                           │  target          │
                           └──────────────────┘
```

## The decision matrix

| Question | Answer | Why |
|---|---|---|
| Should the daemon connect to IBKR? | **No.** | Line cap (90/100), no need for sub-second on the daemon side, would compete with the trader for `client_id=1`. |
| Should the trader connect to MDA? | Rarely.  Only if it needs ad-hoc data the daemon hasn't pre-computed. | Daemon already does this work and writes to SQLite — trader reads from there. |
| Who owns auction_imbalance? | **Trader.** | It needs IBKR streaming, which only the trader has.  Trader writes directly into `signal_scores`. |
| Who runs the LLM? | **Trader only.** | Cost; determinism; latency. |
| What runs when only the daemon is up? | The 50 MDA signals score normally; auction_imbalance abstains until trader is up. | Acceptable — auction_imbalance only fires for ~15 min/day anyway. |
| What runs when only the trader is up? | LLM cycles work; combiner reads whatever signal_scores exist (possibly stale). | Acceptable but degrades quality.  Daemon should be up whenever trader is. |
| What runs when both are down? | Nothing.  Manual restart required. | We chose against complex orchestration; supervisord/systemd is overkill for one workstation. |

## The IC argument (why daemon uptime matches trader uptime)

We considered training signal weights on historical data.  We rejected
that because:

1. ~12 of 51 signals are **live-only** — no historical equivalent exists
   (auction_imbalance, news_sentiment, option_flow, gamma_exposure,
   put_call_ratio, institutional_flow, insider_flow, short_interest,
   earnings_momentum, event_proximity, quote_stability, spread_dynamics).
2. The combiner needs **per-signal IC** AND **per-signal-per-symbol IC**
   to weight scores properly.  Both come from the daemon's persistence
   loop, not from history.
3. IC convergence requires hundreds of (score, forward-return) pairs per
   signal.  Faster cadence + longer uptime = faster convergence = less
   garbage signal in the composite.

Therefore: **the daemon should run whenever the trader runs.**  Idle
daemon means stale IC means worse trading decisions.

## When to break this rule

You can run the daemon **without** the trader:
- Backfilling IC after a clean DB.
- Testing new signals in isolation without paying LLM costs.
- Overnight runs to accumulate IC without trading.

You can run the trader **without** the daemon, but:
- Composite scores will be based on whatever's in `signal_scores`
  (possibly very stale).
- The trader cycles will still execute, place orders, etc.
- This is **dev-only**; in production they should both be up.

## Pointers

- Daemon entry: [`research_daemon.py`](../../research_daemon.py)
- Trader entry: [`__main__.py`](../../__main__.py)
- Cadence rules: [`core/runtime/cadence.py`](../../core/runtime/cadence.py)
- Heartbeat (used to detect daemon liveness): [`core/runtime/heartbeat.py`](../../core/runtime/heartbeat.py)
- Combiner: [`signals/combiner.py`](../../signals/combiner.py)
- Per-symbol IC: [`signals/scorer.py`](../../signals/scorer.py) → `_safe_update_per_symbol_ic`
