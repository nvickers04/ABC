# Grok 4.20 (xAI)

**Used by:** `core/grok_llm.py` → `core/agent.py`
**Library:** `xai_sdk`
**Model:** `grok-4.20-0309-reasoning`, `temperature=0.0`, `seed=42`
**Process:** Trader only.  The research daemon makes zero LLM calls.
**Auth env:** `XAI_API_KEY` or `GROK_API_KEY`

## What we use it for

The Grok LLM IS the trader.  Each agent cycle:
1. Reads attention triggers, intuition, working memory, current state,
   cost ledger, time/continuity, briefing.
2. Calls Grok with the system prompt + tool schemas.
3. Grok decides which tools to call (data lookups, order placement,
   memory writes).
4. Tool results return; Grok produces the cycle reasoning + actions.

## Why Grok is NOT in the research daemon

1. **Cost.**  Per-token pricing is the dominant cost of running the
   trader.  Adding LLM calls to scoring rounds would multiply cost by
   the number of rounds × the number of decisions per round.
2. **Determinism mismatch.**  Scoring is deterministic; LLM calls are
   probabilistic (we use `temp=0.0, seed=42` to get as close to
   deterministic as the API allows, but it's not guaranteed).
3. **Latency.**  A reasoning call takes seconds.  Scoring rounds need to
   complete in under a minute end-to-end.
4. **Wrong abstraction layer.**  Signals are statistical; reasoning is
   for trade decisions, not score combination.

## Cost considerations

- The trader's intended uptime is **~16 hours per day** (pre-market,
  regular hours, after-hours).
- That uptime is **the dominant cost driver.**  Reducing trader hours is
  the lever for cost control.
- The research daemon runs in parallel during the same hours so the IC
  attribution data is rich enough to be useful.  The daemon costs
  ~$nothing in LLM terms; its cost is purely MDA credits.

## Operational notes

- `temperature=0.0`, `seed=42` for reproducibility within a session.
- Prompt order is fixed: ATTENTION → INTUITION → WORKING MEMORY → state →
  cost → time → continuity → briefing.  Don't reorder casually — it
  changes the model's response distribution even with seed pinning.
- Tool dispatch is via the `HANDLERS` dict in `tools/*.py` registered into
  `tools_executor.py:_REGISTRY`.

## Pointers in the codebase

- LLM client: [`core/grok_llm.py`](../../core/grok_llm.py)
- Agent loop: [`core/agent.py`](../../core/agent.py)
- Tool registry: [`tools/tools_executor.py`](../../tools/tools_executor.py)
