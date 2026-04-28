# Cognitive Architecture Plan

**Status:** Design locked. Implementation pending.
**Scope:** Everything below EXCEPT the per-symbol IC system, which is held for the next conversation and will get its own plan doc.
**Date:** 2026-04-28

---

## 0. Why this exists

The agent is competent at one cycle in isolation but has **no decision continuity**. Observed failure mode (UNH cycle 3→4): the agent issued a PASS verdict on UNH and bought it 47 seconds later in the next cycle, with no recollection that it had just rejected the same name. Each cycle starts from a blank slate, re-derives the world from scratch, and contradicts itself.

We are not adding "more data." We are adding the cognitive layers that let the agent **carry intent forward**, **notice what it asked itself to notice**, and **prioritize what's interesting** instead of round-robin scanning.

Six layers, in build order:
1. **Research daemon decoupling** — make scoring/templates/IC their own process so the trader doesn't share its turn with them
2. *(Per-symbol IC system — DEFERRED, separate plan)*
3. **Working memory** — structured short-term store the agent owns
4. **Attention** — triggers the agent registers, evaluator that fires them
5. **Intuition** — naive global ranker that surfaces the most interesting symbols each cycle
6. **System prompt rewrite** — identity, present-moment principle, continuity-vs-change clause
7. **`signal_breakdown(symbol)` tool** — on-demand zoom-in

---

## 1. Research Daemon Decoupling

### Problem
`signals/scorer.py`, `signals/template_evolution.py`, and the agent cycle all run in `__main__.py` as concurrent tasks/threads sharing one process and one DB handle. The trader cycle has no isolation from research load. Worse, when we add per-symbol IC computation, the cost will grow and we want to tune research cadence independently of trading cadence.

### Design
Split research into its own process:

- **`research_daemon.py`** (~50 lines, repo root): standalone entry point that runs the scorer loop, template_evolution, and (later) per-symbol IC. Owns its own DB connection, its own DataProvider instance.
- **`core/runtime/cadence.py`** (~30 lines): session-aware sleep helper.
  - Regular hours (09:30–16:00 ET): 30s between rounds
  - Extended hours (04:00–09:30, 16:00–20:00 ET): 5 min
  - Overnight: 30 min
- **Heartbeat**: `research_config` table gets a `last_heartbeat_ts` row. Daemon updates it each round. Agent reads it on startup and every cycle.
  - If heartbeat is fresh (<60s), agent **skips** its own scorer task — daemon owns it.
  - If heartbeat is stale, agent falls back to running scorer in-process (single-process dev mode still works).
- **DB**: SQLite WAL mode already supports concurrent readers + one writer. Daemon and agent both read; only one writes to a given table at a time (daemon owns signal_*, template_*; agent owns trades, working_memory).

### Cost analysis (back-of-envelope)
~120 MarketData credits/round × 1 round/30s × 6.5h regular = ~94k credits regular
+ ~120 × 1/300s × 8h extended = ~12k
+ ~120 × 1/1800s × 9.5h overnight = ~2k
**Total ≈ 108k credits/day.** Within Trader plan budget with headroom.

### Files
- NEW `research_daemon.py`
- NEW `core/runtime/cadence.py`
- MODIFY `core/runtime/research_config.py` (or wherever the config table lives) — add heartbeat read/write
- MODIFY `__main__.py` — heartbeat check, conditional scorer task spawn
- DOCS update for "how to run" (two processes now)

### Build first because
Everything below benefits from research running on its own clock. And the per-symbol IC work in the next conversation will live in this daemon.

---

## 2. Working Memory

### Problem
The agent has no place to write down what it's thinking. Verdicts evaporate. Theses don't survive a cycle. There is no "I already looked at this 4 minutes ago" awareness.

### Principle
**Working memory is monologue, not data.** The agent writes natural-language entries to it. We do not store quotes, prices, or signal scores there — those are queried fresh each cycle. Working memory holds **what the agent decided, what it's watching for, and why**.

### Schema (5 sections, all in-process dict, persisted to `working_memory` table for crash recovery)

| Section          | Cap | Default expiry           | Per-entry budget |
|------------------|-----|--------------------------|------------------|
| `open_theses`    | 8   | EOD                      | ~200 tok         |
| `recent_verdicts`| 12  | 30 min (PASS), 2h (BUY)  | ~80 tok          |
| `watching_for`   | 10  | 60 min                   | ~120 tok         |
| `regime_notes`   | 5   | EOD                      | ~150 tok         |
| `lessons_today`  | 8   | EOD                      | ~150 tok         |

**Total render budget: ~5400 tokens.** Everything above is a default; the agent can override expiry per entry.

### Eviction
Oldest-expires-first when a section hits cap. No LRU, no scoring — simplicity wins.

### Components
- **`memory/working_memory.py`** (~200 lines)
  - `WorkingMemory` class: `add(section, entry, expires_at=None)`, `clear(section, entry_id=None)`, `snapshot()`, `render()`
  - **Curator**: runs at top of each cycle. Drops expired entries. Cheap.
  - **Renderer**: produces the `WORKING MEMORY` block injected into the cycle prompt.
- **Persistence**: `working_memory` table (id, section, entry_text, created_ts, expires_ts, metadata_json). Survives crashes; pruned to current day on startup.

### Files
- NEW `memory/working_memory.py`
- NEW migration in `memory/__init__.py` for `working_memory` table

---

## 3. Working Memory Tools

The agent **writes** to working memory through tools. It does **not** need a read tool — the rendered block is auto-injected at the top of every cycle prompt.

- `update_working_memory(section, entry, expires_in_minutes=None)`
- `clear_working_memory_entry(section, entry_id=None)` — clear one entry, or whole section if id omitted

### Why no read tool
Reading is automatic and free. A read tool would tempt the agent to call it mid-cycle and burn a turn for information it already has.

### Files
- NEW `tools/tools_working_memory.py` (~80 lines)
- MODIFY `tools/tools_executor.py` — register the two specs
- MODIFY `tools/tools_multiagent.py` (or wherever tools are dispatched) — wire handlers

---

## 4. Attention Layer

### Problem
The agent says "I'll watch UNH for a break above 530" — and then never does. The next cycle, UNH is just one of 200 symbols round-robined.

### Design
When the agent writes a `watching_for` entry, the curator parses it into a structured trigger:
```
{symbol, condition: "above"|"below"|"crosses"|"iv_spike"|..., threshold, confirm_with: [...]}
```
Triggers are stored in an `attention_triggers` table. Cap: **10 active triggers** (oldest evicted).

**Evaluator**: hooks into the scorer round (or runs piggyback after it). For each fresh quote/score, checks active triggers. When a trigger fires, it posts to the existing `wake_events` bus.

**Render**: at the **very top** of the cycle prompt, before anything else:
```
⚡ ATTENTION
- UNH crossed 530.15 at 10:42 (you said "watch for break above 530, confirm with volume")
  Volume confirm: YES (1.8× 20d avg)
- TSLA IV crossed +5pt threshold at 10:39
```

### Trigger parsing
LLM-side: when the agent writes a `watching_for` entry, it can include a structured trailer like `[trigger: UNH > 530 + vol]` and the curator parses that. If absent, we attempt a regex-based parse and fall back to "narrative only" (no trigger registered, just a memory note).

### Files
- NEW `core/runtime/attention.py`
- NEW `attention_triggers` table migration
- MODIFY `signals/scorer.py` — call `attention.evaluate(round_results)` after each round
- REUSE `core/wake_events.py`

---

## 5. Intuition Layer (naive / global IC version)

### Problem
The agent currently round-robins symbols. There's no sense of "which 5 names are most interesting *right now*." The combiner produces composite scores but the agent doesn't see a ranked list.

### Naive design (this plan — per-symbol IC version is next)
**`core/runtime/intuition.py`** (~120 lines): per cycle, compute an attention score for each symbol:

```
attention_score(symbol) =
    |composite_score|                              # how strong the consensus is
  × Σ over signals( |signal_score| × |IC| × weight )  # how much we trust the contributing signals (global IC for now)
  + |novelty_bonus|                               # delta vs last cycle's composite
```

Render the **INTUITION** block right after ATTENTION:
```
INTUITION (top 5 by attention score)
1. UNH  composite=+1.42  novelty=+0.31  drivers: momentum(+), gap(+), iv_rank(-)
2. NVDA composite=-1.18  novelty=-0.05  drivers: mean_reversion(+), short_interest(-)
...
```

This is **decision support, not decision making**. The agent still chooses what to do; we just stop making it scan blindly.

### Why naive version first
- Per-symbol IC needs schema work + computation refactor + warmup data → that's the next conversation
- The naive version using global IC weights is enough to validate that ranking-then-rendering changes agent behavior
- Easy to swap the IC source from "global" to "per-symbol with regime conditioning" later — same interface

### Files
- NEW `core/runtime/intuition.py`
- MODIFY agent cycle prompt builder — inject INTUITION block

---

## 6. System Prompt Rewrite

The current prompt makes the agent dutiful but not opinionated, and gives it no instruction about continuity. Replace the opening with:

```
You are an optimistic, truth-seeking trader. You look for opportunity, but
you tell yourself the truth about what you see. Conviction without evidence
is gambling; evidence without conviction is paralysis.

The only trade that matters is the next one. Past P&L is a fact, not a feeling.
Don't avenge losers. Don't reward winners with size you wouldn't otherwise take.

Continuity matters. If you said something 5 minutes ago, you owe yourself
either to act on it or to explicitly change your mind and say why. Don't
silently contradict yourself.
```

### Files
- MODIFY `core/agent.py` (or wherever the system prompt lives)

---

## 7. `signal_breakdown(symbol)` Tool

For when the INTUITION block surfaces a name and the agent wants to zoom in without re-running the whole research stack.

Returns: composite score, top 8 signal contributors with (score, IC, weight, contribution), regime tag, last update timestamp.

### Files
- NEW handler in `tools/tools_research.py`
- MODIFY `tools/tools_executor.py` — register spec

---

## 8. Cycle Prompt Order

Every agent cycle, the prompt is assembled in this order:

1. **⚡ ATTENTION** — what fired since last cycle
2. **INTUITION** — top 5 by attention score
3. **WORKING MEMORY** — open_theses, recent_verdicts, watching_for, regime_notes, lessons_today
4. **System prompt** (the rewritten one)
5. **Tool specs**

Rationale: attention and intuition are *time-sensitive*; memory is *contextual*; system prompt is *stable*. Read top-down, the agent encounters "what just happened" before "who I am."

---

## 9. Build Sequence

| # | Step                                                           | Blocks |
|---|----------------------------------------------------------------|--------|
| a | Research daemon split + heartbeat                              | —      |
| b | **Per-symbol IC system** *(SEPARATE PLAN — next conversation)* | a      |
| c | Working memory store + curator + renderer                      | —      |
| d | Working memory tools                                           | c      |
| e | Attention triggers + evaluator + render                        | c, d   |
| f | Naive intuition ranker (uses global IC)                        | —      |
| g | `signal_breakdown` tool                                        | —      |
| h | System prompt rewrite                                          | c, e, f|

Steps c, f, g, h can proceed in parallel with a once b is started, since they don't depend on per-symbol IC.

---

## 10. Explicit non-goals (design discipline)

These were considered and **rejected** for v1. Do not slip them in:

- ❌ **Auto-thesis-insertion**: tools do not silently write to working memory. The agent writes to memory only through explicit tool calls. (Otherwise memory becomes log spam.)
- ❌ **Reversal blocking**: we do not refuse to let the agent reverse a recent verdict. Continuity is enforced by *visibility* (the recent verdict is in the prompt), not by lockout. If it reverses, fine — the system prompt asks it to say why.
- ❌ **Cross-session memory**: working memory expires EOD by default. We are not building long-term episodic memory yet. Lessons that should persist get promoted manually to docs.
- ❌ **Vector store / embeddings**: no semantic retrieval. Working memory is small, structured, fully rendered every cycle.
- ❌ **Caching market data in working memory**: prices, quotes, IV — always fetched fresh. Memory holds *interpretations*, not *observations*.
- ❌ **Cross-agent memory sharing**: research daemon does not write to working memory. The trading agent owns it exclusively.

---

## 11. File Inventory

### New files
- `research_daemon.py`
- `core/runtime/cadence.py`
- `core/runtime/attention.py`
- `core/runtime/intuition.py`
- `memory/working_memory.py`
- `tools/tools_working_memory.py`

### Modified files
- `__main__.py` — heartbeat check
- `core/agent.py` — system prompt rewrite, prompt assembly order
- `core/runtime/research_config.py` — heartbeat read/write
- `memory/__init__.py` — `working_memory` and `attention_triggers` migrations
- `signals/scorer.py` — attention evaluator hook
- `tools/tools_executor.py` — register new tool specs
- `tools/tools_multiagent.py` — dispatch new tools
- `tools/tools_research.py` — `signal_breakdown` handler

### Deferred (next conversation)
- Per-symbol IC schema (`signal_symbol_ic` table)
- Per-symbol IC computation refactor
- Regime-conditioned IC
- Time-of-day conditioning
- Template→signal feedback loop
- Intuition ranker upgrade from global IC → per-symbol IC

---

## 12. Open questions to resolve before coding

1. Heartbeat staleness threshold — is 60s right? (Daemon round is ~30s.)
2. Working memory persistence — do we restore from DB on startup, or always start fresh? (Lean: restore today's entries only.)
3. Attention trigger DSL — full structured fields vs. regex parse vs. LLM-trailer-only? (Lean: support all three, prefer structured trailer.)
4. INTUITION top-N — 5 fixed, or scaled to active universe size? (Lean: 5 fixed, simpler.)
5. Should `signal_breakdown` also work for symbols not in the active universe? (Lean: yes, on-demand.)
