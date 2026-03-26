# ABC — TODO

## Execution Autoresearch — Deep Audit Round 10 (latest)

### Files Re-audited (full data-flow trace)
- [x] **memory/__init__.py** — normalization, insert/fill, pending dicts, param review queries
- [x] **core/agent.py** — _run_execution_analysis calibration loops, rollback logic, _test_proposal
- [x] **tools/tools_executor.py** — _plan_order graduated matching, _pending_order_context bridging
- [x] **execution/ibkr_orders.py** — _place_order snapshot capture, bracket snapshot, extra_result paths
- [x] **execution/ibkr_core.py** — _on_execution fill handler, lock usage
- [x] **research/simulator.py** — calibration fallback chain, slippage application
- [x] **tests/test_autoresearch.py** — all 13 test classes (76 tests)

### Findings Fixed
1. **Missing time-bucket calibration aggregation** (MODERATE) — `_run_execution_analysis` stored `(ot, tb, ab)` and `(ot, "all", "all")` entries but never `(ot, tb, "all")`. The simulator's fallback chain expected time-specific entries that never existed — time-of-day calibration from live data was never applied. Added intermediate aggregation loop.
2. **Stale simulator fallback comment** (LOW) — comment said "Prefer broad, then time-specific" but code does the opposite (time-specific preferred). Fixed comment to match code.

### Confirmed Correct (No Change Needed)
- `abs()` slippage in before/after windows: measures deviation magnitude, correct for execution quality comparison
- Rollback threshold `after_median > before_median * 1.10`: appropriate for typical bps values
- Mann-Whitney two-sided: conservative by design, avoids premature graduation
- `_pending_order_context` atr_pct flow: bracket reads `.get()` then `insert_execution_snapshot` pops
- `extra_result.get('_atr_pct')` in `_place_order` is vestigial (always None) but harmless — `_pending_order_context` fallback works
- Threading locks (`_snapshot_lock`, `_execution_lock`): correct, no await across lock boundaries
- `_place_order_with_fill` (IOC/FOK) not tracked by autoresearch: these aren't in the decision tree, no impact
- Single-threaded agent: _pending dict race condition is theoretical only

### Test Suite: 76/76 PASSED

---

## Execution Autoresearch — Proofread Review (Round 9)

## DONE (Previous)
- [x] Inject learned rules into agent state context (auto-injected top 5 rules every cycle)
- [x] Default research pipeline ON (--no-research to opt out)
- [x] Hard-reject duplicate bracket entries (blocks if entry order already exists for symbol)
- [x] IBKR startup retry with exponential backoff (5 attempts)
- [x] Verify emergency flatten result + retry 3x (logs incomplete if positions remain)
- [x] Position concentration guard (max 25% of NLV per symbol, includes existing positions)
- [x] Grok API retry with backoff (3 attempts with 2s/4s wait)
- [x] SYSTEM_PROMPT tells agent to check trader_rules() periodically
