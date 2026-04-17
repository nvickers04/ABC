# Todo — Historical notes (all completed)

Everything below is done. This file is a post-mortem of the issues we hit
and how each one was fixed, kept for context.

## Completed work

- **SIGNAL_REGISTRY autoload** — `signals/__init__.py` now imports every
  signal module on package import, so `SIGNAL_REGISTRY` is actually
  populated at runtime (previously empty until something else imported
  the modules).
- **Phase A3 — IV rank hybrid**: `DataProvider.get_iv_info` now records a
  daily snapshot into `iv_history` and uses a trailing-percentile helper
  (`memory.compute_iv_rank_percentile`) once enough history exists.
- **Phase B — N_eff circuit breaker**: `signals.combiner` now trips to
  equal weights (`status="circuit_breaker_neff"`) after 3 consecutive
  rounds with `N_eff < 8`; streak counter persisted in `research_config`.
- **Phase C — Retire legacy DB tables**: dropped `strategies`,
  `evaluations`, `signals`, `live_signals`, `slot_environment_scores`
  from `init_db`. `_match_trade_to_signal` now joins against
  `template_recommendations` and writes `trade_feedback.template_name`.
- **Tests + pytest**: added regressions for circuit breaker, IV history
  percentile, and trade-feedback matching. Full suite: **124 passing**.
- **Failure-point audit (simple fixes)**:
  - `signals.briefing.query_briefing_data` — `n_eff` truthiness check
    coerced a legitimate `0.0` to `None`; switched to `is not None`.
  - 3× `datetime.utcnow()` deprecations replaced with
    `datetime.now(timezone.utc)` (marketdata_client, tools_options,
    tools_research).
  - 5× silent `except Exception: pass` blocks in `signals/briefing.py`
    now `logger.debug(...)` so failures are diagnosable.
- **PDT gate removed** — SEC eliminated the $25k Pattern Day Trader rule,
  so `tools_executor._check_pdt()` is now a no-op (kept as a stub for
  existing call sites).

## Live-run issue that surfaced today: "scorer is silent"

Running the agent against TWS on paper revealed the research scorer loop
appeared frozen — no `signal_scores` writes, no round-complete log, and
the LLM agent turn also hung.

### Root cause

`signals.scorer._scoring_round` is an `async def` but did a long series
of **synchronous** `DataProvider` calls inside it:

- `dp.get_quotes_bulk(universe)`
- `for sym in universe: dp.get_candles(sym, days_back=60)` (sequential!)
- `dp.get_candles("SPY")` / `dp.get_candles("QQQ")`
- Per-symbol fundamentals fan-out in `_build_symbol_data`
  (`get_fundamentals`, `get_extended_fundamentals`, `get_earnings_info`,
  `get_earnings_history`, `get_analyst_data`, `get_institutional_data`,
  `get_insider_data`, `get_peer_comparison`, `get_news`)
- Tier 2 `get_iv_info` / `get_option_chain`
- Tier 3 `get_iv_info` / `get_atr_percent`

Every one of those sync methods wraps its async MDA client via
`DataProvider._run_async`, which patches the running loop with
`nest_asyncio` and then calls `loop.run_until_complete(...)`. On
Python 3.13 this combination reliably **deadlocks the event loop**:
the scorer task stops yielding, the agent's LLM call never resumes, and
the whole process appears hung. `template_evolution` was unaffected
because it never touches `DataProvider`.

### Fix (applied to `signals/scorer.py` + `data/data_provider.py`)

- Added an `async` sibling `DataProvider.get_quotes_bulk_async` that
  awaits `_mda_client.get_quotes_bulk` directly — no `nest_asyncio`.
- Rewrote `_scoring_round` to be truly cooperative:
  - Bulk quotes: `await dp.get_quotes_bulk_async(universe)`.
  - Per-symbol candles:
    `asyncio.gather(dp.get_candles_async(sym, days_back=60) ...)`.
  - SPY/QQQ candles: `asyncio.gather(...)` pair.
  - `_get_environment`, Tier 1 per-symbol work, Tier 2 per-symbol work,
    Tier 3 recommendation build, and `_compute_forward_returns` are all
    wrapped in `asyncio.to_thread(...)` with bounded semaphores
    (Tier 1 = 8, Tier 2 = 4) so the `nest_asyncio`-laden sync paths run
    on worker threads instead of the main loop.
- Added an entry log so the loop is visibly alive on each round:
  `Round N starting: universe=M, signals=K`, plus progress lines after
  the candle fetch and each tier.

### Validation

- `pytest tests/ -q` → **124 passed**.
- Live-run: after the fix the agent immediately progresses
  (`Cycle 1 turn 1 → LLM response in 5.4s → tool call → turn 2`) while
  the scorer task is actively running in parallel.

## Follow-up bug found after the to_thread fix

A standalone rerun of `_scoring_round` still hung in Tier 2. Root cause:

`MarketDataClient` kept a **single shared** `_http_client` keyed by a
single `_loop_id`. When worker threads (spawned by `asyncio.to_thread`)
invoked `_run_async(...)`, each thread created its own event loop and
called `_get_http_client()`. Seeing a different loop id, the client
**closed the main-loop client and recreated its own** — 25 threads
racing to close/recreate the same shared `httpx.AsyncClient` while the
main loop still had candle/quote requests in flight. Classic cross-loop
contamination; the main loop's outstanding requests stalled forever.

Fix (`data/marketdata_client.py`):

- `_http_client` / `_loop_id` / `_global_semaphore` / `_options_semaphore`
  replaced with per-loop dicts keyed by `id(asyncio.get_running_loop())`:
  `_http_clients`, `_global_semaphores`, `_options_semaphores`.
- `_get_http_client` returns `None` when called with no running loop
  (instead of silently mutating state) and auto-rebuilds if the cached
  client was closed.
- `close()` iterates and `aclose()`s every per-loop client.

After this fix a standalone scoring round completes in **~126 s** on the
25-symbol universe (candles 7 s, Tier 1 82 s, Tier 2 28 s, tail 9 s).
Full pytest suite: **124 passed**.

## Secondary notes

- `get_candles_bulk` is backed by MDA's `/stocks/bulkcandles/D/` endpoint
  which returns **only one bar per symbol** and has no `days_back`
  parameter (confirmed in MDA docs). Not usable as a drop-in for the
  60-day window signals need — that's why we parallelize per-symbol
  `get_candles_async` instead.
- `time.sleep(ROUND_DELAY_SECS)` at the top of the scorer loop was left
  as-is. The comment there claims a Py3.13 asyncio bug; if a future
  restart shows rounds not being paced correctly, swap to
  `await asyncio.sleep(ROUND_DELAY_SECS)`.
