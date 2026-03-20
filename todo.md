# ABC — TODO

## Completed
- [x] Swap model to Grok 4.20 0309 Reasoning
- [x] Migrate from OpenAI SDK to xAI SDK
- [x] Phase 0: Agent context restructuring (briefing tool, research cache, aggressive_nudge removal)
- [x] Phase 1: Token efficiency (indent removal, envelope flattening, compact candles/options)
- [x] Phase 2: MDA API credit optimization (mode=cached, server-side filters, bulk quotes, columns)
- [x] Phase 3: MDA endpoints (earnings, news, market_status), Black-Scholes simulator, sandbox env dict
- [x] Phase 4: yfinance→MDA migration (earnings_info, news, peer_comparison → MDA; 5 yfinance methods kept)

## 1. Dead Strategy Live Scan Gate
- [ ] Gate `_run_live_scan` on `best_fitness > MIN_LIVE_SCAN_FITNESS` (research/agent.py ~L1687)
- [ ] Keep zero/negative-fitness slots visible in briefing (LLM sees what's NOT working)
- [ ] Only suppress live signal generation — don't hide the slot data

## 2. Multi-Timeframe Chart Tools
Add multiple chart tools with preset timeframe combos + token cost scores so LLM can reason about depth vs cost.
- [ ] `chart_intraday(symbol)` — 1min 30 bars + 5min 30 bars + 15min 20 bars. Token score: medium. Use for active day-trade entries.
- [ ] `chart_swing(symbol)` — hourly 20 bars + daily 30 bars + weekly 12 bars. Token score: medium. Use for swing/multi-day setups.
- [ ] `chart_full(symbol)` — 5min 30 + hourly 20 + daily 30. Token score: high. Use when validating a strong thesis across timeframes.
- [ ] `chart_quick(symbol)` — daily 10 bars only + derived analytics (ATR, trend label, rel volume). Token score: low. Use for screening.
- [ ] Each tool fetches timeframes concurrently, returns compact columnar format + derived analytics per frame
- [ ] Register all in HANDLERS + update SYSTEM_PROMPT with token scores and when-to-use guidance
- [ ] Instruct LLM to reason about token budget: scale up research depth when thesis looks valid, stay cheap when screening

## 3. Options Lookup Endpoint
- [ ] Add `get_option_lookup(symbol)` to marketdata_client.py using MDA `/options/lookup/{symbol}/` for OCC symbol generation
- [ ] Wire into options tools where OCC symbols are constructed manually

## 4. Market Status → Early Close Detection
- [ ] Wire `get_market_status()` (already in MDA client) into market_hours.py
- [ ] Detect half-days / early closes from MDA and adjust session boundaries
- [ ] Surface early close info in `_build_state_context()` warnings

## 5. Trader Self-Improvement Loop
Build a learning system that mirrors research's evolution architecture for trader behavior.
### Schema
- [ ] Add `trader_learned_rules` table to memory/__init__.py (id, ts, rule_text, env_key, signed_fitness, kept, actual_pnl, simulated_pnl, parent_id)
- [ ] Add `trader_evaluations` table (rule_id, eval_date, trades_tested, hit_rate, pnl, execution_gap)
- [ ] Wire `trader_hypotheses` reads — currently write-only journal, nothing consumes it
### Daily Review Loop
- [ ] Add `_run_trader_review()` in core/agent.py — runs at market close
- [ ] Aggregate day's trades/signals, compute trader_fitness (P&L-based, regime-adjusted, signed_edge_score)
- [ ] LLM review: generate new rules, keep/discard based on actual vs simulated outcomes
- [ ] Store rules with env_key, signed_fitness, kept flag
### Counterfactual Replay
- [ ] Extend replay harness for trader rule evaluation — "what would this rule have done on past days?"
- [ ] Store both simulated and actual metrics (like research search_fitness vs promotion_fitness)
### Execution Gap Learning
- [ ] Aggregate `trade_feedback.execution_gap` into per-slot/symbol cost model
- [ ] Inject into briefing: "Slot 3 costs 1.2% more on CRWD than simulation"
- [ ] Feed back into strategy proposals: "Factor this execution cost into stops"
### Tools + Briefing
- [ ] Add `get_trader_rules(env_key)` tool — query learned rules for current regime
- [ ] Update briefing to surface top trader rules + hypothesis status
- [ ] Compact format: rule text + fitness + env match + age

## 6. Miscellaneous
- [ ] Tune RISK_PER_TRADE for live account (start at 0.5%, increase after validation)
- [ ] Monitor beta model slug changes — slugs are not stable
- [ ] Add multi-agent research phase (pre-scan with web_search/x_search before ReAct loop)
- [ ] Clean unused code: CREDIT_STRUCTURES in promoter.py, unused strategy_source param in replay.py