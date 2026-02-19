# ABC-Application Trading System

## Status

- **Architecture:** LLM-Autonomous Trading (ReAct Agent → Tools → IBKR)
- **Brain:** Grok LLM decides everything via tool calls
- **Layers:** L1 (IBKR Execution) → L2 (Data/Instruments) → L3 (Agent/Tools)
- **Constraints:** Cash-only, defined-risk instruments only (no margin, no naked options)
- **Mode:** Paper trading validated on port 7497

## Give Grok Better Information

### Enhanced Tools (Let LLM See More)
- [ ] **get_execution_quality tool** - Show actual slippage per trade so Grok can adjust
- [ ] **get_market_regime tool** - VIX level, breadth, correlation data for context
- [ ] **get_time_of_day_stats tool** - Historical win rates by hour from outcome_tracker
- [ ] **get_position_heat tool** - How many positions moving against, correlation between them

### Outcome Tracker Enhancements (Better Feedback Loop)
- [x] **Track slippage** - Record (fill_price - signal_price) / signal_price ✅
- [ ] **Track volatility regime** - Tag each trade with VIX level at entry (PARTIAL: VIX displayed in context but not tagged on trade records)
- [ ] **Memory decay** - Keep last 1000 detailed, summarize older to alpha/beta

## Guardrails & Observability

### Observability (Debug & Learn)
- [ ] **LangFuse tracing** - Add @observe decorators to track LLM decisions
- [ ] **Execution latency** - Time from tool call to IBKR fill
- [x] **Decision logging** - Store full LLM reasoning for each action ✅ (log_thought → events.jsonl)
- [ ] **Alert on anomalies** - Unusual fill prices, high rejection rate, API errors

## LOWER PRIORITY -
### Persistence (Scale Beyond JSON)
- [ ] **File locking** - Prevent corruption from concurrent writes
- [ ] **SQLite migration** - Replace JSON with abc.db for positions, outcomes
- [ ] **Redis for state** - Use existing redis/ for live position updates

### Context Management
- [ ] **Context effectiveness** - Track decision quality vs context size

---

## 💡 Philosophy Notes

**LLM-First Design Principles:**
1. Don't hardcode rules Grok can learn from data
2. Add tools to expose information, not constraints to hide it  
3. Hard guardrails only for things that could blow up the account
4. Let outcome tracking teach the LLM what works
5. Observability over automation - watch Grok, don't micromanage


## 🎯 Success Criteria for Live Trading

1. [ ] 100+ paper trades with positive expectancy
2. [ ] All stops verified on every startup
3. [x] Circuit breakers tested (trigger and recover) ✅
4. [ ] Drawdown <5% over 2 weeks of paper trading
5. [ ] Execution latency <2 seconds signal-to-fill
6. [ ] Win rate calibration within 10% of predicted

### 📝 Documentation Needed

- [ ] **Document screener sources** - Explain FMP→Finviz→yfinance cascade and when each is used (PARTIAL: inline docs in screener_registry.py, but no dedicated page)
- [ ] **Document cost tracking** - Explain the 50% profit allocation concept and how budget enforcement works (or doesn't)

### 💭 Questions to Resolve

- [ ] **Parameter validation** - Should we validate LLM screening requests against available parameters?

