# Change Impact Map (Cohesion Guide)

Before changing code, use this map to check what else might be affected.

## 1) Runtime Orchestration
- Primary files: `__main__.py`, `core/agent.py`, `core/wake_events.py`
- Usually impacts:
  - Cycle timing/wake behavior
  - Startup and shutdown behavior
  - Logging and operational visibility
- Must verify:
  - done/cooldown/wake behavior
  - no unexpected stalls

## 2) Decision and Safety Logic
- Primary files: `core/agent.py`, `core/config.py`
- Usually impacts:
  - Risk thresholds and guardrails
  - Prompt-driven behavior constraints
- Must verify:
  - daily loss, drawdown, EOD flatten, cost cap paths
  - no prompt policy drift unless intended

## 3) Tooling Layer
- Primary files: `tools/tools_executor.py`, `tools/tools_*.py`
- Usually impacts:
  - Tool call format and error handling
  - Agent compatibility with tool results
- Must verify:
  - tool result contract (`success`, `data`, `error`, metadata)
  - malformed output handling

## 4) Signal and Research Pipeline
- Primary files: `signals/scorer.py`, `signals/combiner.py`, `signals/briefing.py`, `research/*`
- Usually impacts:
  - Forward-return math
  - IC calculations and gating
  - Recommendations and edge scoring
- Must verify:
  - `tests/test_signals.py`
  - `tests/test_forward_returns.py`
  - `tests/test_ic_honesty.py`

## 5) Persistence and Schema
- Primary files: `memory/__init__.py`
- Usually impacts:
  - DB compatibility
  - startup migrations
  - data quality assumptions
- Must verify:
  - schema version/migration path
  - no silent compatibility break

## 6) Broker and Data Integrations
- Primary files: `execution/*`, `data/*`
- Usually impacts:
  - order lifecycle
  - quote/market data reliability
  - failure and retry behavior
- Must verify:
  - broker disconnect path
  - partial/missing data handling

---

## Required Cross-Impact Questions (Ask Before Every Non-Trivial Change)
1. Which subsystem is being changed?
2. Which adjacent subsystem can this break?
3. Which tests prove this did not regress?
4. What is the rollback plan if behavior changes in paper mode?

## Hotspot Reminder
Treat these as high-risk and require extra caution:
- `core/agent.py`
- `signals/scorer.py`
- `signals/combiner.py`
- `core/config.py`
- `__main__.py`
