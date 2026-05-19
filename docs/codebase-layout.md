# Codebase layout

Where logic lives and what is still migrating.

**See also:** [entry-points.md](entry-points.md) · [plain-english-glossary.md](plain-english-glossary.md) · [operations/independent-mode.md](operations/independent-mode.md)

---

## Processes

| Process | Module | Notes |
|---------|--------|--------|
| Research host | `research/host.py` via `python -m research` | Scoring + template evolution; no Grok |
| Trader | `__main__.py` → `core/agent.py` | Grok ReAct + IBKR; CLI in `core/entry_cli.py`; structlog via `core/log_context.py` |
| Heartbeat | `core/runtime/heartbeat.py` | DB key `research_host_heartbeat_ts` (reads legacy `daemon_heartbeat_ts`) |

---

## Quality and runtime (trader)

| Concern | Module |
|---------|--------|
| QualityMatrix policy | `core/quality/quality_matrix.py` |
| Operating mode + heartbeat sync | `core/runtime/operating_context.py` |
| WM routing (Postgres vs local) | `core/runtime/working_memory_access.py` |
| Local WM fallback store | `core/runtime/local_memory_fallback.py` |
| Cycle prompt assembly | `core/runtime/cycle_context.py` |
| Safety rails | `core/runtime/safety.py` |

Glossary: [QualityMatrix](plain-english-glossary.md#quality-context-and-working-memory), [WM routing](plain-english-glossary.md#quality-context-and-working-memory).

---

## Profitability configuration

| Concern | Module |
|---------|--------|
| Master ProfitConfig singleton + `simulate_backtest` | `core/central_profit_config.py` |
| Built-in / evolved profiles | `core/profit_profiles.py`, `data/evolved_profiles.json` |
| Profile compose cache (optimizer) | `core/profit_profile_cache.py` |
| Thread-local composed config (`--parallel`) | `core/profit_config_context.py` |
| Sub-configs (centralized levers) | `core/risk_execution_config.py`, `loop_config.py`, `memory_config.py`, `prompt_config.py`, `tool_registry.py` |
| Grid / genetic scoring | `core/profile_optimization.py`, `core/optimizer_backtest.py` |
| Historical sim runner | `core/simulation/runner.py`, `backtest_llm.py`, `replay_data.py` |
| Cycle logs + dashboard aggregation | `core/profit_cycle_logger.py`, `core/profit_summary.py` |
| CLI scripts | `scripts/optimize_profiles.py`, `scripts/dashboard.py` |

Guide: [simulation-and-optimization.md](simulation-and-optimization.md). Glossary: [Profitability configuration](plain-english-glossary.md#profitability-configuration-and-simulation).

---

## Canonical imports (prefer these)

| Concern | Import from |
|---------|-------------|
| DB / trades / config | `memory` (public API); implementation in `memory.repos.*` |
| Research config keys | `memory` or `memory.repos.config_repo` |
| JSON repair helpers | `core.json_parse` |
| Research cache TTL / topics | `core.research_topics` |
| Daily review / risk ramp | `core.runtime.review` |
| Execution analysis | `core.runtime.execution_analysis` |
| Agent state text | `core.runtime.state_context.StateContextBuilder` |
| Tool results | `tools.tool_contract` |
| Liveness | `core.runtime.heartbeat.is_research_host_alive` |
| Quality policy | `core.quality.quality_matrix` |
| ProfitConfig / backtest | `core.central_profit_config.get_profit_config`, `simulate_backtest` |
| Active sub-config levers | `get_risk_execution_config`, `get_loop_config`, `get_memory_config`, `get_prompt_config`, `get_tool_registry` |

---

## Intentional agent methods (not duplicates)

`TradingAgent` still exposes some methods that delegate elsewhere so tools and tests stay stable:

- `_build_state_context` → `StateContextBuilder` (wired as `tools._state_builder`)
- `_run_daily_review` / `_evaluate_risk_ramp` → `core.runtime.review`
- `_run_execution_analysis` / `_test_proposal` → `core.runtime.execution_analysis`

---

## Memory package

| Module | Status |
|--------|--------|
| `memory/repos/config_repo.py` | Canonical for `research_config`, graduated params |
| `memory/repos/feedback_repo.py` | Canonical for hypotheses |
| `memory/repos/execution_repo.py` | Trades, execution snapshots, IV, slippage |
| `memory/repos/schema.py` | Canonical for `get_db` / `init_db` / migrations |
| `memory/session_state.py` | Process-local pending order context |
| `memory/working_memory.py` | Canonical Postgres WM |
| `memory/repos/provenance_repo.py` | QualityMatrix tool/decision provenance rows |

`from memory import …` remains the supported public API.

---

## Migration backlog (do not half-do)

1. ~~Move `get_db` / `init_db` into `memory/repos/schema.py`~~ (done).
2. ~~Move execution writers into `execution_repo`~~ (done).
3. ~~Extract execution analysis from `core/agent.py`~~ (done).
4. ~~Move QualityMatrix provenance writers~~ (done).
5. ~~Rename heartbeat DB key~~ (done).
6. ~~`research/daemon.py` → `research/host.py`~~ (done).

---

## Tests

Characterization tests with `PR##` in the module docstring are **stabilization locks**, not duplicate suites. Prefer extending them when changing behavior under a hotspot.
