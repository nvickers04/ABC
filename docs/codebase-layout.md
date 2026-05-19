# Codebase layout

Where logic lives and what is still migrating. **Entry commands:** [entry-points.md](entry-points.md).

---

## Processes

| Process | Module | Notes |
|---------|--------|--------|
| Research host | `research/daemon.py` via `python -m research` | Scoring + template evolution; no Grok |
| Trader | `__main__.py` → `core/agent.py` | Grok ReAct + IBKR; use `__main__.py` only (not `python core/agent.py`) |
| Heartbeat | `core/runtime/heartbeat.py` | DB key `daemon_heartbeat_ts` (historical name) |

---

## Canonical imports (prefer these)

| Concern | Import from |
|---------|-------------|
| DB / trades / config | `memory` (public API); implementation split into `memory.repos.*` |
| Research config keys | `memory` or `memory.repos.config_repo` |
| JSON repair helpers | `core.json_parse` |
| Research cache TTL / topics | `core.research_topics` |
| Daily review / risk ramp | `core.runtime.review` |
| Agent state text | `core.runtime.state_context.StateContextBuilder` |
| Tool results | `tools.tool_contract` |
| Liveness | `core.runtime.heartbeat.is_research_host_alive` |
| Quality policy | `core.quality.quality_matrix` |

---

## Intentional agent methods (not duplicates)

`TradingAgent` still exposes some methods that delegate elsewhere so tools and tests stay stable:

- `_build_state_context` → `StateContextBuilder` (wired as `tools._state_builder`)
- `_run_daily_review` / `_evaluate_risk_ramp` → `core.runtime.review`
- `_run_execution_analysis` / `_test_proposal` — **still implemented on the agent** (future extract)

---

## Memory package (in progress)

| Module | Status |
|--------|--------|
| `memory/repos/config_repo.py` | Canonical for `research_config`, graduated params |
| `memory/repos/feedback_repo.py` | Canonical for hypotheses |
| `memory/repos/execution_repo.py` | Read-side SQL here; writers still in `memory/__init__.py` |
| `memory/repos/schema.py` | Scaffold — `get_db` / `init_db` still in `memory/__init__.py` |
| `memory/working_memory.py` | Canonical WM |

`from memory import …` remains the supported public API until migration finishes.

---

## Migration backlog (do not half-do)

1. Move `get_db` / `init_db` into `memory/repos/schema.py`; slim `memory/__init__.py`.
2. Move execution writers out of `memory/__init__.py` into `execution_repo`.
3. Extract `_run_execution_analysis` and `_test_proposal` from `core/agent.py` into `core/runtime/review` (or sibling).
4. Optional: DB migration alias for `daemon_heartbeat_ts` → clearer name (low priority).

---

## Tests

Characterization tests with `PR##` in the module docstring are **stabilization locks**, not duplicate suites. Prefer extending them when changing behavior under a hotspot.
