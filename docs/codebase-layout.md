# Codebase layout

Where logic lives and what is still migrating. **Entry commands:** [entry-points.md](entry-points.md).

---

## Processes

| Process | Module | Notes |
|---------|--------|--------|
| Research host | `research/host.py` via `python -m research` | Scoring + template evolution; no Grok |
| Trader | `__main__.py` → `core/agent.py` | Grok ReAct + IBKR; CLI flags in `core/entry_cli.py` |
| Heartbeat | `core/runtime/heartbeat.py` | DB key `research_host_heartbeat_ts` (reads legacy `daemon_heartbeat_ts` too) |

---

## Canonical imports (prefer these)

| Concern | Import from |
|---------|-------------|
| DB / trades / config | `memory` (public API); implementation split into `memory.repos.*` |
| Research config keys | `memory` or `memory.repos.config_repo` |
| JSON repair helpers | `core.json_parse` |
| Research cache TTL / topics | `core.research_topics` |
| Daily review / risk ramp | `core.runtime.review` |
| Execution analysis / Mann-Whitney proposals | `core.runtime.execution_analysis` |
| Agent state text | `core.runtime.state_context.StateContextBuilder` |
| Tool results | `tools.tool_contract` |
| Liveness | `core.runtime.heartbeat.is_research_host_alive` |
| Quality policy | `core.quality.quality_matrix` |

---

## Intentional agent methods (not duplicates)

`TradingAgent` still exposes some methods that delegate elsewhere so tools and tests stay stable:

- `_build_state_context` → `StateContextBuilder` (wired as `tools._state_builder`)
- `_run_daily_review` / `_evaluate_risk_ramp` → `core.runtime.review`
- `_run_execution_analysis` / `_test_proposal` → `core.runtime.execution_analysis`

---

## Memory package (in progress)

| Module | Status |
|--------|--------|
| `memory/repos/config_repo.py` | Canonical for `research_config`, graduated params |
| `memory/repos/feedback_repo.py` | Canonical for hypotheses |
| `memory/repos/execution_repo.py` | Trades, execution snapshots, IV, slippage writers + reads |
| `memory/repos/schema.py` | Canonical for `get_db` / `init_db` / migrations |
| `memory/session_state.py` | Process-local pending order context + calibration version |
| `memory/working_memory.py` | Canonical WM |
| `memory/repos/provenance_repo.py` | QualityMatrix tool/decision provenance rows |

`from memory import …` remains the supported public API until migration finishes.

---

## Migration backlog (do not half-do)

1. ~~Move `get_db` / `init_db` into `memory/repos/schema.py`~~ (done).
2. ~~Move execution writers into `execution_repo`~~ (done); `memory/__init__.py` is thin shims.
3. ~~Extract execution analysis from `core/agent.py`~~ → `core/runtime/execution_analysis.py` (done).
4. ~~Move QualityMatrix provenance writers~~ → `memory/repos/provenance_repo.py` (done).
5. ~~Rename heartbeat DB key~~ → `research_host_heartbeat_ts` with legacy read (done).
6. ~~`research/daemon.py` → `research/host.py`~~; trader CLI `--require-research-host` (legacy `--require-daemon`) (done).

---

## Tests

Characterization tests with `PR##` in the module docstring are **stabilization locks**, not duplicate suites. Prefer extending them when changing behavior under a hotspot.
