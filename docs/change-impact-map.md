# Change impact map

Which subsystems break when you touch which files. Use before non-trivial changes.

**See also:** [engineering.md](engineering.md) · [codebase-layout.md](codebase-layout.md) · [plain-english-glossary.md](plain-english-glossary.md)

---

| Subsystem | Primary files | Usually impacts | Must verify |
|-----------|---------------|-----------------|-------------|
| Runtime orchestration | `__main__.py`, `core/agent.py`, `core/wake_events.py` | Cycle timing, startup/shutdown | done/cooldown/wake, no stalls |
| Decision & safety | `core/agent.py`, `core/config.py` | Risk thresholds, prompts | loss/drawdown/EOD/cost caps |
| Tooling | `tools/tools_executor.py`, `tools/tools_*.py` | Tool format, agent compatibility | result contract, malformed output |
| Signals & research | `signals/scorer.py`, `signals/combiner.py`, `research/*` | Forward returns, IC, briefing | `test_signals`, `test_forward_returns`, `test_ic_honesty` |
| Persistence | `memory/__init__.py`, WM paths | Migrations, data quality | schema version; [independent-mode.md](operations/independent-mode.md) |
| Quality / WM routing | `core/quality/*`, `core/runtime/operating_context.py`, `working_memory_access.py` | Independent Mode, tool blocks | heartbeat + WM authority tests |
| Broker & data | `execution/*`, `data/*` | Orders, quotes, retries | disconnect path, missing data |
| Master ProfitConfig | `core/central_profit_config.py`, `core/profit_profiles.py`, five `*_config.py` | Trader, sim, optimizer, cycle logs, `core.config` exports | `test_central_profit_config`, `test_profit_config_context`, `test_simulation`, `test_optimizer` |
| Simulation / optimizer | `core/simulation/*`, `scripts/optimize_profiles.py`, `core/profile_optimization.py` | Backtest metrics, composite ranking, evolved profiles | `test_simulation`, `test_optimizer`; re-run short `--simulate` after profile changes |
| Cycle logs / dashboard | `core/profit_cycle_logger.py`, `scripts/dashboard.py` | Live P&L attribution, profile suggestions | `test_logger`, `test_live_profile_optimize` |

**Hotspots (extra caution):** `core/agent.py`, `signals/scorer.py`, `signals/combiner.py`, `core/config.py`, `__main__.py`, `core/central_profit_config.py`.

**Before every non-trivial change:** (1) which subsystem? (2) what can break? (3) which tests prove no regression? (4) rollback in paper?
