# Engineering guide

PR quality gate, blast-radius map, and plain-English glossary for reviews and AI assistants.

---

## Documentation and file moves

When moving or consolidating docs, scripts, or entry points:

- **Move and update** — merge or relocate content, delete the old path, grep and fix all references (README, ops docs, Docker, scripts, rules) in the same change.
- **No redirect stubs** — do not leave files that only say “moved to …” or link to the new location.
- **No extra root shims** — prefer `python -m <package>` or `scripts/` launchers; implementation lives in the package (e.g. `research/daemon.py`).

Cursor enforces this in `.cursor/rules/docs-and-file-moves.mdc`.  
Canonical process commands: [entry-points.md](entry-points.md).

**Removed paths (do not recreate):** `research_daemon.py`, `docs/data-sources/*`,
`docs/PLAN_*.md`, `docs/*_HOST_SETUP.md`, split engineering checklists,
`scripts/check_researcher.py`, `scripts/check_trader.py`, `scripts/smoke_*_tools.py`
(plural legacy names — use `health.py` and `smoke_tools.py`).

---

## Stabilization PR checklist

Use before every stabilization PR merge.

### Scope and intent
- [ ] Stabilization-focused, not feature expansion
- [ ] Exact files/modules in scope are listed
- [ ] No unrelated file changes

### Hotspot guardrails
- [ ] Hotspot edits justified (`core/agent.py`, `signals/scorer.py`, `signals/combiner.py`, `core/config.py`, `__main__.py`)
- [ ] No net-new behavior in hotspots unless critical bugfix
- [ ] Critical hotspot bugfix includes regression test in same PR

### Behavior parity for refactors
- [ ] Characterization/parity tests added or updated
- [ ] Refactor behavior unchanged for same inputs
- [ ] Explicitly states what is unchanged

### Tests and verification
- [ ] Relevant unit/integration tests added or updated
- [ ] Failure-path tests where applicable
- [ ] Existing critical suites pass
- [ ] Manual verification steps documented

### Safety and runtime contracts
- [ ] Tool result envelope valid (`success`, `data`, `error`, metadata)
- [ ] Wake/cooldown/done semantics unchanged unless targeted
- [ ] Loss/drawdown/EOD/LLM-cost safety rails preserved or tested

### Rollback and blast radius
- [ ] Rollback steps documented
- [ ] Blast radius stated
- [ ] Schema/data compatibility notes when relevant

### Independent Mode / working memory
- [ ] Follow [operations/independent-mode.md](operations/independent-mode.md) (single writer per mode)
- [ ] No dual-write Postgres WM + local JSON without recovery design

### Reviewer summary
- [ ] Problem statement · what changed · what did not · risks · how to verify

---

## Change impact map

Before changing code, check adjacent subsystems.

| Subsystem | Primary files | Usually impacts | Must verify |
|-----------|---------------|---------------|-------------|
| Runtime orchestration | `__main__.py`, `core/agent.py`, `core/wake_events.py` | Cycle timing, startup/shutdown | done/cooldown/wake, no stalls |
| Decision & safety | `core/agent.py`, `core/config.py` | Risk thresholds, prompts | loss/drawdown/EOD/cost caps |
| Tooling | `tools/tools_executor.py`, `tools/tools_*.py` | Tool format, agent compatibility | result contract, malformed output |
| Signals & research | `signals/scorer.py`, `signals/combiner.py`, `research/*` | Forward returns, IC, briefing | `test_signals`, `test_forward_returns`, `test_ic_honesty` |
| Persistence | `memory/__init__.py`, WM paths | Migrations, data quality | schema version; [independent-mode.md](operations/independent-mode.md) |
| Broker & data | `execution/*`, `data/*` | Orders, quotes, retries | disconnect path, missing data |

**Hotspots (extra caution):** `core/agent.py`, `signals/scorer.py`, `signals/combiner.py`, `core/config.py`, `__main__.py`.

**Before every non-trivial change:** (1) which subsystem? (2) what can break? (3) which tests prove no regression? (4) rollback in paper?

---

## Glossary

| Term | Meaning |
|------|---------|
| **Boundary extraction** | Splitting a large module into smaller pieces without changing behavior |
| **State builder** | Gathers facts for the agent (session, account, positions, risk) |
| **Refactor** | Restructure code; behavior unchanged |
| **Contract** | Expected shape between parts (e.g. tool results always have `success`, `data`, `error`) |
| **Invariant** | Rule that must stay true (e.g. safety rails always fire at threshold) |
| **Regression** | Something that worked before and broke after a change |
| **Blast radius** | How far a change can affect the rest of the system |
| **Parity / characterization test** | Captures current behavior so refactors stay safe |
| **Hotspot file** | High-churn, high-risk file |
| **Cycle loop** | Read state → reason → tools → wait/trade |
| **Cooldown / wake** | Wait between cycles unless an event wakes the loop |
| **Safety rails** | Daily loss, drawdown, EOD flatten, LLM budget cap |
| **Independent Mode** | Trader without fresh researcher feed; conservative risk; may use local JSON WM |
| **QualityMatrix** | In-process policy: risk scale, tool blocks, LLM caps; host enforces, tools only expose state |
| **Working memory authority** | Which store owns theses (Postgres vs local file); one writer per mode |

If a term is unclear, explain it in plain English before implementing.
