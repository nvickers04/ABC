# Engineering guide

PR quality gate and blast-radius map for reviews and AI assistants.

**See also:** [plain-english-glossary.md](plain-english-glossary.md) · [entry-points.md](entry-points.md) · [codebase-layout.md](codebase-layout.md) · [stabilization-pr-checklist.md](stabilization-pr-checklist.md) · [change-impact-map.md](change-impact-map.md)

---

## Documentation and file moves

When moving or consolidating docs, scripts, or entry points:

- **Move and update** — merge or relocate content, delete the old path, grep and fix all references (README, ops docs, Docker, scripts, rules) in the same change.
- **No redirect stubs** — do not leave files that only say “moved to …” or link to the new location.
- **No extra root shims** — prefer `python -m <package>` or `scripts/` launchers; implementation lives in the package (e.g. `research/host.py`).

Cursor enforces this in `.cursor/rules/docs-and-file-moves.mdc`.  
Canonical process commands: [entry-points.md](entry-points.md).  
Module map and migration status: [codebase-layout.md](codebase-layout.md).

### Python environment

- Use **one** venv at repo root: `.venv` (`python -m venv .venv`, then `pip install -r requirements.txt`).
- **Do not** commit `.venv/`, `.venv-1/`, or other IDE-created copies (all ignored via `.gitignore`).
- `scripts/run_research.ps1` uses `.venv\Scripts\python.exe` when it exists.

**Removed paths (do not recreate):** `research_daemon.py`, `research/daemon.py` (use `research/host.py`), `docs/data-sources/*`,
`docs/PLAN_*.md`, `docs/*_HOST_SETUP.md`, split engineering checklists,
`scripts/check_researcher.py`, `scripts/check_trader.py`, `scripts/smoke_*_tools.py`
(plural legacy names — use `health.py` and `smoke_tools.py`).

---

## Stabilization PR checklist

Use **[stabilization-pr-checklist.md](stabilization-pr-checklist.md)** before every stabilization PR merge (single canonical copy).

---

## Change impact map

Before changing code, check **[change-impact-map.md](change-impact-map.md)** for adjacent subsystems and hotspot files.

---

## Glossary

Domain and engineering terms live in **[plain-english-glossary.md](plain-english-glossary.md)** (research host, **QualityMatrix**, **WM routing**, **master ProfitConfig**, `--simulate`, heartbeat, Independent Mode, MDA, token cap, safety rails, etc.). Link there in PRs instead of redefining terms.

Profitability / simulation changes: also read **[simulation-and-optimization.md](simulation-and-optimization.md)** and run `tests/test_simulation.py`, `tests/test_optimizer.py`, `tests/test_central_profit_config.py` as appropriate.

If a term is unclear, add or update the glossary entry before implementing.
