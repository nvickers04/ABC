# Stabilization PR checklist

Canonical copy for pre-merge quality gates. The engineering guide embeds this
list for convenience; **edit this file** when the checklist changes.

**See also:** [engineering.md](engineering.md) · [plain-english-glossary.md](plain-english-glossary.md) · [codebase-layout.md](codebase-layout.md)

---

## Scope and intent

- [ ] Stabilization-focused, not feature expansion
- [ ] Exact files/modules in scope are listed
- [ ] No unrelated file changes

## Hotspot guardrails

- [ ] Hotspot edits justified (`core/agent.py`, `signals/scorer.py`, `signals/combiner.py`, `core/config.py`, `__main__.py`)
- [ ] No net-new behavior in hotspots unless critical bugfix
- [ ] Critical hotspot bugfix includes regression test in same PR

## Behavior parity for refactors

- [ ] Characterization/parity tests added or updated
- [ ] Refactor behavior unchanged for same inputs
- [ ] Explicitly states what is unchanged

## Tests and verification

- [ ] Relevant unit/integration tests added or updated
- [ ] Failure-path tests where applicable
- [ ] Existing critical suites pass
- [ ] Manual verification steps documented

## Safety and runtime contracts

- [ ] Tool result envelope valid (`success`, `data`, `error`, metadata)
- [ ] Wake/cooldown/done semantics unchanged unless targeted
- [ ] Loss/drawdown/EOD/LLM-cost safety rails preserved or tested

## Rollback and blast radius

- [ ] Rollback steps documented
- [ ] Blast radius stated
- [ ] Schema/data compatibility notes when relevant

## Independent Mode / working memory

- [ ] Follow [operations/independent-mode.md](operations/independent-mode.md) (single writer per mode)
- [ ] No dual-write Postgres WM + local JSON without recovery design

## Reviewer summary

- [ ] Problem statement · what changed · what did not · risks · how to verify
