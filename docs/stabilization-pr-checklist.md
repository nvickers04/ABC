# Stabilization PR Checklist

Use this checklist for every stabilization PR before merge.

## 1) Scope and Intent
- [ ] Stabilization-focused, not feature expansion
- [ ] Exact files/modules in scope are listed
- [ ] No unrelated file changes

## 2) Hotspot Guardrails
- [ ] Hotspot edits are justified (`core/agent.py`, `signals/scorer.py`, `signals/combiner.py`, `core/config.py`, `__main__.py`)
- [ ] No net-new behavior in restricted hotspots unless critical bugfix
- [ ] Critical hotspot bugfix includes regression test in same PR

## 3) Behavior Parity for Refactors
- [ ] Characterization/parity tests added or updated
- [ ] Refactor behavior unchanged for same inputs
- [ ] Explicitly states what is unchanged

## 4) Tests and Verification
- [ ] Relevant unit/integration tests added or updated
- [ ] Failure-path tests included where applicable (disconnects/retries/malformed output)
- [ ] Existing critical suites pass
- [ ] Manual verification steps documented

## 5) Safety and Runtime Contracts
- [ ] Tool result envelope remains valid (`success`, `data`, `error`, metadata)
- [ ] Wake/cooldown/done semantics unchanged unless intentionally targeted
- [ ] Loss/drawdown/EOD/LLM-cost safety rails preserved or explicitly tested

## 6) Rollback and Blast Radius
- [ ] Rollback steps documented
- [ ] Blast radius clearly stated
- [ ] Schema/data compatibility notes included when relevant

## 7) Checkpoint Gate
- [ ] Meets current stabilization checkpoint (runtime, wake contract, signal invariants, ops gate)
- [ ] If incomplete, next required PR is identified

## 8) Reviewer Summary
- [ ] Problem statement
- [ ] What changed
- [ ] What did not change
- [ ] Key risks
- [ ] How to verify
