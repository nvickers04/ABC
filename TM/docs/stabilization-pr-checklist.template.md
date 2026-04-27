# Stabilization PR Checklist (Template)

Use this checklist for every stabilization PR before merge.

## 1) Scope and Intent
- [ ] Stabilization-focused, not feature expansion
- [ ] Exact files/modules in scope are listed
- [ ] No unrelated file changes

## 2) Hotspot Guardrails
- [ ] Hotspot edits are justified (`<hotspot-file-1>`, `<hotspot-file-2>`, `<hotspot-file-3>`)
- [ ] No net-new behavior in restricted hotspots unless critical bugfix
- [ ] Critical hotspot bugfix includes regression test in same PR

## 3) Behavior Parity for Refactors
- [ ] Characterization/parity tests added or updated
- [ ] Refactor behavior unchanged for same inputs
- [ ] Explicitly states what is unchanged

## 4) Tests and Verification
- [ ] Relevant unit/integration tests added or updated
- [ ] Failure-path tests included where applicable
- [ ] Existing critical suites pass (`<critical-test-suite-1>`, `<critical-test-suite-2>`)
- [ ] Manual verification steps documented

## 5) Safety and Runtime Contracts
- [ ] Core result envelope/contract remains valid
- [ ] Wake/cooldown/scheduler semantics unchanged unless intentionally targeted
- [ ] Safety rails preserved or explicitly tested

## 6) Rollback and Blast Radius
- [ ] Rollback steps documented
- [ ] Blast radius clearly stated
- [ ] Schema/data compatibility notes included when relevant

## 7) Checkpoint Gate
- [ ] Meets current stabilization checkpoint
- [ ] If incomplete, next required PR is identified

## 8) Reviewer Summary
- [ ] Problem statement
- [ ] What changed
- [ ] What did not change
- [ ] Key risks
- [ ] How to verify
