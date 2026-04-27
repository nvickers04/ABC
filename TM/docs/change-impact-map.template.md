# Change Impact Map (Template)

Before changing code, check what else might be affected.

## 1) Runtime Orchestration
- Primary files: `<runtime-entrypoint>`, `<orchestrator-file>`, `<event-bus-file>`
- Usually impacts:
  - loop timing and wake behavior
  - startup/shutdown behavior
  - operational visibility
- Must verify:
  - done/cooldown/wake behavior
  - no stalls/deadlocks

## 2) Decision and Safety Logic
- Primary files: `<decision-file>`, `<config-file>`
- Usually impacts:
  - guardrails and thresholds
  - policy constraints
- Must verify:
  - risk/budget/timeout/safety paths
  - no policy drift unless intended

## 3) Tool/Adapter Layer
- Primary files: `<tool-executor-file>`, `<tool-module-pattern>`
- Usually impacts:
  - request/response contract
  - error handling and retries
- Must verify:
  - normalized result envelope
  - malformed output handling

## 4) Domain/Scoring Pipeline
- Primary files: `<domain-core-1>`, `<domain-core-2>`
- Usually impacts:
  - scoring/gating math
  - ranking/recommendation behavior
- Must verify:
  - `<critical-test-suite-1>`
  - `<critical-test-suite-2>`

## 5) Persistence and Schema
- Primary files: `<storage-layer-file>`
- Usually impacts:
  - compatibility/migrations
  - startup readiness
  - data quality assumptions
- Must verify:
  - schema version path
  - no silent compatibility break

## 6) External Integrations
- Primary files: `<integration-file-1>`, `<integration-file-2>`
- Usually impacts:
  - API reliability
  - partial/missing data behavior
- Must verify:
  - disconnect/retry paths
  - degraded-mode handling

---

## Required Cross-Impact Questions
1. What subsystem is changing?
2. Which adjacent subsystem could break?
3. Which tests prove no regression?
4. What is rollback plan if behavior changes?

## Hotspot Reminder
Treat these as high-risk and require extra caution:
- `<hotspot-file-1>`
- `<hotspot-file-2>`
- `<hotspot-file-3>`
