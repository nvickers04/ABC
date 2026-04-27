# TM Starter Pack

Reusable AI-collaboration docs and Cursor rules for new projects.

## What This Is

This folder contains project-agnostic templates you can copy into a new repo to:
- keep AI changes cohesive
- reduce risky broad edits
- require cross-impact checks
- keep explanations in plain English

## How To Use In A New Project

1. Copy contents into the target repo:
   - `TM/docs/*` -> `docs/*`
   - `TM/.cursor/rules/*` -> `.cursor/rules/*`
2. Replace placeholders such as:
   - `<runtime-entrypoint>`
   - `<orchestrator-file>`
   - `<hotspot-file-1>`
   - `<critical-test-suite-1>`
3. Keep `alwaysApply: true` in rules if you want enforcement in every chat.
4. Do one short pass after first week of work to tune hotspots and verification steps.

## Included Templates

- `docs/plain-english-glossary.template.md`
- `docs/stabilization-pr-checklist.template.md`
- `docs/change-impact-map.template.md`
- `.cursor/rules/stabilization-pr-checklist.template.mdc`
- `.cursor/rules/cohesion-and-impact-review.template.mdc`

## Notes

- These are intentionally strict for stabilization phases.
- You can relax checklist sections for greenfield prototyping.
