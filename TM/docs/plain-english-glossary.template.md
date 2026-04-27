# Plain-English Project Glossary (Template)

Use this when AI or reviews mention technical terms.

## Core Terms

- **Boundary extraction**  
  Splitting one large file/class into smaller pieces with clear jobs, without changing behavior.

- **State builder**  
  Code that gathers all current facts the agent/service needs before a decision.

- **Diff**  
  The exact lines changed between before and after.

- **Refactor**  
  Improving code structure while keeping behavior the same.

- **Contract**  
  Expected input/output shape between parts of the system.

- **Invariant**  
  A rule that must remain true after every change.

- **Regression**  
  Something that used to work but broke after a change.

- **Blast radius**  
  How far a change can affect the system.

- **Parity test**  
  Test proving behavior is unchanged before vs after a refactor.

- **Hotspot file**  
  File changed frequently and higher risk.

## Runtime Terms

- **Cycle loop**  
  Repeated pass: gather state -> decide -> act -> wait.

- **Cooldown**  
  Wait time before next cycle unless an event wakes early.

- **Wake event**  
  Signal that triggers immediate re-evaluation.

- **Safety rails**  
  Hard protections, e.g. budget/risk/timeout limits.

## Process Terms

- **Characterization test**  
  Captures current behavior so refactors stay safe.

- **Rollback**  
  Reverting a problematic change safely.

- **Migration**  
  Controlled update to data/schema/config format.

- **Cohesion**  
  Related logic stays grouped and understandable.

## Rule of Thumb

If a term is unclear, require plain-English explanation before implementation continues.
