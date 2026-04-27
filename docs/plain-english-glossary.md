# Plain-English Project Glossary

Use this when AI or code reviews mention technical terms.

## Core Terms

- **Boundary extraction**  
  Splitting one large file/class into smaller pieces with clear jobs, without changing behavior.

- **State builder**  
  The code that gathers all current facts the agent needs (market session, account, positions, orders, risk status).

- **Diff**  
  The exact list of lines that changed between before and after.

- **Refactor**  
  Changing code structure to make it cleaner/safer, while keeping behavior the same.

- **Contract**  
  The expected input/output shape between two parts of the system (for example tool results always include `success`, `data`, `error`).

- **Invariant**  
  A rule that must stay true after every change (for example: safety rails always trigger when thresholds are breached).

- **Regression**  
  Something that used to work but broke after a change.

- **Blast radius**  
  How far a change can impact the rest of the project.

- **Parity test**  
  A test proving behavior is unchanged before vs after a refactor.

- **Hotspot file**  
  A file changed often and therefore higher risk.

## Runtime Terms

- **Cycle loop**  
  The repeated agent pass: read state -> reason -> call tools -> decide next wait/action.

- **Cooldown**  
  How long to wait before next cycle unless an event wakes the loop sooner.

- **Wake event / wake bus**  
  A signal that tells the loop to run now (for example order fill or scorer update).

- **Safety rails**  
  Hard protections (daily loss, drawdown, end-of-day flatten, budget cap).

## Change Process Terms

- **Characterization test**  
  A test that captures current behavior, even if code is messy, so refactors can be safe.

- **Rollback**  
  Reverting a change if it causes problems.

- **Migration**  
  Controlled update to DB/schema/state format.

- **Cohesion**  
  How well related logic is grouped together and stays understandable.

## Rule of Thumb

If a term is unclear, require the AI to explain it in plain English before continuing implementation.
