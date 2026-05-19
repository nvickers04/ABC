"""Per-domain memory repositories.

Public DB access remains ``from memory import …`` until schema migration completes.
Implementations:

- ``config_repo`` — research_config, graduated params
- ``feedback_repo`` — hypotheses
- ``execution_repo`` — execution snapshots, IV, slippage (read paths; some writers still in ``memory``)
- ``schema`` — ``get_db`` / ``init_db`` / migrations
- ``provenance_repo`` — QualityMatrix tool/decision provenance tables
- ``session_state`` (package root) — pending context + calibration version counter

See ``docs/codebase-layout.md`` for migration status.
"""

__all__: list[str] = []
