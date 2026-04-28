"""Per-domain memory repositories (scaffolding).

This package provides domain-grouped re-exports of the public API
currently living in :mod:`memory.__init__`. The legacy single-module
layout still owns the SQL — these modules only re-export — so existing
imports (``from memory import record_trade``) keep working byte-for-byte.

Migration plan (future PR, not this one):
    1. Move SQL bodies for each domain out of ``memory/__init__.py`` into
       the corresponding ``memory.repos.*`` module.
    2. Have ``memory/__init__.py`` import-and-re-export from
       ``memory.repos.*`` for back-compat.
    3. Update callers to import from the new paths over time.

Domain split:
    - schema:       init_db / get_db / migrations
    - config_repo:  research_config + graduated_params + calibration_version
    - execution_repo: trades, execution snapshots, IV snapshots,
                      execution costs, calibrated slippage
    - feedback_repo:  trade_feedback matching, open hypotheses
"""

from __future__ import annotations

__all__: list[str] = []
