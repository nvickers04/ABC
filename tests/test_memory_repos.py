"""Memory repository scaffolding tests.

Locks the contract that ``memory.repos.*`` modules re-export the same
callables as ``memory.__init__``. This is the seam that lets us move
SQL bodies into per-domain modules in a future PR without breaking
existing imports.
"""

from __future__ import annotations

import memory
from memory.repos import config_repo, execution_repo, feedback_repo, schema


class TestSchemaRepo:
    def test_reexports_match_legacy(self):
        assert schema.init_db is memory.init_db
        assert schema.get_db is memory.get_db


class TestConfigRepo:
    def test_legacy_imports_still_work(self):
        # Implementations have moved into memory.repos.config_repo;
        # memory.__init__ keeps thin shims for back-compat. Both must
        # remain importable and callable.
        names = [
            "deactivate_graduated_param",
            "get_all_research_config",
            "get_calibration_version",
            "get_graduated_params",
            "get_research_config",
            "insert_graduated_param",
            "set_research_config",
            "validate_param_key",
        ]
        for name in names:
            assert callable(getattr(config_repo, name)), name
            assert callable(getattr(memory, name)), name

    def test_all_listed_in_dunder_all(self):
        for name in config_repo.__all__:
            assert hasattr(config_repo, name), name


class TestExecutionRepo:
    def test_legacy_imports_still_work(self):
        # PR13 moved the read-side implementations (record_iv_snapshot,
        # compute_iv_rank_percentile, get_execution_cost,
        # get_calibrated_slippage) into memory.repos.execution_repo;
        # memory.__init__ keeps thin shims for back-compat. Writer
        # functions (insert_execution_snapshot, etc.) still live in
        # memory.__init__ and are re-exported by execution_repo.
        # Both sides must remain importable and callable.
        names = [
            "cancel_execution_snapshot",
            "compute_iv_rank_percentile",
            "get_calibrated_slippage",
            "get_execution_cost",
            "get_filled_snapshots",
            "get_new_snapshot_count",
            "get_snapshots_for_param_review",
            "insert_execution_snapshot",
            "record_iv_snapshot",
            "record_trade",
            "update_execution_snapshot_fill",
            "upsert_calibrated_slippage",
        ]
        for name in names:
            assert callable(getattr(execution_repo, name)), name
            assert callable(getattr(memory, name)), name


class TestFeedbackRepo:
    def test_legacy_imports_still_work(self):
        # The implementation has been moved into memory.repos.feedback_repo;
        # memory.__init__ keeps a thin shim for back-compat. Both must remain
        # importable and callable.
        for name in ("get_open_hypotheses", "mark_hypothesis_incorporated"):
            assert callable(getattr(feedback_repo, name)), name
            assert callable(getattr(memory, name)), name
