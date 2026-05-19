"""Shared fakes and reset helpers for quality / operating-context tests."""

from __future__ import annotations

from typing import Any


def make_fake_quality_db(
    feedback_rows: list[dict] | None = None,
    tool_log: list[dict] | None = None,
    provenance: list[dict] | None = None,
) -> Any:
    """Minimal DB stub for QualityMatrixService.populate() and provenance loads."""
    feedback_rows = feedback_rows or []
    tool_log = tool_log or []
    provenance = provenance or []

    class _FakeDB:
        def execute(self, sql: str, params: tuple = ()):
            s = sql.lower()
            if "trade_feedback" in s and "group by symbol" in s:
                return _rows_result(feedback_rows)
            if "tool_usage_log" in s:
                return _rows_result(tool_log)
            if "decision_provenance" in s:
                return _rows_result(provenance)
            return _rows_result([])

        def commit(self) -> None:
            pass

    return _FakeDB()


def _rows_result(data: list[dict]):
    class _Res:
        def fetchall(self):
            return [dict(x) for x in data]

        def fetchone(self):
            return dict(data[0]) if data else None

    return _Res()


def reset_quality_runtime_state(tmp_wm_path=None) -> None:
    """Reset operating context, quality matrix, and optional local WM file."""
    from core.runtime.operating_context import reset_operating_context_for_tests
    from core.quality.quality_matrix import reset_quality_matrix_service_for_tests
    from core.runtime.local_memory_fallback import reset_local_working_memory_for_tests

    reset_operating_context_for_tests()
    reset_quality_matrix_service_for_tests()
    reset_local_working_memory_for_tests(tmp_wm_path)
