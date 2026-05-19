"""
QualityMatrix package (PR1 core).

Hybrid design:
- Conservative reuse of trade_feedback, execution analysis, daily_review, calibrated_slippage, graduated_params.
- Tool-usage purist: first-class ToolUsageRecord + DecisionProvenanceSnapshot (decision-scoped, no retroactive per-trade P&L attribution).
- Clean boundaries: lives in core/quality/.

Provides get_quality_matrix_service() singleton for orchestration and prompt injection.
Backward compatible: does not modify ContextQuality / OperatingContext signatures.
"""

from __future__ import annotations

from core.quality.quality_matrix import (
    QualityMatrix,
    QualityMatrixService,
    SymbolQuality,
    DecisionProvenanceSnapshot,
    ToolUsageRecord,
    get_quality_matrix_service,
    reset_quality_matrix_service_for_tests,
)

__all__ = [
    "QualityMatrix",
    "QualityMatrixService",
    "SymbolQuality",
    "DecisionProvenanceSnapshot",
    "ToolUsageRecord",
    "get_quality_matrix_service",
    "reset_quality_matrix_service_for_tests",
]
