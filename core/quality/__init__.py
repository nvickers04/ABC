"""QualityMatrix — host policy for context quality, tool gates, and provenance.

See ``docs/operations/independent-mode.md`` for Independent Mode and WM routing.
"""

from __future__ import annotations

from core.quality.quality_learning import (
    learning_enabled,
    persist_learned_weights_to_active_profile,
    record_trade_outcome_and_maybe_refit,
    trade_outcomes_from_cycle_logs,
    train_from_trade_outcomes,
)
from core.quality.quality_matrix import (
    ContextQualityLevel,
    DecisionProvenanceSnapshot,
    DecisionType,
    OverallQuality,
    ProvenanceContextQuality,
    QualityMatrix,
    QualityMatrixService,
    SymbolQuality,
    ToolGateResult,
    ToolSource,
    ToolUsageRecord,
    get_quality_matrix_service,
    reset_quality_matrix_service_for_tests,
)

__all__ = [
    "ContextQualityLevel",
    "DecisionProvenanceSnapshot",
    "DecisionType",
    "OverallQuality",
    "ProvenanceContextQuality",
    "QualityMatrix",
    "QualityMatrixService",
    "SymbolQuality",
    "ToolGateResult",
    "ToolSource",
    "ToolUsageRecord",
    "get_quality_matrix_service",
    "reset_quality_matrix_service_for_tests",
]
