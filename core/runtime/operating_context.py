"""
Operating Context & Context Quality for the Trader.

This is the central place where the trader tracks:
- Whether the researcher is available
- What source its durable memory is coming from (Postgres vs local fallback)
- The overall quality/reliability of the information it has available

The goal is to make the agent explicitly aware of the quality of its own context
so it can reason about it (via tools) and so the system can enforce hard policies
when quality is low (Independent Mode / researcher unavailable).

This is designed for day-trading resilience first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

# Canonical QualityMatrix lives in core/quality; this module holds mode flags
# and legacy ContextQuality for backward compatibility.
try:
    from core.quality.quality_matrix import (
        QualityMatrix as _CoreQM,
        ToolUsageRecord as _CoreToolRec,
        DecisionProvenanceSnapshot as _CoreProv,
        get_quality_matrix_service,
    )
except Exception:  # safe degrade if import order / circular during bootstrap
    _CoreQM = None
    _CoreToolRec = None
    _CoreProv = None
    get_quality_matrix_service = None

# Bind canonical name into this module's namespace so dataclass annotations
# and default_factory work regardless of import success (fallback handled below).
if _CoreQM is not None:
    QualityMatrix = _CoreQM
else:
    # Minimal fallback only for catastrophic import failure (keeps module loadable)
    @dataclass
    class QualityMatrix:  # type: ignore[no-redef]
        overall_quality: Literal["full", "limited", "minimal", "degraded"] = "minimal"
        risk_multiplier: float = 0.4
        suggested_temperature: float = 0.3
        suggested_max_tokens: int = 2048
        force_conservative_reasoning: bool = True
        blocked_tool_categories: List[str] = field(default_factory=list)
        recent_tool_usage: List[Any] = field(default_factory=list)
        recent_provenance: List[Any] = field(default_factory=list)
        global_execution_quality: float = 0.5
        last_populated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

        def to_prompt_block(self, risk_multiplier: Optional[float] = None) -> str:
            rm = risk_multiplier or self.risk_multiplier
            return f"═══ QUALITY MATRIX (FALLBACK) ═══\nOverall: {self.overall_quality.upper()} (risk ×{rm:.2f})"

        def recommended_policies(self, symbol: Optional[str] = None) -> Dict[str, Any]:
            return {"risk_multiplier": self.risk_multiplier, "force_conservative_reasoning": self.force_conservative_reasoning}

        def get_llm_call_config(self) -> Dict[str, Any]:
            return {"temperature": 0.15, "max_tokens": 3072, "reasoning_bias": "conservative", "source": "fallback"}

        def should_allow_tool(self, action: str, params: Optional[dict] = None, *, is_independent_mode: bool = False) -> tuple[bool, Optional[str]]:
            return True, None

        def get_scaled_quantity(self, requested: float | int, symbol: Optional[str] = None, intent: str = "entry") -> int:
            return max(1, int(requested * 0.5))

        def can_initiate_new_risk(self, symbol: Optional[str] = None) -> bool:
            return False

        # compat shims (match what was added to real class)
        def update_researcher(self, available: bool, ts: Optional[datetime] = None) -> None:
            if not available:
                self.overall_quality = "minimal"
                self.risk_multiplier = 0.4

        def record_tool_usage(self, *a: Any, **k: Any) -> None: pass
        def snapshot_decision(self, *a: Any, **k: Any) -> None: pass
        def get_model_overrides(self) -> Dict[str, Any]:
            return self.get_llm_call_config()
        def get_blocked_tool_categories(self) -> List[str]:
            return list(self.blocked_tool_categories)
        def reset_for_new_session(self) -> None: pass

        def __getattr__(self, name: str) -> Any:
            # Graceful degrade for any other attribute access during fallback
            if name in ("symbol_quality", "recent_tool_usage"):
                return {}
            return None

# Ensure ToolUsage* names exist for any local references (though primary now in core.quality)
ToolUsageRecord = _CoreToolRec or object
DecisionProvenanceSnapshot = _CoreProv or object


@dataclass
class ContextQuality:
    """Inspectable view of how good the trader's information currently is."""

    researcher_available: bool = False
    memory_source: Literal["postgres", "local_fallback"] = "postgres"
    last_research_update: Optional[datetime] = None
    working_memory_completeness: float = 0.0          # 0.0 = empty, 1.0 = full expected
    hypotheses_available: bool = False
    overall_quality: Literal["full", "limited", "minimal"] = "minimal"

    def to_prompt_block(self, risk_multiplier: float = 1.0) -> str:
        """Human-readable block for the LLM prompt."""
        lines = ["═══ RESEARCHER & CONTEXT STATUS ═══"]

        if self.researcher_available:
            lines.append("Researcher: CONNECTED")
            if self.last_research_update:
                age = (datetime.now(timezone.utc) - self.last_research_update).total_seconds() / 60
                lines.append(f"Last research update: {age:.0f} minutes ago")
        else:
            lines.append("Researcher: UNAVAILABLE")
            lines.append("Running in Independent Mode (no fresh research data)")

        lines.append(f"Memory Source: {self.memory_source.upper()}")
        lines.append(f"Working Memory Completeness: {self.working_memory_completeness * 100:.0f}%")

        if not self.hypotheses_available:
            lines.append("Hypotheses: NOT AVAILABLE (local or stale)")

        if self.overall_quality == "minimal":
            lines.append(f"System Policy: CONSERVATIVE MODE (risk ×{risk_multiplier}) — New risk is reduced. Prioritize position management and high-conviction setups only.")
        elif self.overall_quality == "limited":
            lines.append(f"System Policy: ELEVATED CAUTION (risk ×{risk_multiplier}) — Smaller size, higher bar for new entries.")

        return "\n".join(lines)


@dataclass
class OperatingContext:
    """Trader operating mode: researcher availability, memory source, quality summary."""

    quality: ContextQuality = field(default_factory=ContextQuality)
    matrix: QualityMatrix = field(default_factory=QualityMatrix)

    def __post_init__(self) -> None:
        """Rebind .matrix to the live QualityMatrixService singleton when available."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    # Rebind the instance attribute to the service-owned matrix object.
                    # Safe because dataclass is not frozen.
                    object.__setattr__(self, "matrix", svc.get_matrix())
        except Exception:
            # Keep the default_factory instance (or fallback) — non-fatal.
            pass

    @property
    def is_independent_mode(self) -> bool:
        return not self.quality.researcher_available or self.quality.memory_source == "local_fallback"

    def set_researcher_unavailable(self):
        self.quality.researcher_available = False
        if self.quality.memory_source == "postgres":
            self.quality.memory_source = "local_fallback"
        self._recalculate_overall_quality()
        try:
            self.matrix.update_researcher(False)
        except Exception:
            pass
        self._refresh_canonical_matrix()

    def set_researcher_available(self):
        had_local_fallback = self.quality.memory_source == "local_fallback"
        self.quality.researcher_available = True
        self.quality.memory_source = "postgres"
        self._recalculate_overall_quality()
        try:
            self.matrix.update_researcher(True)
        except Exception:
            pass
        self._refresh_canonical_matrix()
        # Recovery log only (no merge). See docs/operations/independent-mode.md
        try:
            from core.runtime.working_memory_access import log_wm_recovery_on_reconnect

            log_wm_recovery_on_reconnect(had_local_fallback=had_local_fallback)
        except Exception as exc:
            logger.debug("WM recovery summary skipped: %s", exc)

    def _recalculate_overall_quality(self):
        if self.quality.researcher_available and self.quality.memory_source == "postgres":
            self.quality.overall_quality = "full"
        elif self.quality.memory_source == "local_fallback" and self.quality.working_memory_completeness > 0.3:
            self.quality.overall_quality = "limited"
        else:
            self.quality.overall_quality = "minimal"

    @property
    def legacy_risk_multiplier(self) -> float:
        """Context-only risk scale (no QualityMatrix feedback).

        Used when populating QualityMatrix so policy does not read its own output.
        """
        if self.quality.researcher_available:
            return 1.0
        if self.quality.overall_quality == "minimal":
            return 0.4
        if self.quality.overall_quality == "limited":
            return 0.65
        return 0.85

    def sync_researcher_from_heartbeat(self) -> bool:
        """Reconcile researcher availability with the live research host heartbeat.

        Returns True when the researcher is considered available.
        """
        try:
            from core.runtime.heartbeat import is_research_host_alive

            alive = bool(is_research_host_alive())
        except Exception:
            alive = False

        if alive:
            if not self.quality.researcher_available:
                self.set_researcher_available()
            return True

        if self.quality.researcher_available:
            self.set_researcher_unavailable()
        return False

    def cycle_guidance_footer(self) -> str:
        """Tail instructions for the per-cycle user prompt."""
        if self.is_independent_mode:
            return _INDEPENDENT_MODE_GUIDANCE
        return (
            "Start by calling briefing() to assess research status. "
            "Use quality_status() when you need the current host risk posture."
        )

    def _refresh_canonical_matrix(self) -> None:
        """Re-populate QualityMatrix from current context (non-fatal)."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    svc.populate()  # will read ctx.quality + exec data and recompute policies
        except Exception:
            # Never let matrix refresh break context updates
            pass

    @property
    def risk_multiplier(self) -> float:
        """Conservative blend of QualityMatrix and legacy context risk."""
        core_rm = 1.0
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    core_rm = float(svc.get_matrix().risk_multiplier)
        except Exception:
            core_rm = 1.0

        legacy = self.legacy_risk_multiplier

        # Conservative: never take the larger value
        return min(core_rm, legacy)

    @property
    def quality_matrix(self) -> "Any":
        """Canonical QualityMatrix from core/quality when available.
        This is the source of truth for to_prompt_block(), get_llm_call_config(),
        recommended_policies(), blocked_tool_categories, should_allow_tool(), etc.
        Falls back to the local shim only on import/bootstrap failure.
        """
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    return svc.get_matrix()
        except Exception:
            pass
        return self.matrix

    def record_tool_usage(self, *a, **k):
        """Proxy to matrix; canonical recording is in ToolExecutor."""
        self.matrix.record_tool_usage(*a, **k)
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    # Best-effort: if args match ToolUsageRecord, pass through; else ignore
                    # Most callers now record directly to service, this is belt-and-suspenders.
                    if a and hasattr(a[0], "tool_name"):
                        svc.record_tool_usage(a[0])
        except Exception:
            pass

    def snapshot_decision(self, *a, **k):
        self.matrix.snapshot_decision(*a, **k)
        # Canonical provenance is recorded via DecisionProvenanceSnapshot in agent loop directly to service.

    def get_model_overrides(self) -> Dict[str, Any]:
        """LLM sampling overrides from canonical matrix when available."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    cfg = svc.get_matrix().get_llm_call_config()
                    # Adapt to the older "overrides" shape some legacy paths expect
                    return {
                        "temperature": cfg.get("temperature", 0.3),
                        "max_tokens": cfg.get("max_tokens", 8192),
                        "reasoning_bias": cfg.get("reasoning_bias", "balanced"),
                        "source": "QualityMatrix-canonical",
                    }
        except Exception:
            pass
        return self.matrix.get_model_overrides()

    def get_blocked_tool_categories(self) -> List[str]:
        """Blocked tool categories from canonical matrix when available."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    return list(svc.get_matrix().blocked_tool_categories or [])
        except Exception:
            pass
        return self.matrix.get_blocked_tool_categories()


# Global-ish instance for the running trader process
_current_context: Optional[OperatingContext] = None


def get_operating_context() -> OperatingContext:
    global _current_context
    if _current_context is None:
        _current_context = OperatingContext()
    return _current_context


def reset_operating_context_for_tests() -> None:
    """Drop the process singleton so tests do not leak mode across cases."""
    global _current_context
    _current_context = None


_INDEPENDENT_MODE_GUIDANCE = """=== INDEPENDENT MODE (researcher unavailable) ===
You are running with limited / local context only (no live researcher feed).

**First actions — read-only quality inspect tools:**
1. quality_status() — canonical QualityMatrix: overall_quality, risk_multiplier, blocked_tool_categories, llm_call_config, can_initiate_new_risk(), provenance summary.
2. quality_for_symbol(symbol) — per-symbol execution quality from local trade_feedback.
3. provenance_audit(window=12, symbol?) — recent tool usage + decision provenance snapshots.

**Decision rules:**
- New risk only on high conviction: strong quality_for_symbol score, local WM thesis, provenance_audit showing recent tool diversity.
- Prefer managing or reducing existing positions over new entries.
- Treat briefing, signals, and WM as potentially stale; cross-check with the quality tools.
- Host enforces risk_multiplier, tool blocks, and quantity scaling — do not attempt to override."""
