"""Trader operating mode: researcher availability, context quality, QualityMatrix bridge.

Tracks whether the research host is alive, which durable memory backend is active,
and how conservative the trader should be. Feeds:

* **Prompt text** — :meth:`ContextQuality.to_prompt_block`, matrix via
  :attr:`OperatingContext.quality_matrix`
* **Risk posture** — :attr:`OperatingContext.risk_multiplier` blends context-only
  and :class:`~core.quality.quality_matrix.QualityMatrix` policy
* **WM routing** — :attr:`OperatingContext.is_independent_mode` drives
  :func:`~core.runtime.working_memory_access.get_active_working_memory`

Hard tool gates and provenance recording live in :mod:`core.quality`; this module
holds mode flags and syncs with the research host heartbeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from core.log_context import get_logger
from core.memory_config import get_memory_config
from core.prompt_config import get_prompt_config

logger = get_logger(__name__)

try:
    from core.quality.quality_matrix import (
        DecisionProvenanceSnapshot as _CoreProv,
    )
    from core.quality.quality_matrix import (
        QualityMatrix as _CoreQM,
    )
    from core.quality.quality_matrix import (
        ToolUsageRecord as _CoreToolRec,
    )
    from core.quality.quality_matrix import (
        get_quality_matrix_service,
    )
except Exception:
    _CoreQM = None  # type: ignore[assignment,misc]
    _CoreToolRec = None  # type: ignore[assignment,misc]
    _CoreProv = None  # type: ignore[assignment,misc]
    get_quality_matrix_service = None  # type: ignore[assignment]

if _CoreQM is not None:
    QualityMatrix = _CoreQM
else:

    @dataclass
    class QualityMatrix:  # type: ignore[no-redef]
        """Minimal fallback when :mod:`core.quality` cannot be imported."""

        overall_quality: Literal["full", "limited", "minimal", "degraded"] = "minimal"
        risk_multiplier: float = 0.4
        suggested_temperature: float = field(
            default_factory=lambda: get_prompt_config().fallback_matrix_temperature
        )
        suggested_max_tokens: int = field(
            default_factory=lambda: get_prompt_config().fallback_matrix_max_tokens
        )
        force_conservative_reasoning: bool = True
        blocked_tool_categories: list[str] = field(default_factory=list)
        recent_tool_usage: list[Any] = field(default_factory=list)
        recent_provenance: list[Any] = field(default_factory=list)
        global_execution_quality: float = 0.5
        last_populated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

        def to_prompt_block(self, risk_multiplier: Optional[float] = None) -> str:
            rm = risk_multiplier or self.risk_multiplier
            return (
                f"═══ QUALITY MATRIX (FALLBACK) ═══\n"
                f"Overall: {self.overall_quality.upper()} (risk ×{rm:.2f})"
            )

        def recommended_policies(self, symbol: Optional[str] = None) -> dict[str, Any]:
            return {
                "risk_multiplier": self.risk_multiplier,
                "force_conservative_reasoning": self.force_conservative_reasoning,
            }

        def get_llm_call_config(self) -> dict[str, Any]:
            pc = get_prompt_config()
            return {
                "temperature": pc.fallback_conservative_temperature,
                "max_tokens": pc.fallback_conservative_max_tokens,
                "reasoning_bias": "conservative",
                "source": "fallback",
            }

        def should_allow_tool(
            self,
            action: str,
            params: Optional[dict[str, Any]] = None,
            *,
            is_independent_mode: bool = False,
        ) -> tuple[bool, Optional[str]]:
            return True, None

        def get_scaled_quantity(
            self, requested: float | int, symbol: Optional[str] = None, intent: str = "entry"
        ) -> int:
            return max(1, int(requested * 0.5))

        def can_initiate_new_risk(self, symbol: Optional[str] = None) -> bool:
            return False

        def update_researcher(self, available: bool, ts: Optional[datetime] = None) -> None:
            if not available:
                self.overall_quality = "minimal"
                self.risk_multiplier = 0.4

        def record_tool_usage(self, *a: Any, **k: Any) -> None:
            pass

        def snapshot_decision(self, *a: Any, **k: Any) -> None:
            pass

        def get_model_overrides(self) -> dict[str, Any]:
            return self.get_llm_call_config()

        def get_blocked_tool_categories(self) -> list[str]:
            return list(self.blocked_tool_categories)

        def reset_for_new_session(self) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            if name in ("symbol_quality", "recent_tool_usage"):
                return {}
            return None


ToolUsageRecord = _CoreToolRec if _CoreToolRec is not None else Any
DecisionProvenanceSnapshot = _CoreProv if _CoreProv is not None else Any

MemorySource = Literal["postgres", "local_fallback"]
ContextOverallQuality = Literal["full", "limited", "minimal"]


@dataclass
class ContextQuality:
    """Inspectable view of trader information quality (researcher + WM).

    Distinct from :class:`~core.quality.quality_matrix.QualityMatrix`, which owns
    execution feedback and tool gates. This dataclass answers "is the research host
    connected and is WM on Postgres?"

    Attributes:
        researcher_available: True when research host heartbeat is fresh.
        memory_source: ``postgres`` or ``local_fallback``.
        last_research_update: Last seen research update time (reserved).
        working_memory_completeness: Fraction of expected WM sections populated.
        hypotheses_available: Whether hypothesis data is considered present.
        overall_quality: Coarse tier derived from researcher + WM source.
    """

    researcher_available: bool = False
    memory_source: MemorySource = "postgres"
    last_research_update: Optional[datetime] = None
    working_memory_completeness: float = 0.0
    hypotheses_available: bool = False
    overall_quality: ContextOverallQuality = "minimal"

    def to_prompt_block(self, risk_multiplier: float = 1.0) -> str:
        """Build the researcher/context section for the LLM prompt.

        Args:
            risk_multiplier: Display scale for conservative policy text.

        Returns:
            Multi-line status block.
        """
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
            lines.append(
                f"System Policy: CONSERVATIVE MODE (risk ×{risk_multiplier}) — "
                "New risk is reduced. Prioritize position management and high-conviction setups only."
            )
        elif self.overall_quality == "limited":
            lines.append(
                f"System Policy: ELEVATED CAUTION (risk ×{risk_multiplier}) — "
                "Smaller size, higher bar for new entries."
            )

        return "\n".join(lines)


@dataclass
class OperatingContext:
    """Process-wide trader mode: researcher, memory source, quality matrix link.

    Access via :func:`get_operating_context`. On init, rebinds :attr:`matrix` to the
    live :class:`~core.quality.quality_matrix.QualityMatrixService` singleton when
    available.

    Attributes:
        quality: Researcher/WM context flags.
        matrix: Quality matrix instance (may be rebound to service-owned object).
    """

    quality: ContextQuality = field(default_factory=ContextQuality)
    matrix: QualityMatrix = field(default_factory=QualityMatrix)

    def __post_init__(self) -> None:
        """Rebind ``matrix`` to the live QualityMatrixService singleton when available."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    object.__setattr__(self, "matrix", svc.get_matrix())
        except Exception:
            pass

    @property
    def is_independent_mode(self) -> bool:
        """True when researcher is down or WM is on local JSON fallback."""
        return (
            not self.quality.researcher_available
            or self.quality.memory_source == "local_fallback"
        )

    def set_researcher_unavailable(self) -> None:
        """Mark researcher offline; switch WM routing to local fallback when on Postgres."""
        self.quality.researcher_available = False
        if self.quality.memory_source == "postgres":
            self.quality.memory_source = "local_fallback"
        self._recalculate_overall_quality()
        try:
            self.matrix.update_researcher(False)
        except Exception:
            pass
        self._refresh_canonical_matrix()

    def set_researcher_available(self) -> None:
        """Mark researcher online; restore Postgres WM source and log recovery."""
        had_local_fallback = self.quality.memory_source == "local_fallback"
        self.quality.researcher_available = True
        self.quality.memory_source = "postgres"
        self._recalculate_overall_quality()
        try:
            self.matrix.update_researcher(True)
        except Exception:
            pass
        self._refresh_canonical_matrix()
        try:
            from core.runtime.working_memory_access import log_wm_recovery_on_reconnect

            log_wm_recovery_on_reconnect(had_local_fallback=had_local_fallback)
        except Exception as exc:
            logger.debug("WM recovery summary skipped: %s", exc)

    def _recalculate_overall_quality(self) -> None:
        """Derive :attr:`~ContextQuality.overall_quality` from researcher + WM source."""
        if self.quality.researcher_available and self.quality.memory_source == "postgres":
            self.quality.overall_quality = "full"
        elif (
            self.quality.memory_source == "local_fallback"
            and self.quality.working_memory_completeness
            > get_memory_config().wm_completeness_limited_threshold
        ):
            self.quality.overall_quality = "limited"
        else:
            self.quality.overall_quality = "minimal"

    @property
    def legacy_risk_multiplier(self) -> float:
        """Context-only risk scale (does not read QualityMatrix output).

        Used when populating QualityMatrix to avoid feedback loops.
        """
        return get_memory_config().legacy_risk_multiplier_for_quality(
            researcher_available=bool(self.quality.researcher_available),
            overall_quality=str(self.quality.overall_quality),
        )

    def sync_researcher_from_heartbeat(self) -> bool:
        """Reconcile researcher flags with :func:`~core.runtime.heartbeat.is_research_host_alive`.

        Returns:
            True when the research host is considered available after sync.
        """
        try:
            from core.runtime.heartbeat import is_research_host_operational

            alive = bool(is_research_host_operational())
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
        """Tail instructions appended to each cycle user prompt.

        Returns:
            Independent-mode guidance or standard connected-mode guidance.
        """
        return get_prompt_config().cycle_guidance_footer(
            independent_mode=self.is_independent_mode
        )

    def _refresh_canonical_matrix(self) -> None:
        """Trigger a non-fatal QualityMatrix populate from current context."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    svc.populate()
        except Exception:
            pass

    @property
    def risk_multiplier(self) -> float:
        """Conservative blend of matrix and legacy context risk (min of both)."""
        core_rm = 1.0
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    core_rm = float(svc.get_matrix().risk_multiplier)
        except Exception:
            core_rm = 1.0

        return min(core_rm, self.legacy_risk_multiplier)

    @property
    def quality_matrix(self) -> QualityMatrix:
        """Canonical matrix for gates, LLM config, and provenance (preferred)."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    return svc.get_matrix()
        except Exception:
            pass
        return self.matrix

    def record_tool_usage(self, *args: Any, **kwargs: Any) -> None:
        """Proxy legacy calls; prefer :class:`~core.quality.quality_matrix.QualityMatrixService`."""
        self.matrix.record_tool_usage(*args, **kwargs)
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None and args and hasattr(args[0], "tool_name"):
                    svc.record_tool_usage(args[0])
        except Exception:
            pass

    def snapshot_decision(self, *args: Any, **kwargs: Any) -> None:
        """Proxy legacy calls; provenance snapshots go to QualityMatrixService."""
        self.matrix.snapshot_decision(*args, **kwargs)

    def get_model_overrides(self) -> dict[str, Any]:
        """LLM sampling overrides from canonical matrix when available."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    cfg = svc.get_matrix().get_llm_call_config()
                    pc = get_prompt_config()
                    return {
                        "temperature": cfg.get("temperature", pc.fallback_matrix_temperature),
                        "max_tokens": cfg.get("max_tokens", pc.fallback_matrix_max_tokens),
                        "reasoning_bias": cfg.get("reasoning_bias", "balanced"),
                        "source": "QualityMatrix-canonical",
                    }
        except Exception:
            pass
        return self.matrix.get_model_overrides()

    def get_blocked_tool_categories(self) -> list[str]:
        """Blocked tool categories from canonical matrix when available."""
        try:
            if get_quality_matrix_service is not None:
                svc = get_quality_matrix_service()
                if svc is not None:
                    return list(svc.get_matrix().blocked_tool_categories or [])
        except Exception:
            pass
        return self.matrix.get_blocked_tool_categories()


_current_context: Optional[OperatingContext] = None


def get_operating_context() -> OperatingContext:
    """Return the process-wide :class:`OperatingContext` singleton."""
    global _current_context
    if _current_context is None:
        _current_context = OperatingContext()
    return _current_context


def reset_operating_context_for_tests() -> None:
    """Clear the singleton so tests do not leak mode across cases."""
    global _current_context
    _current_context = None


