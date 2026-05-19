"""Host-side context quality, tool gates, and decision provenance.

QualityMatrix is the authoritative policy layer for:

* **Risk gates** — ``should_allow_tool``, ``get_scaled_quantity``, ``can_initiate_new_risk``
* **LLM sampling** — ``get_llm_call_config`` overrides temperature and token limits
* **Provenance** — ``ToolUsageRecord`` and ``DecisionProvenanceSnapshot`` persisted via
  :class:`QualityMatrixService`

Population reuses existing DB aggregates (``trade_feedback``, execution analysis) and
:mod:`core.runtime.operating_context`. The LLM sees policy via ``to_prompt_block()`` and
quality tools; the host enforces hard gates in :mod:`tools.tools_executor` before side effects.

See ``docs/operations/independent-mode.md`` for Independent Mode and WM routing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal, Optional

from core.loop_config import get_loop_config
from core.memory_config import get_memory_config
from memory import get_db, get_research_config, set_research_config

logger = logging.getLogger(__name__)

# ── Type aliases ───────────────────────────────────────────────────────────

OverallQuality = Literal["full", "limited", "minimal", "degraded"]
ContextQualityLevel = Literal["full", "limited", "minimal"]
ProvenanceContextQuality = Literal["full", "limited", "minimal"]
ToolGateResult = tuple[bool, Optional[str]]
"""``(allowed, rejection_reason)`` from :meth:`QualityMatrix.should_allow_tool`."""

DecisionType = Literal["cycle_decision", "entry_idea", "sizing", "review", "done"]
ToolSource = Literal["executor", "researcher", "internal"]


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class ToolUsageRecord:
    """One tool invocation for provenance and audit trails.

    Recorded at execution time by :class:`QualityMatrixService.record_tool_usage`.
    Deliberately omits per-trade P&L attribution; decision snapshots capture the
    tool set active at a decision boundary instead.

    Attributes:
        tool_name: Canonical tool/action name (e.g. ``plan_order``, ``research``).
        called_at: UTC timestamp of the call.
        symbol: Optional symbol context.
        success: Whether the tool reported success.
        latency_ms: Wall-clock latency in milliseconds.
        source: Call path — ``executor``, ``researcher``, or ``internal``.
        context: Lightweight metadata (intent snippet, query id, etc.).
    """

    tool_name: str
    called_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: Optional[str] = None
    success: bool = True
    latency_ms: float = 0.0
    source: ToolSource = "executor"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionProvenanceSnapshot:
    """Quality state and tools active at a decision boundary.

    Written at cycle ``done``, entry ideas, sizing, or review triggers. Forward-looking
    only — outcomes are stored at high level in ``outcome`` without mutating past
    :class:`ToolUsageRecord` rows.

    Attributes:
        ts: Snapshot timestamp (UTC).
        cycle_id: Trader cycle identifier when known.
        decision_type: Kind of decision (``cycle_decision``, ``entry_idea``, etc.).
        symbol: Primary symbol, if any.
        tools_used: Tools considered active for this decision.
        quality_state: JSON-serializable copy of matrix/context fields at decision time.
        context_quality: Coarse context tier at decision time.
        outcome: Optional high-level outcome label (e.g. ``trade_placed``, ``wait``).
        notes: Free-form operator or host notes.
    """

    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cycle_id: int = 0
    decision_type: DecisionType = "cycle_decision"
    symbol: Optional[str] = None
    tools_used: list[ToolUsageRecord] = field(default_factory=list)
    quality_state: dict[str, Any] = field(default_factory=dict)
    context_quality: ProvenanceContextQuality = "full"
    outcome: Optional[str] = None
    notes: str = ""


@dataclass
class SymbolQuality:
    """Aggregated execution quality for one symbol (not a per-trade ledger).

    Derived from ``trade_feedback`` and execution analysis hooks. Used by
    :meth:`QualityMatrix.recommended_policies` and per-symbol risk scaling.

    Attributes:
        symbol: Ticker symbol.
        execution_quality: Score in ``[0.05, 0.95]`` — higher is better execution.
        avg_execution_gap: Mean simulated-vs-actual gap from feedback.
        trade_count_7d: Trade count proxy over the lookback window.
        recent_feedback_count: Feedback rows in the lookback window.
        tool_usage_count: Reserved for future tool-density signals.
        last_tool_success_rate: Reserved; defaults to ``1.0``.
        last_updated: UTC time of last aggregate refresh.
        notes: Optional human-readable note.
    """

    symbol: str
    execution_quality: float = 0.5
    avg_execution_gap: float = 0.0
    trade_count_7d: int = 0
    recent_feedback_count: int = 0
    tool_usage_count: int = 0
    last_tool_success_rate: float = 1.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""


@dataclass
class QualityMatrix:
    """In-process quality policy state for one trader run.

    Exposes prompt text (:meth:`to_prompt_block`), structured policy
    (:meth:`recommended_policies`), and **hard gates** that the LLM cannot bypass.
    Persisted slices are loaded/stored by :class:`QualityMatrixService`.

    Attributes:
        overall_quality: Coarse system posture.
        risk_multiplier: Host risk scale in ``(0, 1]`` applied to sizing.
        symbol_quality: Per-symbol aggregates keyed by ticker.
        recent_tool_usage: In-memory ring of recent tool calls.
        recent_provenance: In-memory ring of decision snapshots.
        suggested_temperature: LLM temperature cap suggestion.
        suggested_max_tokens: LLM max-token cap suggestion.
        force_conservative_reasoning: When True, tightens sampling and blocks entries.
        blocked_tool_categories: Category names rejected by :meth:`should_allow_tool`.
        global_execution_quality: Portfolio-wide execution score.
        last_populated: UTC time of last full populate.
    """

    overall_quality: OverallQuality = "full"
    risk_multiplier: float = 1.0
    symbol_quality: dict[str, SymbolQuality] = field(default_factory=dict)
    recent_tool_usage: list[ToolUsageRecord] = field(default_factory=list)
    recent_provenance: list[DecisionProvenanceSnapshot] = field(default_factory=list)

    suggested_temperature: float = field(
        default_factory=lambda: get_loop_config().matrix_default_suggested_temperature
    )
    suggested_max_tokens: int = field(
        default_factory=lambda: get_loop_config().matrix_default_suggested_max_tokens
    )
    force_conservative_reasoning: bool = False
    blocked_tool_categories: list[str] = field(default_factory=list)

    global_execution_quality: float = 0.5

    last_populated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_prompt_block(
        self,
        risk_multiplier: float | None = None,
        *,
        compact: bool = False,
    ) -> str:
        """Build a prompt block for the LLM user context.

        Args:
            risk_multiplier: Optional override for display; defaults to
                :attr:`risk_multiplier`.
            compact: When True, keep posture signals but drop verbose provenance
                and model-config lines (saves prompt tokens each cycle).

        Returns:
            Multi-line human-readable summary of posture, symbols, and provenance.
        """
        rm = risk_multiplier if risk_multiplier is not None else self.risk_multiplier
        lines = ["═══ QUALITY MATRIX ═══"]

        policy = " CONSERVATIVE" if self.force_conservative_reasoning else ""
        lines.append(f"Overall: {self.overall_quality.upper()} (risk ×{rm:.2f}){policy}")

        if self.blocked_tool_categories:
            lines.append(f"Blocked: {', '.join(self.blocked_tool_categories)}")

        if not compact:
            if self.force_conservative_reasoning:
                lines.append("Policy: CONSERVATIVE REASONING FORCED (higher bar for new risk)")
            lines.append(
                f"Suggested model config: temp={self.suggested_temperature:.2f}, "
                f"max_tokens={self.suggested_max_tokens}"
            )

        sym_cap = 1 if compact else 3
        if self.symbol_quality:
            sorted_syms = sorted(
                self.symbol_quality.values(),
                key=lambda s: (s.trade_count_7d + s.recent_feedback_count, -abs(s.avg_execution_gap)),
                reverse=True,
            )[:sym_cap]
            sym_lines = []
            for sq in sorted_syms:
                sym_lines.append(
                    f"  {sq.symbol}: exec_q={sq.execution_quality:.2f} "
                    f"gap={sq.avg_execution_gap:+.3f} "
                    f"n={sq.trade_count_7d + sq.recent_feedback_count}"
                )
            if sym_lines:
                prefix = "Top symbol:" if compact and len(sym_lines) == 1 else "Symbol quality:"
                lines.append(prefix)
                lines.extend(sym_lines)

        prov_cap = 1 if compact else 2
        if self.recent_provenance:
            if not compact:
                lines.append("Recent decisions (provenance):")
            for p in self.recent_provenance[-prov_cap:]:
                tools = ",".join(t.tool_name for t in p.tools_used[-2 if compact else -3:]) or "none"
                lines.append(
                    f"  C{p.cycle_id} {p.decision_type}({p.symbol or '-'}) "
                    f"tools=[{tools}] q={p.context_quality}"
                )

        age_m = (datetime.now(timezone.utc) - self.last_populated).total_seconds() / 60.0
        lines.append(f"Updated {age_m:.0f}m ago")
        return "\n".join(lines)

    def recommended_policies(self, symbol: str | None = None) -> dict[str, Any]:
        """Return structured policy knobs for orchestration and risk code.

        Does not mutate state. Symbol-specific caps apply when ``symbol`` is set
        and execution quality is poor.

        Args:
            symbol: Optional ticker for per-symbol execution quality.

        Returns:
            Dict with keys such as ``risk_multiplier``, ``force_conservative_reasoning``,
            ``suggested_temperature``, ``blocked_tool_categories``, ``symbol_execution_quality``.
        """
        lc = get_loop_config()
        sq = self.symbol_quality.get(symbol) if symbol else None

        base_risk = self.risk_multiplier
        if sq and sq.execution_quality < lc.symbol_exec_poor_threshold:
            base_risk = min(base_risk, lc.symbol_risk_cap_multiplier)

        return {
            "risk_multiplier": round(base_risk, 3),
            "force_conservative_reasoning": self.force_conservative_reasoning
            or (sq is not None and sq.execution_quality < lc.symbol_exec_conservative_threshold),
            "suggested_temperature": self.suggested_temperature,
            "suggested_max_tokens": self.suggested_max_tokens,
            "blocked_tool_categories": list(self.blocked_tool_categories),
            "symbol_execution_quality": sq.execution_quality if sq else None,
            "notes": (sq.notes if sq else "") or f"overall={self.overall_quality}",
        }

    _ENTRY_ACTIONS: ClassVar[set[str]] = {
        "plan_order", "buy", "sell", "market_order", "limit_order",
        "enter_option", "bracket_order", "stop_order", "trailing_stop",
        "vertical_spread", "iron_condor", "straddle", "butterfly",
    }

    def _categorize_tool(self, action: str) -> Optional[str]:
        """Map a tool action name to a policy category for gating.

        Args:
            action: Tool or action string from the executor.

        Returns:
            Category name (e.g. ``research``, ``complex_options``), or ``None`` if
            uncategorized.
        """
        a = (action or "").lower()
        if a in ("research", "web_search", "x_search", "deep_research", "research_engine"):
            return "research"
        if a in ("enter_option", "vertical_spread", "iron_condor", "straddle", "butterfly", "calendar_spread"):
            return "complex_options"
        if a in ("complex_orders", "adaptive_order", "vwap_order", "iceberg_order") or a.endswith("_spread"):
            return "complex_orders"
        lc = get_loop_config()
        if (
            a in ("plan_order", "market_order", "limit_order", "buy", "sell")
            and self.risk_multiplier < lc.high_risk_entry_rm_threshold
        ):
            return "high_risk_entry"
        return None

    def get_llm_call_config(self) -> dict[str, Any]:
        """Return host-controlled LLM sampling parameters.

        Values override agent defaults unconditionally when quality is degraded.

        Returns:
            Dict with ``temperature``, ``max_tokens``, ``top_p``, ``reasoning_bias``,
            ``source``, and ``quality`` keys.
        """
        lc = get_loop_config()
        temp = float(self.suggested_temperature)
        mt = int(self.suggested_max_tokens)

        if self.overall_quality == "degraded":
            temp = min(temp, lc.llm_temp_degraded_cap)
            mt = min(mt, lc.llm_tokens_degraded_cap)
        elif self.overall_quality in ("minimal", "limited"):
            temp = min(temp, lc.llm_temp_minimal_limited_cap)
            mt = min(mt, lc.llm_tokens_limited_cap)

        if self.force_conservative_reasoning:
            temp = min(temp, lc.llm_temp_conservative_cap)

        return {
            "temperature": round(temp, 3),
            "max_tokens": int(mt),
            "top_p": (
                lc.llm_top_p_conservative
                if self.force_conservative_reasoning
                else lc.llm_top_p_normal
            ),
            "reasoning_bias": (
                "conservative"
                if self.force_conservative_reasoning or self.overall_quality != "full"
                else "balanced"
            ),
            "source": "QualityMatrix",
            "quality": self.overall_quality,
        }

    def should_allow_tool(
        self,
        action: str,
        params: Optional[dict[str, Any]] = None,
        *,
        is_independent_mode: bool = False,
    ) -> ToolGateResult:
        """Hard gate: whether a tool may run (called before dispatch).

        Args:
            action: Tool/action name.
            params: Tool parameters; ``intent`` may classify entry vs exit.
            is_independent_mode: When True, applies stricter entry blocks.

        Returns:
            ``(True, None)`` if allowed; ``(False, reason)`` if rejected. The reason
            is surfaced to the LLM as a tool error.
        """
        params = params or {}
        cat = self._categorize_tool(action)

        if cat and cat in (self.blocked_tool_categories or []):
            return False, (
                f"Tool '{action}' REJECTED (category={cat}) by QualityMatrix policy. "
                f"overall_quality={self.overall_quality}, risk_mult={self.risk_multiplier:.2f}. "
                "Host-level block — matrix is authoritative."
            )

        if action in QualityMatrix._ENTRY_ACTIONS:
            intent = (params.get("intent") or "entry").lower()
            is_entry = intent in ("entry", "new", "") or action in ("buy", "enter_option")
            if is_entry:
                if self.overall_quality in ("minimal", "degraded"):
                    return False, (
                        "New risk / entry actions are HARD BLOCKED by QualityMatrix (overall="
                        f"{self.overall_quality}). Only position management, hedging, exits, "
                        "and information tools are permitted. Call quality_status() + "
                        "quality_for_symbol() to understand posture."
                    )
                if (
                    is_independent_mode
                    and self.risk_multiplier < get_loop_config().independent_mode_entry_rm_threshold
                ):
                    return False, (
                        "Independent Mode + QualityMatrix risk posture prohibits new entries. "
                        f"(rm={self.risk_multiplier:.2f}). Focus on existing positions only."
                    )

        return True, None

    def get_scaled_quantity(
        self,
        requested: float | int,
        symbol: Optional[str] = None,
        intent: str = "entry",
    ) -> int:
        """Scale an order quantity by policy (host authoritative).

        Args:
            requested: Agent-proposed quantity.
            symbol: Optional symbol for per-symbol policy.
            intent: ``entry`` vs exit/management; entries are capped more aggressively.

        Returns:
            Final integer quantity (at least 1).
        """
        lc = get_loop_config()
        pol = self.recommended_policies(symbol)
        mult = float(pol.get("risk_multiplier", self.risk_multiplier))

        if intent == "entry" and self.overall_quality in ("minimal", "degraded"):
            mult = min(mult, lc.entry_scale_minimal_degraded)

        if self.force_conservative_reasoning and intent == "entry":
            mult = min(mult, lc.entry_scale_conservative)

        return max(1, int(float(requested) * mult))

    def can_initiate_new_risk(self, symbol: Optional[str] = None) -> bool:
        """Return whether new risk entries are permitted at current posture.

        Args:
            symbol: Optional symbol for per-symbol checks.

        Returns:
            True when risk multiplier and overall quality allow new entries.
        """
        lc = get_loop_config()
        pol = self.recommended_policies(symbol)
        rm = pol["risk_multiplier"]
        return rm > lc.min_rm_for_new_risk and self.overall_quality not in ("minimal", "degraded")

    def update_researcher(self, available: bool, ts: Optional[datetime] = None) -> None:
        """Apply a lightweight researcher availability hint.

        Full population remains on :class:`QualityMatrixService.populate`.

        Args:
            available: Whether the research host heartbeat is considered fresh.
            ts: Optional timestamp of the hint (unused in v1).
        """
        del ts
        if not available:
            if self.overall_quality == "full":
                self.overall_quality = "limited"
                self.force_conservative_reasoning = True
            self.risk_multiplier = min(
                self.risk_multiplier, get_loop_config().researcher_unavailable_rm_cap
            )

    def record_tool_usage(
        self,
        tool_name: str,
        symbol: Optional[str] = None,
        freshness: float = 1.0,
        source: str = "live",
        **meta: Any,
    ) -> None:
        """Legacy no-op on matrix; use :class:`QualityMatrixService` for recording."""
        del tool_name, symbol, freshness, source, meta

    def snapshot_decision(
        self,
        action: str,
        symbols: Optional[list[str]] = None,
        tools_used: Optional[list[str]] = None,
    ) -> None:
        """Legacy no-op on matrix; use :class:`QualityMatrixService` for recording."""
        del action, symbols, tools_used

    def get_model_overrides(self) -> dict[str, Any]:
        """Legacy alias for :meth:`get_llm_call_config` shape used by older paths.

        Returns:
            Dict with ``temperature``, ``max_tokens``, and ``reasoning_bias``.
        """
        cfg = self.get_llm_call_config()
        return {
            "temperature": cfg.get("temperature", self.suggested_temperature),
            "max_tokens": cfg.get("max_tokens", self.suggested_max_tokens),
            "reasoning_bias": cfg.get("reasoning_bias", "balanced"),
        }

    def get_blocked_tool_categories(self) -> list[str]:
        """Return tool categories blocked by :meth:`should_allow_tool`.

        Returns:
            Copy of :attr:`blocked_tool_categories`.
        """
        return list(self.blocked_tool_categories or [])

    def reset_for_new_session(self) -> None:
        """Trim in-memory provenance rings at session boundaries."""
        mem = get_memory_config()
        max_tools = mem.quality_session_reset_max_tools
        max_prov = mem.quality_session_reset_max_provenance
        if len(self.recent_tool_usage) > max_tools:
            self.recent_tool_usage = self.recent_tool_usage[-max_tools:]
        if len(self.recent_provenance) > max_prov:
            self.recent_provenance = self.recent_provenance[-max_prov:]

    def learn_from_trade(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Record trade outcome for optional RL weight updates (see ``ProfitConfig.learn_from_history``)."""
        from core.quality.quality_learning import record_trade_outcome_and_maybe_refit

        return record_trade_outcome_and_maybe_refit(outcome)


class QualityMatrixService:
    """Singleton orchestrator for populate, persist, and record operations.

    Access via :func:`get_quality_matrix_service`. Toggles and retention limits
    are read from ``research_config`` keys ``quality_matrix_*``.
    """

    _instance: Optional[QualityMatrixService] = None

    def __init__(self) -> None:
        self.matrix = QualityMatrix()
        self._enabled = True
        mem = get_memory_config()
        self._max_recent_tools = mem.quality_max_recent_tools
        self._max_recent_provenance = mem.quality_max_recent_provenance
        self._logger = logging.getLogger(__name__ + ".service")

    @classmethod
    def get(cls) -> QualityMatrixService:
        """Return the process-wide singleton, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _refresh_knobs(self) -> None:
        """Reload enablement and retention limits from ``research_config``."""
        mem = get_memory_config()
        try:
            self._enabled = bool(
                get_research_config("quality_matrix_enabled", 1.0)
                >= mem.quality_matrix_enabled_threshold
            )
            self._max_recent_tools = int(
                get_research_config("quality_matrix_max_tools", float(mem.quality_max_recent_tools))
            )
            self._max_recent_provenance = int(
                get_research_config(
                    "quality_matrix_max_provenance",
                    float(mem.quality_max_recent_provenance),
                )
            )
        except Exception:
            pass

    def maybe_populate(self, db: Any = None, *, max_age_seconds: float = 60.0) -> None:
        """Populate only when the in-memory matrix is older than ``max_age_seconds``.

        Args:
            db: Optional DB handle; defaults to :func:`memory.get_db`.
            max_age_seconds: Minimum seconds since :attr:`QualityMatrix.last_populated`.
        """
        if not self._enabled:
            return
        try:
            age = (
                datetime.now(timezone.utc) - self.matrix.last_populated
            ).total_seconds()
            if age < max_age_seconds:
                return
        except Exception:
            pass
        self.populate(db)

    def populate(self, db: Any = None) -> None:
        """Recompute matrix state from DB aggregates and operating context.

        Args:
            db: Optional database connection; uses :func:`memory.get_db` when omitted.
        """
        if not self._enabled:
            return
        self._refresh_knobs()
        if db is None:
            db = get_db()

        try:
            from core.runtime.operating_context import get_operating_context

            ctx = get_operating_context()
            base_quality = ctx.quality.overall_quality
            researcher_ok = bool(ctx.quality.researcher_available)
            base_rm = float(ctx.legacy_risk_multiplier)

            mem = get_memory_config()
            lookback = mem.quality_symbol_lookback_days
            sym_limit = mem.quality_symbol_sql_limit
            rows = db.execute(
                f"""SELECT symbol,
                          COUNT(*) as n,
                          AVG(execution_gap) as avg_gap,
                          SUM(CASE WHEN actual_pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as winrate
                   FROM trade_feedback
                   WHERE ts >= date('now', '-{lookback} days') || 'T00:00:00' AND symbol IS NOT NULL
                   GROUP BY symbol
                   ORDER BY n DESC LIMIT {sym_limit}""",
            ).fetchall()

            now = datetime.now(timezone.utc)
            new_symq: dict[str, SymbolQuality] = {}
            total_gap = 0.0
            total_n = 0

            from core.quality.quality_learning import get_scoring_loop_config

            lc = get_scoring_loop_config()
            for r in rows:
                sym = r["symbol"]
                n = int(r["n"] or 0)
                avg_gap = float(r["avg_gap"] or 0.0)
                exec_q = max(
                    lc.symbol_exec_quality_min,
                    min(
                        lc.symbol_exec_quality_max,
                        lc.symbol_exec_quality_base - abs(avg_gap) * lc.symbol_exec_quality_gap_coeff,
                    ),
                )
                sq = SymbolQuality(
                    symbol=sym,
                    execution_quality=round(exec_q, 3),
                    avg_execution_gap=round(avg_gap, 4),
                    recent_feedback_count=n,
                    trade_count_7d=n,
                    last_updated=now,
                )
                new_symq[sym] = sq
                total_gap += abs(avg_gap) * n
                total_n += n

            self.matrix.symbol_quality = new_symq

            global_exec = lc.global_exec_quality_default
            if total_n > 0:
                global_exec = max(
                    lc.global_exec_min,
                    min(
                        lc.global_exec_max,
                        lc.global_exec_formula_base
                        - (total_gap / total_n) * lc.global_exec_formula_gap_coeff,
                    ),
                )
            self.matrix.global_execution_quality = round(global_exec, 3)

            self._load_recent_from_db(db)

            has_bad_exec = global_exec < lc.global_exec_degraded_threshold or any(
                s.execution_quality < lc.symbol_exec_bad_threshold for s in new_symq.values()
            )

            if not researcher_ok or base_quality == "minimal":
                self.matrix.overall_quality = "minimal"
                self.matrix.risk_multiplier = min(base_rm, lc.minimal_posture_rm_cap)
                self.matrix.force_conservative_reasoning = True
                self.matrix.suggested_temperature = lc.matrix_temp_minimal
                self.matrix.blocked_tool_categories = ["research"] if not researcher_ok else []
            elif has_bad_exec or base_quality == "limited":
                self.matrix.overall_quality = "limited"
                self.matrix.risk_multiplier = min(base_rm, lc.limited_posture_rm_cap)
                self.matrix.force_conservative_reasoning = True
                self.matrix.suggested_temperature = lc.matrix_temp_limited
            else:
                self.matrix.overall_quality = "full"
                self.matrix.risk_multiplier = base_rm
                self.matrix.force_conservative_reasoning = False
                self.matrix.suggested_temperature = lc.matrix_temp_full
                self.matrix.blocked_tool_categories = []

            self.matrix.last_populated = now

            try:
                set_research_config(
                    "quality_matrix_last_populated",
                    float(now.timestamp()),
                    f"overall={self.matrix.overall_quality} symbols={len(new_symq)}",
                )
            except Exception as cfg_err:
                self._logger.debug(
                    "quality_matrix_last_populated config write skipped: %s", cfg_err
                )

            self._logger.info(
                "QualityMatrix populated: overall=%s rm=%.2f symbols=%d tools=%d prov=%d",
                self.matrix.overall_quality,
                self.matrix.risk_multiplier,
                len(new_symq),
                len(self.matrix.recent_tool_usage),
                len(self.matrix.recent_provenance),
            )
        except Exception as exc:
            self._logger.warning("QualityMatrix populate failed (non-fatal): %s", exc)

    def _load_recent_from_db(self, db: Any) -> None:
        """Hydrate in-memory tool usage and provenance rings from Postgres.

        Args:
            db: Database connection with ``tool_usage_log`` and ``decision_provenance``.
        """
        try:
            rows = db.execute(
                "SELECT * FROM tool_usage_log ORDER BY ts DESC LIMIT ?",
                (self._max_recent_tools,),
            ).fetchall()
            self.matrix.recent_tool_usage = []
            for r in reversed(rows):
                rec = ToolUsageRecord(
                    tool_name=r["tool_name"],
                    called_at=datetime.fromisoformat(r["ts"]) if r["ts"] else datetime.now(timezone.utc),
                    symbol=r["symbol"],
                    success=bool(r["success"]),
                    latency_ms=float(r["latency_ms"] or 0),
                    source=r["decision_context"] or "executor",
                )
                self.matrix.recent_tool_usage.append(rec)

            prow = db.execute(
                "SELECT * FROM decision_provenance ORDER BY ts DESC LIMIT ?",
                (self._max_recent_provenance,),
            ).fetchall()
            self.matrix.recent_provenance = []
            for r in reversed(prow):
                tools_json = r["tools_json"] or "[]"
                try:
                    tools_list = json.loads(tools_json)
                    tools = [
                        ToolUsageRecord(**t) if isinstance(t, dict)
                        else ToolUsageRecord(tool_name=str(t))
                        for t in tools_list
                    ]
                except Exception:
                    tools = []
                qstate: dict[str, Any] = {}
                try:
                    qstate = json.loads(r["quality_state_json"] or "{}")
                except Exception:
                    pass
                snap = DecisionProvenanceSnapshot(
                    ts=datetime.fromisoformat(r["ts"]) if r["ts"] else datetime.now(timezone.utc),
                    cycle_id=int(r["cycle_id"] or 0),
                    decision_type=r["decision_type"] or "cycle_decision",
                    symbol=r["symbol"],
                    tools_used=tools,
                    quality_state=qstate,
                    context_quality=r["context_quality"] or "full",
                    outcome=r["outcome"],
                    notes=r["notes"] or "",
                )
                self.matrix.recent_provenance.append(snap)
        except Exception as e:
            if "no such table" not in str(e).lower():
                self._logger.debug("Provenance load skipped: %s", e)

    def update_from_daily_review(
        self,
        gap_rows: list[dict[str, Any]],
        db: Any = None,
        today: str = "",
    ) -> None:
        """Hook after daily review; triggers a full populate in v1.

        Args:
            gap_rows: Gap aggregates from review (reserved for incremental v2).
            db: Optional DB handle.
            today: Review date string (unused in v1).
        """
        del gap_rows, today
        if not self._enabled:
            return
        if db is None:
            db = get_db()
        self.populate(db)

    def update_from_execution_analysis(
        self,
        snapshots: list[dict[str, Any]],
        calibrated_count: int,
        graduated_active: list[dict[str, Any]],
        db: Any = None,
    ) -> None:
        """Hook after execution analysis completes.

        Args:
            snapshots: Filled execution snapshots (reserved for incremental use).
            calibrated_count: Number of calibration rows written.
            graduated_active: Active graduated params (reserved).
            db: Optional DB handle.
        """
        del snapshots, calibrated_count, graduated_active
        if not self._enabled:
            return
        if db is None:
            db = get_db()
        self.populate(db)

    def record_tool_usage(self, record: ToolUsageRecord) -> None:
        """Append a tool usage record to memory and ``tool_usage_log``.

        Args:
            record: Completed tool invocation metadata.
        """
        if not self._enabled:
            return
        self.matrix.recent_tool_usage.append(record)
        if len(self.matrix.recent_tool_usage) > self._max_recent_tools:
            self.matrix.recent_tool_usage = self.matrix.recent_tool_usage[-self._max_recent_tools:]

        try:
            db = get_db()
            db.execute(
                """INSERT INTO tool_usage_log
                   (ts, cycle_id, tool_name, symbol, success, latency_ms, decision_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.called_at.isoformat(),
                    0,
                    record.tool_name,
                    record.symbol,
                    1 if record.success else 0,
                    record.latency_ms,
                    json.dumps(record.context) if record.context else None,
                ),
            )
            db.commit()
        except Exception as e:
            logger.debug("tool_usage_log insert skipped: %s", e)

    def record_decision_snapshot(self, snap: DecisionProvenanceSnapshot) -> None:
        """Append a decision provenance snapshot to memory and ``decision_provenance``.

        Args:
            snap: Decision-boundary provenance payload.
        """
        if not self._enabled:
            return
        self.matrix.recent_provenance.append(snap)
        if len(self.matrix.recent_provenance) > self._max_recent_provenance:
            self.matrix.recent_provenance = self.matrix.recent_provenance[
                -self._max_recent_provenance:
            ]

        try:
            db = get_db()
            tools_json = json.dumps(
                [
                    {
                        "tool_name": t.tool_name,
                        "called_at": t.called_at.isoformat(),
                        "symbol": t.symbol,
                        "success": t.success,
                        "latency_ms": t.latency_ms,
                    }
                    for t in snap.tools_used
                ]
            )
            db.execute(
                """INSERT INTO decision_provenance
                   (ts, cycle_id, decision_type, symbol, tools_json, quality_state_json,
                    context_quality, outcome, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snap.ts.isoformat(),
                    snap.cycle_id,
                    snap.decision_type,
                    snap.symbol,
                    tools_json,
                    json.dumps(snap.quality_state),
                    snap.context_quality,
                    snap.outcome,
                    snap.notes,
                ),
            )
            db.commit()
        except Exception as e:
            logger.debug("decision_provenance insert skipped: %s", e)

    def get_matrix(self) -> QualityMatrix:
        """Return the live in-process matrix (mutated by populate/record)."""
        return self.matrix

    def learn_from_trade(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Store a closed-trade outcome and optionally refit bounded scoring weights.

        Expected keys (all optional except realized result): ``symbol``, ``won`` or
        ``win``, ``realized_rr``, ``profit_profile``, ``pnl_usd``, ``source``.

        Returns:
            Summary dict from :mod:`core.quality.quality_learning` (refit stats or skip reason).
        """
        result = self.matrix.learn_from_trade(outcome)
        if result.get("refitted"):
            self.populate()
        return result


def get_quality_matrix_service() -> QualityMatrixService:
    """Return the process-wide :class:`QualityMatrixService` singleton."""
    return QualityMatrixService.get()


def reset_quality_matrix_service_for_tests() -> None:
    """Clear the singleton so tests start from a fresh matrix."""
    QualityMatrixService._instance = None
