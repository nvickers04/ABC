"""
QualityMatrix — host-side context and execution quality policy.

Dataclasses + service implementing the hybrid design:
- Reuses existing evaluation engines (trade_feedback, execution_snapshots,
  calibrated_slippage, graduated_params, daily review outputs).
- First-class ToolUsageRecord and DecisionProvenanceSnapshot for provenance
  (decision-scoped; explicitly avoids noisy per-trade / per-tool outcome
  attribution as diagnosed in session analysis).
- Orchestration-heavy: provides to_prompt_block() + recommended_policies().
  Population from daily review, execution analysis, and cycle hooks.
- research_config knobs for enablement, retention, scoring weights.
- Singleton service pattern consistent with get_operating_context().

Persistence: tool_usage_log + decision_provenance tables (additive, IF NOT EXISTS).
No new heavy dependencies. Dataclass style matches codebase (OperatingContext etc).

Open decisions flagged at bottom of file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, ClassVar, Literal, Optional

from memory import get_db, get_research_config, set_research_config

logger = logging.getLogger(__name__)


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class ToolUsageRecord:
    """First-class record of a single tool invocation (tool-usage emphasis).

    Freshness / outcome back-fill deliberately omitted from this record to
    avoid the symbol-specificity + multi-tool entanglement problem. Instead,
    provenance snapshots capture the active set at decision time.
    """

    tool_name: str
    called_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: Optional[str] = None
    success: bool = True
    latency_ms: float = 0.0
    source: str = "executor"  # executor | researcher | internal
    # Optional lightweight context at call time (e.g. intent, query snippet)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionProvenanceSnapshot:
    """Decision-time provenance capture.

    Records the exact quality state + active/recent tools at the moment of a
    key decision (cycle done, entry idea, sizing, review trigger, etc.).
    Forward-looking only. Outcomes (if any) are attached at high level, never
    retroactively mutate historical ToolUsageRecords with P&L.
    """

    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cycle_id: int = 0
    decision_type: str = "cycle_decision"  # cycle_decision | entry_idea | sizing | review | done
    symbol: Optional[str] = None
    tools_used: list[ToolUsageRecord] = field(default_factory=list)
    quality_state: dict[str, Any] = field(default_factory=dict)
    context_quality: Literal["full", "limited", "minimal"] = "full"
    outcome: Optional[str] = None  # e.g. "trade_placed", "passed", "wait"
    notes: str = ""


@dataclass
class SymbolQuality:
    """Aggregate, symbol-scoped quality metrics.

    Populated from trade_feedback + execution analysis (conservative reuse).
    Not a per-trade ledger.
    """

    symbol: str
    execution_quality: float = 0.5  # 0.0 (bad) .. 1.0 (excellent)
    avg_execution_gap: float = 0.0  # from trade_feedback
    trade_count_7d: int = 0
    recent_feedback_count: int = 0
    tool_usage_count: int = 0
    last_tool_success_rate: float = 1.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""


@dataclass
class QualityMatrix:
    """Top-level quality orchestration object.

    Lives in memory for the process; persisted via provenance + usage logs.
    Provides the two key methods required by charter: to_prompt_block() and
    recommended_policies().
    """

    overall_quality: Literal["full", "limited", "minimal", "degraded"] = "full"
    risk_multiplier: float = 1.0
    symbol_quality: dict[str, SymbolQuality] = field(default_factory=dict)
    recent_tool_usage: list[ToolUsageRecord] = field(default_factory=list)
    recent_provenance: list[DecisionProvenanceSnapshot] = field(default_factory=list)

    suggested_temperature: float = 0.3
    suggested_max_tokens: int = 2048
    force_conservative_reasoning: bool = False
    blocked_tool_categories: list[str] = field(default_factory=list)  # e.g. ["research", "complex_options"]

    global_execution_quality: float = 0.5

    last_populated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_prompt_block(self, risk_multiplier: float | None = None) -> str:
        """Compact, human-readable block suitable for LLM prompt injection.

        Mirrors the style of ContextQuality.to_prompt_block for consistency.
        """
        rm = risk_multiplier if risk_multiplier is not None else self.risk_multiplier
        lines = ["═══ QUALITY MATRIX ═══"]

        lines.append(f"Overall: {self.overall_quality.upper()} (risk ×{rm:.2f})")
        if self.force_conservative_reasoning:
            lines.append("Policy: CONSERVATIVE REASONING FORCED (higher bar for new risk)")

        if self.blocked_tool_categories:
            lines.append(f"Restricted categories: {', '.join(self.blocked_tool_categories)}")

        lines.append(f"Suggested model config: temp={self.suggested_temperature:.2f}, max_tokens={self.suggested_max_tokens}")

        # Symbol highlights (top 3 by data volume or worst quality)
        if self.symbol_quality:
            sorted_syms = sorted(
                self.symbol_quality.values(),
                key=lambda s: (s.trade_count_7d + s.recent_feedback_count, -abs(s.avg_execution_gap)),
                reverse=True,
            )[:3]
            sym_lines = []
            for sq in sorted_syms:
                sym_lines.append(
                    f"  {sq.symbol}: exec_q={sq.execution_quality:.2f} gap={sq.avg_execution_gap:+.3f} n={sq.trade_count_7d + sq.recent_feedback_count}"
                )
            if sym_lines:
                lines.append("Symbol quality (recent):")
                lines.extend(sym_lines)

        # Recent provenance summary (last 2)
        if self.recent_provenance:
            lines.append("Recent decisions (provenance):")
            for p in self.recent_provenance[-2:]:
                tools = ",".join(t.tool_name for t in p.tools_used[-3:]) or "none"
                lines.append(f"  C{p.cycle_id} {p.decision_type}({p.symbol or ''}) tools=[{tools}] q={p.context_quality}")

        lines.append(f"Last updated: {(datetime.now(timezone.utc) - self.last_populated).total_seconds() / 60:.0f}m ago")
        return "\n".join(lines)

    def recommended_policies(self, symbol: str | None = None) -> dict[str, Any]:
        """Structured policy recommendations for orchestration / risk / filtering.

        Safe to call from anywhere; never mutates state.
        """
        sq = self.symbol_quality.get(symbol) if symbol else None

        base_risk = self.risk_multiplier
        if sq and sq.execution_quality < 0.35:
            base_risk = min(base_risk, 0.5)

        return {
            "risk_multiplier": round(base_risk, 3),
            "force_conservative_reasoning": self.force_conservative_reasoning or (sq is not None and sq.execution_quality < 0.4),
            "suggested_temperature": self.suggested_temperature,
            "suggested_max_tokens": self.suggested_max_tokens,
            "blocked_tool_categories": list(self.blocked_tool_categories),
            "symbol_execution_quality": sq.execution_quality if sq else None,
            "notes": (sq.notes if sq else "") or f"overall={self.overall_quality}",
        }

    # ── Host enforcement (hard gates; LLM cannot bypass) ─────────────────────
    # Host code (not the LLM) makes
    # irreversible decisions. The LLM sees the state via to_prompt_block() +
    # quality_* tools (soft), but cannot bypass the gates below (hard).

    _ENTRY_ACTIONS: ClassVar[set[str]] = {
        "plan_order", "buy", "sell", "market_order", "limit_order",
        "enter_option", "bracket_order", "stop_order", "trailing_stop",
        "vertical_spread", "iron_condor", "straddle", "butterfly",
    }

    def _categorize_tool(self, action: str) -> Optional[str]:
        """Internal category mapper used by both prompt rendering and hard gates."""
        a = (action or "").lower()
        if a in ("research", "web_search", "x_search", "deep_research", "research_engine"):
            return "research"
        if a in ("enter_option", "vertical_spread", "iron_condor", "straddle", "butterfly", "calendar_spread"):
            return "complex_options"
        if a in ("complex_orders", "adaptive_order", "vwap_order", "iceberg_order") or a.endswith("_spread"):
            return "complex_orders"
        if a in ("plan_order", "market_order", "limit_order", "buy", "sell") and self.risk_multiplier < 0.55:
            return "high_risk_entry"
        return None

    def get_llm_call_config(self) -> dict[str, Any]:
        """Hard host-controlled sampling parameters for the xAI chat.create() call.

        This is the primary 'model call configuration' lever. Lower temp + tokens
        when quality is poor forces the model toward conservative, short, high-signal
        reasoning instead of creative exploration. The values here override any
        defaults in the agent loop unconditionally.
        """
        temp = float(self.suggested_temperature)
        mt = int(self.suggested_max_tokens)

        if self.overall_quality == "degraded":
            temp = min(temp, 0.05)
            mt = min(mt, 3072)
        elif self.overall_quality in ("minimal", "limited"):
            temp = min(temp, 0.18)
            mt = min(mt, 5500)

        if self.force_conservative_reasoning:
            temp = min(temp, 0.12)

        return {
            "temperature": round(temp, 3),
            "max_tokens": int(mt),
            "top_p": 0.78 if self.force_conservative_reasoning else 0.93,
            "reasoning_bias": "conservative" if self.force_conservative_reasoning or self.overall_quality != "full" else "balanced",
            "source": "QualityMatrix",
            "quality": self.overall_quality,
        }

    def should_allow_tool(self, action: str, params: Optional[dict] = None, *, is_independent_mode: bool = False) -> tuple[bool, Optional[str]]:
        """Authoritative hard gate for the ToolExecutor.

        Returns (allowed: bool, rejection_reason: str | None).
        Called *before* any dispatch or side-effect. The returned error is
        surfaced to the LLM as a ToolResult so it can adapt, but the action
        is never executed.

        Host is authoritative; the LLM only sees the rejection reason.
        """
        params = params or {}
        cat = self._categorize_tool(action)

        if cat and cat in (self.blocked_tool_categories or []):
            return False, (
                f"Tool '{action}' REJECTED (category={cat}) by QualityMatrix policy. "
                f"overall_quality={self.overall_quality}, risk_mult={self.risk_multiplier:.2f}. "
                "Host-level block — matrix is authoritative."
            )

        # Strong new-risk prohibition when quality is critically low or in strict independent posture
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
                if is_independent_mode and self.risk_multiplier < 0.55:
                    return False, (
                        "Independent Mode + QualityMatrix risk posture prohibits new entries. "
                        f"(rm={self.risk_multiplier:.2f}). Focus on existing positions only."
                    )

        return True, None

    def get_scaled_quantity(self, requested: float | int, symbol: Optional[str] = None, intent: str = "entry") -> int:
        """Hard scaling applied by executor on every order path.

        Agent proposes; host guarantees the cap. Returns the final safe quantity.
        """
        pol = self.recommended_policies(symbol)
        mult = float(pol.get("risk_multiplier", self.risk_multiplier))

        if intent == "entry" and self.overall_quality in ("minimal", "degraded"):
            mult = min(mult, 0.25)

        if self.force_conservative_reasoning and intent == "entry":
            mult = min(mult, 0.6)

        scaled = max(1, int(float(requested) * mult))
        return scaled

    def can_initiate_new_risk(self, symbol: Optional[str] = None) -> bool:
        """Quick predicate for host paths that want a single yes/no before even scaling."""
        pol = self.recommended_policies(symbol)
        rm = pol["risk_multiplier"]
        return rm > 0.40 and self.overall_quality not in ("minimal", "degraded")

    # ── Lightweight hooks on OperatingContext (population stays on QualityMatrixService) ──

    def update_researcher(self, available: bool, ts: Optional[datetime] = None) -> None:
        """Light hint from OperatingContext.set_researcher_*.
        Does not replace the authoritative orchestration-driven populate()."""
        if not available:
            if self.overall_quality == "full":
                self.overall_quality = "limited"
                self.force_conservative_reasoning = True
            self.risk_multiplier = min(self.risk_multiplier, 0.65)

    def record_tool_usage(self, tool_name: str, symbol: Optional[str] = None,
                          freshness: float = 1.0, source: str = "live", **meta: Any) -> None:
        """No-op — ToolUsageRecord is recorded via QualityMatrixService in the executor."""
        # Intentionally minimal — real provenance captured in executor/agent via service.
        pass

    def snapshot_decision(self, action: str, symbols: Optional[list[str]] = None, tools_used: Optional[list[str]] = None) -> None:
        """No-op — provenance is recorded via QualityMatrixService.record_decision_snapshot."""
        pass

    def get_model_overrides(self) -> dict[str, Any]:
        """Legacy compat alias. New code should prefer get_llm_call_config() for the strong host-controlled values."""
        cfg = self.get_llm_call_config()
        return {
            "temperature": cfg.get("temperature", self.suggested_temperature),
            "max_tokens": cfg.get("max_tokens", self.suggested_max_tokens),
            "reasoning_bias": cfg.get("reasoning_bias", "balanced"),
        }

    def get_blocked_tool_categories(self) -> list[str]:
        """Legacy compat. Returns the blocked list used by should_allow_tool."""
        return list(self.blocked_tool_categories or [])

    def reset_for_new_session(self) -> None:
        """Light day-boundary trim (keeps recent history for provenance)."""
        if len(self.recent_tool_usage) > 5:
            self.recent_tool_usage = self.recent_tool_usage[-5:]
        if len(self.recent_provenance) > 3:
            self.recent_provenance = self.recent_provenance[-3:]


# ── Service ─────────────────────────────────────────────────────────────────


class QualityMatrixService:
    """Orchestration-facing service. Singleton via get_quality_matrix_service()."""

    _instance: Optional["QualityMatrixService"] = None

    def __init__(self) -> None:
        self.matrix = QualityMatrix()
        self._enabled = True
        self._max_recent_tools = 50
        self._max_recent_provenance = 30
        self._logger = logging.getLogger(__name__ + ".service")

    @classmethod
    def get(cls) -> "QualityMatrixService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _refresh_knobs(self) -> None:
        try:
            self._enabled = bool(get_research_config("quality_matrix_enabled", 1.0) >= 0.5)
            self._max_recent_tools = int(get_research_config("quality_matrix_max_tools", 50.0))
            self._max_recent_provenance = int(get_research_config("quality_matrix_max_provenance", 30.0))
        except Exception:
            pass

    # ── Population (hybrid reuse) ─────────────────────────────────────────

    def maybe_populate(self, db: Any = None, *, max_age_seconds: float = 60.0) -> None:
        """Re-populate only when the in-memory matrix is stale (default 60s)."""
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
        """Full (re)population from all available sources.

        Called from daily review, after execution analysis, and on demand.
        Conservative: derives from trade_feedback aggregates + operating context.
        """
        if not self._enabled:
            return
        self._refresh_knobs()
        if db is None:
            db = get_db()

        try:
            # 1. Researcher / operating context baseline (reuse existing)
            from core.runtime.operating_context import get_operating_context
            ctx = get_operating_context()
            base_quality = ctx.quality.overall_quality
            researcher_ok = bool(ctx.quality.researcher_available)
            # Context-only baseline — must not read matrix.risk_multiplier (feedback loop).
            base_rm = float(ctx.legacy_risk_multiplier)

            # 2. Aggregate execution quality from trade_feedback (reuse daily review data)
            rows = db.execute(
                """SELECT symbol,
                          COUNT(*) as n,
                          AVG(execution_gap) as avg_gap,
                          SUM(CASE WHEN actual_pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as winrate
                   FROM trade_feedback
                   WHERE ts >= date('now', '-7 days') || 'T00:00:00' AND symbol IS NOT NULL
                   GROUP BY symbol
                   ORDER BY n DESC LIMIT 50""",
            ).fetchall()

            now = datetime.now(timezone.utc)
            new_symq: dict[str, SymbolQuality] = {}
            total_gap = 0.0
            total_n = 0

            for r in rows:
                sym = r["symbol"]
                n = int(r["n"] or 0)
                avg_gap = float(r["avg_gap"] or 0.0)
                # Map gap to quality: 0 gap → 0.9, 2% gap → 0.1 (conservative scaling)
                exec_q = max(0.05, min(0.95, 0.75 - abs(avg_gap) * 25.0))
                sq = SymbolQuality(
                    symbol=sym,
                    execution_quality=round(exec_q, 3),
                    avg_execution_gap=round(avg_gap, 4),
                    recent_feedback_count=n,
                    trade_count_7d=n,  # proxy; could join trades if needed
                    last_updated=now,
                )
                new_symq[sym] = sq
                total_gap += abs(avg_gap) * n
                total_n += n

            self.matrix.symbol_quality = new_symq

            # Global exec quality (weighted) — now a declared field
            global_exec = 0.5
            if total_n > 0:
                global_exec = max(0.1, min(0.9, 0.7 - (total_gap / total_n) * 20))
            self.matrix.global_execution_quality = round(global_exec, 3)

            # 3. Load recent provenance / tool usage from persistent log (tool purist)
            self._load_recent_from_db(db)

            # 4. Derive overall + guidance (orchestration heavy)
            has_bad_exec = global_exec < 0.35 or any(
                s.execution_quality < 0.3 for s in new_symq.values()
            )

            if not researcher_ok or base_quality == "minimal":
                self.matrix.overall_quality = "minimal"
                self.matrix.risk_multiplier = min(base_rm, 0.4)
                self.matrix.force_conservative_reasoning = True
                self.matrix.suggested_temperature = 0.15
                self.matrix.blocked_tool_categories = ["research"] if not researcher_ok else []
            elif has_bad_exec or base_quality == "limited":
                self.matrix.overall_quality = "limited"
                self.matrix.risk_multiplier = min(base_rm, 0.65)
                self.matrix.force_conservative_reasoning = True
                self.matrix.suggested_temperature = 0.22
            else:
                self.matrix.overall_quality = "full"
                self.matrix.risk_multiplier = base_rm
                self.matrix.force_conservative_reasoning = False
                self.matrix.suggested_temperature = 0.3
                self.matrix.blocked_tool_categories = []

            self.matrix.last_populated = now

            # Persist a lightweight snapshot marker (non-fatal if DB/config unavailable).
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
        """Load recent ToolUsage + Provenance for the in-memory matrix (tool provenance)."""
        try:
            # Tool usage
            rows = db.execute(
                "SELECT * FROM tool_usage_log ORDER BY ts DESC LIMIT ?",
                (self._max_recent_tools,),
            ).fetchall()
            self.matrix.recent_tool_usage = []
            for r in reversed(rows):  # oldest first for list order
                rec = ToolUsageRecord(
                    tool_name=r["tool_name"],
                    called_at=datetime.fromisoformat(r["ts"]) if r["ts"] else datetime.now(timezone.utc),
                    symbol=r["symbol"],
                    success=bool(r["success"]),
                    latency_ms=float(r["latency_ms"] or 0),
                    source=r["decision_context"] or "executor",
                )
                self.matrix.recent_tool_usage.append(rec)

            # Provenance
            prow = db.execute(
                "SELECT * FROM decision_provenance ORDER BY ts DESC LIMIT ?",
                (self._max_recent_provenance,),
            ).fetchall()
            self.matrix.recent_provenance = []
            for r in reversed(prow):
                tools_json = r["tools_json"] or "[]"
                try:
                    tools_list = json.loads(tools_json)
                    tools = [ToolUsageRecord(**t) if isinstance(t, dict) else ToolUsageRecord(tool_name=str(t)) for t in tools_list]
                except Exception:
                    tools = []
                qstate = {}
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
            # Tables may not exist on first run before schema applied; degrade gracefully
            if "no such table" not in str(e).lower():
                self._logger.debug("Provenance load skipped: %s", e)

    def update_from_daily_review(self, gap_rows: list[dict], db: Any = None, today: str = "") -> None:
        """Lightweight incremental update hook called from run_daily_review.

        Reuses the exact gap aggregation already computed by the review engine.
        """
        if not self._enabled:
            return
        if db is None:
            db = get_db()
        # Trigger a light refresh + re-derive
        self.populate(db)  # full for v1 is cheap enough; future: incremental

    def update_from_execution_analysis(
        self,
        snapshots: list[dict],
        calibrated_count: int,
        graduated_active: list[dict],
        db: Any = None,
    ) -> None:
        """Hook called after the heavy lifting in _run_execution_analysis.

        Feeds calibrated_slippage / graduated_params success into symbol/global exec quality.
        """
        if not self._enabled:
            return
        if db is None:
            db = get_db()
        # For v1 we simply re-populate (reuses the feedback + new analysis side-effects)
        self.populate(db)
        # Future: could boost execution_quality for symbols that benefited from graduated params.

    # ── Recording (tool purist + decision provenance) ─────────────────────

    def record_tool_usage(self, record: ToolUsageRecord) -> None:
        """Record a tool call. Called from agent/tool executor path."""
        if not self._enabled:
            return
        self.matrix.recent_tool_usage.append(record)
        if len(self.matrix.recent_tool_usage) > self._max_recent_tools:
            self.matrix.recent_tool_usage = self.matrix.recent_tool_usage[-self._max_recent_tools:]

        # Persist
        try:
            db = get_db()
            db.execute(
                """INSERT INTO tool_usage_log
                   (ts, cycle_id, tool_name, symbol, success, latency_ms, decision_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.called_at.isoformat(),
                    0,  # cycle_id filled by caller when known
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
        """Record a decision provenance snapshot (at done / key choice points)."""
        if not self._enabled:
            return
        self.matrix.recent_provenance.append(snap)
        if len(self.matrix.recent_provenance) > self._max_recent_provenance:
            self.matrix.recent_provenance = self.matrix.recent_provenance[-self._max_recent_provenance:]

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
        return self.matrix


def get_quality_matrix_service() -> QualityMatrixService:
    """Public entry point (mirrors get_operating_context pattern)."""
    return QualityMatrixService.get()


def reset_quality_matrix_service_for_tests() -> None:
    """Drop the process singleton so tests start from a clean matrix."""
    QualityMatrixService._instance = None
