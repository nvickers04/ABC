"""Assemble per-cycle user prompt blocks with prompt-budget trimming."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from core.log_context import get_logger
from core.runtime.prompt_budget import (
    CYCLE_ATTENTION_MAX_CHARS,
    CYCLE_GUIDANCE_MAX_CHARS,
    CYCLE_INTUITION_TOP_N,
    CYCLE_WM_MAX_CHARS,
    CyclePromptMetrics,
    build_continuity_block,
    truncate_text,
)

if TYPE_CHECKING:
    from core.runtime.operating_context import OperatingContext

logger = get_logger(__name__)


def compact_quality_block(matrix: Any, risk_multiplier: float) -> str:
    """QualityMatrix prompt block — full signals, fewer tokens."""
    if hasattr(matrix, "to_prompt_block"):
        try:
            return matrix.to_prompt_block(risk_multiplier, compact=True) + "\n\n"
        except TypeError:
            return matrix.to_prompt_block(risk_multiplier) + "\n\n"
    return ""


def compact_wm_block(wm: Any) -> str:
    """Render working memory with per-section caps."""
    render = getattr(wm, "render", None)
    if not callable(render):
        return ""
    try:
        text = render(max_entries_per_section=3)
    except TypeError:
        text = render()
    text = truncate_text(text, CYCLE_WM_MAX_CHARS)
    return text + "\n\n" if text.strip() else ""


async def load_attention_block(conn: Any) -> str:
    from core.runtime import attention as _attention

    _attention.sync_from_working_memory(conn)
    rendered = _attention.render_attention_block(
        conn,
        max_rows=6,
        max_source_chars=48,
    )
    if not rendered:
        return ""
    return truncate_text(rendered, CYCLE_ATTENTION_MAX_CHARS) + "\n\n"


async def load_intuition_block(conn: Any) -> str:
    from core.runtime import intuition as _intuition

    rendered = _intuition.render_intuition_block(conn, top_n=CYCLE_INTUITION_TOP_N)
    return rendered + "\n\n" if rendered else ""


async def build_cycle_user_context(
    *,
    operating_context: OperatingContext,
    state_text: str,
    cost_line: str,
    continuity_text: str,
    pre_scan_prompt: str,
    gap_guard_prompt: str,
    et_now: datetime,
) -> tuple[str, CyclePromptMetrics]:
    """Build trimmed user context and size metrics (QualityMatrix block first)."""
    metrics = CyclePromptMetrics()
    ctx = operating_context
    ctx.sync_researcher_from_heartbeat()

    try:
        from core.quality.quality_matrix import get_quality_matrix_service

        svc = get_quality_matrix_service()
        svc.maybe_populate(max_age_seconds=60.0)
    except Exception as exc:
        logger.debug("quality_matrix_refresh_skipped", error=str(exc))

    try:
        from core.log_context import refresh_trader_cycle_context

        refresh_trader_cycle_context()
    except Exception:
        pass

    qm_text = compact_quality_block(ctx.quality_matrix, ctx.risk_multiplier)
    metrics.quality_chars = len(qm_text)

    wm_block = ""
    try:
        from core.runtime.working_memory_access import get_active_working_memory

        wm = get_active_working_memory()
        wm.curate()
        wm_block = compact_wm_block(wm)
        ctx.quality.working_memory_completeness = 0.85 if ctx.is_independent_mode else 0.95
    except Exception as exc:
        logger.warning("Working Memory load failed: %s", exc)
        wm_block = "(Working Memory unavailable)\n\n"
        ctx.quality.working_memory_completeness = 0.0
    metrics.wm_chars = len(wm_block)

    attn_block = ""
    try:
        from memory import get_db

        attn_block = await load_attention_block(get_db())
    except Exception as exc:
        logger.debug("attention render failed: %s", exc)
    metrics.attention_chars = len(attn_block)

    intu_block = ""
    try:
        from memory import get_db

        intu_block = await load_intuition_block(get_db())
    except Exception as exc:
        logger.debug("intuition render failed: %s", exc)
    metrics.intuition_chars = len(intu_block)

    metrics.state_chars = len(state_text)
    metrics.continuity_chars = len(continuity_text)
    inject = f"{cost_line}{pre_scan_prompt}{gap_guard_prompt}"
    metrics.inject_chars = len(inject)

    guidance = truncate_text(ctx.cycle_guidance_footer(), CYCLE_GUIDANCE_MAX_CHARS)
    metrics.guidance_chars = len(guidance)

    context = (
        f"{qm_text}{attn_block}{intu_block}{wm_block}{state_text}"
        f"{cost_line}\n"
        f"Time: {et_now.strftime('%Y-%m-%d %H:%M:%S')} ET\n"
        f"{continuity_text}{pre_scan_prompt}{gap_guard_prompt}"
        f"Account state above.\n{guidance}\n"
    )
    metrics.record_user_total()
    return context, metrics


def build_continuity_from_agent(agent: Any) -> str:
    """Rolling summaries from agent session state."""
    return build_continuity_block(
        last_cycle_summary=getattr(agent, "_last_cycle_summary", "") or "",
        last_wait_reason=getattr(agent, "_last_wait_reason", "") or "",
        last_wake_reason=getattr(agent, "_last_wake_reason", "") or "",
        market_snapshots=list(getattr(agent, "_market_snapshots", []) or []),
    )
