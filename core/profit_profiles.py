"""Validated profitability presets applied across all ProfitConfig sub-configs."""

from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from core.loop_config import LoopConfig
from core.memory_config import MemoryConfig
from core.prompt_config import PromptConfig
from core.risk_execution_config import RiskExecutionConfig
from core.tool_registry import ToolRegistry

ProfitProfile = Literal["conservative", "balanced", "aggressive"]

PROFIT_PROFILE_ENV = "PROFIT_PROFILE"

VALID_PROFILES: frozenset[str] = frozenset({"conservative", "balanced", "aggressive"})

_PROFILE_NOTES: dict[str, str] = {
    "conservative": (
        "Capital preservation: tighter loss rails, lower LLM spend, stricter quality gates, "
        "smaller prompts, de-emphasized expensive tools."
    ),
    "balanced": "Default env-derived settings (no profile overlay).",
    "aggressive": (
        "Edge seeking: higher spend caps, looser gates, richer context, research tools favored "
        "(still bounded by safety rails)."
    ),
}

M = TypeVar("M", bound=BaseModel)


def normalize_profit_profile(profile: str) -> ProfitProfile:
    """Return a validated profile name."""
    key = str(profile or "balanced").strip().lower()
    if key not in VALID_PROFILES:
        allowed = ", ".join(sorted(VALID_PROFILES))
        raise ValueError(f"profit profile must be one of: {allowed} (got {profile!r})")
    return key  # type: ignore[return-value]


def _patch_model(model: M, updates: dict[str, Any]) -> M:
    if not updates:
        return model
    return type(model).model_validate({**model.model_dump(), **updates})


# ── Risk & execution ────────────────────────────────────────────────────────

RISK_PROFILE_PATCHES: dict[ProfitProfile, dict[str, Any]] = {
    "conservative": {
        "max_daily_loss_pct": 10.0,
        "intraday_drawdown_pct": 2.0,
        "max_daily_llm_cost": 3.0,
        "max_daily_multi_agent_research_usd": 0.35,
        "multi_agent_research_enabled": False,
        "cycle_sleep_seconds": 45,
        "researcher_daily_token_cap": 75_000,
        "max_daily_llm_completion_tokens": 60_000,
        "max_daily_llm_reasoning_tokens": 90_000,
    },
    "balanced": {},
    "aggressive": {
        "max_daily_loss_pct": 20.0,
        "intraday_drawdown_pct": 5.0,
        "max_daily_llm_cost": 6.5,
        "max_daily_multi_agent_research_usd": 1.25,
        "multi_agent_research_enabled": True,
        "cycle_sleep_seconds": 20,
        "researcher_daily_token_cap": 150_000,
        "max_daily_llm_completion_tokens": 100_000,
        "max_daily_llm_reasoning_tokens": 160_000,
    },
}

# ── Agent loop ────────────────────────────────────────────────────────────────

LOOP_PROFILE_PATCHES: dict[ProfitProfile, dict[str, Any]] = {
    "conservative": {
        "react_max_turns_per_cycle": 10,
        "react_max_consecutive_tool_failures": 3,
        "react_max_failures_per_cycle": 5,
        "react_tool_feedback_max_chars": 3500,
        "react_playbook_max_chars": 900,
        "gap_guard_spy_move_pct": 1.5,
        "gap_guard_delay_minutes": 20,
        "min_rm_for_new_risk": 0.5,
        "limited_posture_rm_cap": 0.55,
        "minimal_posture_rm_cap": 0.30,
        "entry_scale_conservative": 0.5,
        "llm_tokens_limited_cap": 4500,
        "llm_tokens_degraded_cap": 2500,
    },
    "balanced": {},
    "aggressive": {
        "react_max_turns_per_cycle": 0,
        "react_max_consecutive_tool_failures": 8,
        "react_max_failures_per_cycle": 12,
        "react_tool_feedback_max_chars": 6000,
        "react_playbook_max_chars": 1500,
        "gap_guard_spy_move_pct": 2.5,
        "gap_guard_delay_minutes": 10,
        "min_rm_for_new_risk": 0.55,
        "limited_posture_rm_cap": 0.75,
        "minimal_posture_rm_cap": 0.5,
        "entry_scale_conservative": 0.75,
        "llm_tokens_limited_cap": 6500,
        "llm_tokens_degraded_cap": 4000,
    },
}

# ── Memory & prompt budget ───────────────────────────────────────────────────

MEMORY_PROFILE_PATCHES: dict[ProfitProfile, dict[str, Any]] = {
    "conservative": {
        "cycle_wm_max_chars": 1800,
        "cycle_attention_max_chars": 900,
        "cycle_intuition_top_n": 2,
        "wm_render_cycle_max_entries": 2,
        "multi_agent_research_ttl_seconds": 1800,
        "legacy_risk_multiplier_degraded": 0.35,
        "legacy_risk_multiplier_limited": 0.55,
    },
    "balanced": {},
    "aggressive": {
        "cycle_wm_max_chars": 3200,
        "cycle_attention_max_chars": 1600,
        "cycle_intuition_top_n": 4,
        "wm_render_cycle_max_entries": 4,
        "multi_agent_research_ttl_seconds": 7200,
        "legacy_risk_multiplier_degraded": 0.55,
        "legacy_risk_multiplier_limited": 0.75,
    },
}

PROMPT_PROFILE_PATCHES: dict[ProfitProfile, dict[str, Any]] = {
    "conservative": {
        "llm_max_tokens": 6144,
        "llm_temperature": 0.0,
        "min_rr_paper": 2.5,
        "min_rr_aggressive_paper": 1.75,
        "high_conviction_confidence_floor": 0.85,
        "idle_cash_slack_pct": 20.0,
        "symbol_exec_poor_threshold": 0.4,
    },
    "balanced": {},
    "aggressive": {
        "llm_max_tokens": 10240,
        "llm_temperature": 0.05,
        "min_rr_paper": 1.8,
        "min_rr_aggressive_paper": 1.4,
        "high_conviction_confidence_floor": 0.70,
        "idle_cash_slack_pct": 40.0,
        "symbol_exec_poor_threshold": 0.30,
    },
}

# Tool weight multipliers (applied on top of registry defaults).
TOOL_WEIGHT_SCALE: dict[ProfitProfile, dict[str, float]] = {
    "conservative": {
        "research": 0.35,
        "chart_full": 0.5,
        "chart_swing": 0.6,
        "briefing": 0.85,
        "calculate_size": 1.2,
        "plan_order": 1.15,
        "limit_order": 1.1,
    },
    "balanced": {},
    "aggressive": {
        "research": 1.35,
        "chart_full": 1.15,
        "signal_breakdown": 1.1,
        "briefing": 1.05,
        "calculate_size": 1.0,
    },
}

TOOL_DISABLE: dict[ProfitProfile, tuple[str, ...]] = {
    "conservative": (),
    "balanced": (),
    "aggressive": (),
}


def apply_risk_profile(risk: RiskExecutionConfig, profile: ProfitProfile) -> RiskExecutionConfig:
    return _patch_model(risk, RISK_PROFILE_PATCHES[profile])


def apply_loop_profile(loop: LoopConfig, profile: ProfitProfile) -> LoopConfig:
    return _patch_model(loop, LOOP_PROFILE_PATCHES[profile])


def apply_memory_profile(memory: MemoryConfig, profile: ProfitProfile) -> MemoryConfig:
    return _patch_model(memory, MEMORY_PROFILE_PATCHES[profile])


def apply_prompt_profile(prompt: PromptConfig, profile: ProfitProfile) -> PromptConfig:
    return _patch_model(prompt, PROMPT_PROFILE_PATCHES[profile])


def apply_tool_profile(registry: ToolRegistry, profile: ProfitProfile) -> None:
    """Mutate registry weights / enabled flags for a profile (in-place)."""
    for name in TOOL_DISABLE[profile]:
        if registry.get_spec(name):
            registry.set_enabled(name, False)
    for name, scale in TOOL_WEIGHT_SCALE[profile].items():
        spec = registry.get_spec(name)
        if not spec:
            continue
        base = _default_tool_weight(name, spec.profitability_weight)
        registry.set_weight(name, min(10.0, max(0.0, base * scale)))


def _default_tool_weight(name: str, current: float) -> float:
    """Use current weight as baseline (already includes category defaults)."""
    return float(current)


def profile_note(profile: ProfitProfile) -> str:
    return _PROFILE_NOTES[profile]


def profile_change_lines(
    profile: ProfitProfile,
    *,
    risk: RiskExecutionConfig,
    loop: LoopConfig,
    memory: MemoryConfig,
    prompt: PromptConfig,
    tools: ToolRegistry,
) -> list[str]:
    """Human-readable lines describing non-default profile levers."""
    if profile == "balanced":
        return ["balanced: using environment defaults (no overlay)."]

    lines = [f"profile={profile!r}: {profile_note(profile)}"]
    for title, patches in (
        ("risk", RISK_PROFILE_PATCHES[profile]),
        ("loop", LOOP_PROFILE_PATCHES[profile]),
        ("memory", MEMORY_PROFILE_PATCHES[profile]),
        ("prompt", PROMPT_PROFILE_PATCHES[profile]),
    ):
        for key, val in sorted(patches.items()):
            lines.append(f"  {title}.{key} = {val!r}")
    weights = TOOL_WEIGHT_SCALE[profile]
    if weights:
        lines.append(f"  tools.weight_scale = {json.dumps(weights)}")
    disabled = [n for n in TOOL_DISABLE[profile] if tools.get_spec(n)]
    if disabled:
        lines.append(f"  tools.disabled = {disabled!r}")
    return lines


__all__ = [
    "PROFIT_PROFILE_ENV",
    "LOOP_PROFILE_PATCHES",
    "MEMORY_PROFILE_PATCHES",
    "PROMPT_PROFILE_PATCHES",
    "RISK_PROFILE_PATCHES",
    "ProfitProfile",
    "VALID_PROFILES",
    "apply_loop_profile",
    "apply_memory_profile",
    "apply_prompt_profile",
    "apply_risk_profile",
    "apply_tool_profile",
    "normalize_profit_profile",
    "profile_change_lines",
    "profile_note",
]
