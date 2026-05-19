"""Single source for agent loop timing, ReAct limits, quality gates, and JSON parse policy.

Import via :func:`get_loop_config` from ``core.agent``, ``core.runtime.cycle``,
``core.runtime.scheduler``, ``core.quality.quality_matrix``, and ``core.json_parse``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, Field, model_validator


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class LoopConfig(BaseModel):
    """Agent outer loop, ReAct inner loop, quality gates, and parse fallbacks."""

    model_config = {"frozen": True}

    # ── ReAct turn limits & circuit breakers ────────────────────────────────
    react_max_turns_per_cycle: int = Field(
        default=0,
        description=(
            "Profitability: 0 = unlimited turns until done (current behavior); "
            ">0 caps LLM/tool spend per cycle."
        ),
    )
    react_max_consecutive_tool_failures: int = Field(
        default=5,
        description="Profitability: halts cycle after N consecutive tool errors — limits loss spirals.",
    )
    react_max_failures_per_cycle: int = Field(
        default=8,
        description="Profitability: per-cycle tool failure budget before cooldown exit.",
    )
    react_invalid_json_streak_limit: int = Field(
        default=3,
        description="Profitability: ends cycle after repeated unparseable model output (saves tokens).",
    )
    react_grok_api_max_attempts: int = Field(
        default=3,
        description="Profitability: sample retries with backoff before turn abort.",
    )
    react_grok_api_backoff_base_seconds: int = Field(
        default=2,
        description="Profitability: exponential backoff base (2, 4, …) between API retries.",
    )
    react_cooldown_min_seconds: int = Field(
        default=5,
        description="Profitability: floor on done-action cooldown — keeps loop responsive on edge days.",
    )
    react_cooldown_max_seconds: int = Field(
        default=3600,
        description="Profitability: ceiling on done cooldown — prevents multi-hour blind sleeps.",
    )
    react_circuit_breaker_cooldown_seconds: int = Field(
        default=60,
        description="Profitability: sleep after JSON/tool circuit trips — time to recover context.",
    )
    react_halt_cooldown_seconds: int = Field(
        default=999999,
        description="Profitability: sentinel cooldown when halted (loss/drawdown/token limit).",
    )
    react_tool_feedback_max_chars: int = Field(
        default=4500,
        description=(
            "Profitability: caps tool result re-injected each turn; lower = cheaper context, "
            "less detail for multi-step plans."
        ),
    )
    react_tool_feedback_truncate_min_room: int = Field(
        default=200,
        description="Profitability: minimum chars kept when truncating tool feedback.",
    )
    react_playbook_max_chars: int = Field(
        default=1200,
        description="Profitability: compact tool playbook size in system prompt.",
    )
    react_grok_response_log_chars: int = Field(
        default=500,
        description="Profitability: log truncation for model output (observability only).",
    )
    react_done_summary_max_chars: int = Field(
        default=200,
        description="Profitability: caps done-summary length in logs and provenance.",
    )
    react_wait_reason_max_chars: int = Field(
        default=160,
        description="Profitability: wait_reason stored for accountability on long cooldowns.",
    )
    react_provenance_tools_tail: int = Field(
        default=5,
        description="Profitability: tool records attached to done provenance snapshot.",
    )
    react_session_research_cache_max: int = Field(
        default=15,
        description="Profitability: in-agent research result cache size per session.",
    )
    react_research_summary_max_chars: int = Field(
        default=2000,
        description="Profitability: max chars stored per research() result in session cache.",
    )
    react_tool_log_data_chars: int = Field(
        default=200,
        description="Profitability: tool OK log truncation.",
    )
    react_tool_log_error_chars: int = Field(
        default=300,
        description="Profitability: tool FAIL log truncation.",
    )
    react_cycle_actions_tail: int = Field(
        default=4,
        description="Profitability: recent actions kept in last-cycle summary.",
    )
    react_json_reprompt_cooldown_hint: int = Field(
        default=30,
        description="Profitability: example cooldown in JSON repair user message.",
    )

    # ── JSON parse fallbacks ──────────────────────────────────────────────────
    json_repair_on_decode_failure: bool = Field(
        default=True,
        description="Profitability: attempt fence/smart-quote repair before giving up on a turn.",
    )
    json_strip_code_fences: bool = Field(
        default=True,
        description="Profitability: extract JSON from ``` fences — reduces invalid-json streaks.",
    )
    json_close_truncated_braces: bool = Field(
        default=True,
        description="Profitability: balance braces on token-truncated output — recovers partial actions.",
    )
    json_normalize_smart_quotes: bool = Field(
        default=True,
        description="Profitability: fix curly quotes so valid actions parse without retry spend.",
    )
    json_strip_trailing_commas: bool = Field(
        default=True,
        description="Profitability: tolerate trailing commas common in model JSON.",
    )

    # ── Outer scheduler (CycleScheduler) ──────────────────────────────────────
    scheduler_halted_poll_seconds: int = Field(
        default=300,
        description="Profitability: poll interval when halted — avoids busy-wait, wakes on market open.",
    )
    scheduler_after_hours_sleep_seconds: int = Field(
        default=1800,
        description="Profitability: sleep when market closed after 8 PM ET — saves LLM $ overnight.",
    )
    scheduler_after_hours_threshold_hour: int = Field(
        default=20,
        description="Profitability: ET hour (24h) after which closed session triggers long sleep.",
    )
    scheduler_cycle_error_cooldown_seconds: int = Field(
        default=30,
        description="Profitability: backoff after unhandled cycle exception before retry.",
    )

    # ── Cycle helpers (gap guard — core.runtime.cycle) ────────────────────────
    gap_guard_spy_move_pct: float = Field(
        default=2.0,
        description=(
            "Profitability: overnight SPY gap % triggering entry delay — "
            "avoids chasing gap-and-go whipsaws."
        ),
    )
    gap_guard_delay_minutes: int = Field(
        default=15,
        description="Profitability: minutes to block new entries after large gap.",
    )
    gap_guard_regular_session_minutes: int = Field(
        default=390,
        description="Profitability: assumed regular-session length for minutes-since-open math.",
    )
    gap_guard_check_window_minutes: int = Field(
        default=5,
        description="Profitability: only measure gap in first N minutes after open.",
    )

    # ── Intuition block weights ─────────────────────────────────────────────
    intuition_top_n: int = Field(
        default=5,
        description="Profitability: symbols shown in INTUITION block — focuses LLM on movers.",
    )
    intuition_top_drivers: int = Field(
        default=3,
        description="Profitability: signal drivers per row — balances detail vs tokens.",
    )
    intuition_novelty_lookback_seconds: float = Field(
        default=43_200.0,
        description="Profitability: 12h novelty window — rewards fresh composite shifts.",
    )

    # ── Daily review / risk ramp ──────────────────────────────────────────────
    review_execution_gap_threshold: float = Field(
        default=0.005,
        description="Profitability: |avg execution gap| >0.5% emits sizing hypothesis.",
    )
    review_execution_analysis_min_snapshots: int = Field(
        default=10,
        description="Profitability: min new snapshots before LLM execution analysis (cost control).",
    )
    review_risk_ramp_min_trading_days: int = Field(
        default=10,
        description="Profitability: live risk ramp requires this many active days.",
    )
    review_risk_ramp_min_win_rate: float = Field(
        default=0.45,
        description="Profitability: win rate floor for 0.5% → 1.0% live risk ramp.",
    )
    review_risk_ramp_lookback_days: int = Field(
        default=30,
        description="Profitability: window for risk-ramp trade statistics.",
    )

    # ── QualityMatrix gate thresholds ─────────────────────────────────────────
    symbol_exec_quality_base: float = Field(
        default=0.75,
        description="Profitability: baseline per-symbol execution score before gap penalty.",
    )
    symbol_exec_quality_gap_coeff: float = Field(
        default=25.0,
        description="Profitability: sensitivity of symbol score to |avg execution gap|.",
    )
    symbol_exec_quality_min: float = Field(
        default=0.05,
        description="Profitability: floor on symbol execution_quality.",
    )
    symbol_exec_quality_max: float = Field(
        default=0.95,
        description="Profitability: cap on symbol execution_quality.",
    )
    global_exec_quality_default: float = Field(
        default=0.5,
        description="Profitability: neutral book-wide execution prior.",
    )
    global_exec_formula_base: float = Field(
        default=0.7,
        description="Profitability: starting point for portfolio execution score.",
    )
    global_exec_formula_gap_coeff: float = Field(
        default=20.0,
        description="Profitability: portfolio-wide gap sensitivity.",
    )
    global_exec_min: float = Field(default=0.1, description="Profitability: minimum global execution score.")
    global_exec_max: float = Field(default=0.9, description="Profitability: maximum global execution score.")
    symbol_exec_poor_threshold: float = Field(
        default=0.35,
        description="Profitability: caps risk_multiplier per symbol when execution is poor.",
    )
    symbol_exec_conservative_threshold: float = Field(
        default=0.4,
        description="Profitability: triggers conservative reasoning for weak symbols.",
    )
    symbol_risk_cap_multiplier: float = Field(
        default=0.5,
        description="Profitability: max risk scale when symbol execution_quality is poor.",
    )
    global_exec_degraded_threshold: float = Field(
        default=0.35,
        description="Profitability: book-wide score forcing limited posture.",
    )
    symbol_exec_bad_threshold: float = Field(
        default=0.3,
        description="Profitability: contributes to has_bad_exec population flag.",
    )
    high_risk_entry_rm_threshold: float = Field(
        default=0.55,
        description="Profitability: categorizes entries as high_risk when rm below this.",
    )
    independent_mode_entry_rm_threshold: float = Field(
        default=0.55,
        description="Profitability: blocks new entries in independent mode below this rm.",
    )
    min_rm_for_new_risk: float = Field(
        default=0.40,
        description="Profitability: can_initiate_new_risk requires rm above this.",
    )
    entry_scale_minimal_degraded: float = Field(
        default=0.25,
        description="Profitability: entry quantity cap in minimal/degraded quality.",
    )
    entry_scale_conservative: float = Field(
        default=0.6,
        description="Profitability: entry scale when force_conservative_reasoning.",
    )
    researcher_unavailable_rm_cap: float = Field(
        default=0.65,
        description="Profitability: max rm when researcher heartbeat is stale.",
    )
    minimal_posture_rm_cap: float = Field(
        default=0.4,
        description="Profitability: rm cap when overall_quality is minimal.",
    )
    limited_posture_rm_cap: float = Field(
        default=0.65,
        description="Profitability: rm cap when execution quality is limited.",
    )
    matrix_default_suggested_temperature: float = Field(
        default=0.3,
        description="Profitability: full-quality LLM temperature from matrix.",
    )
    matrix_default_suggested_max_tokens: int = Field(
        default=2048,
        description="Profitability: default matrix max_tokens before tier caps.",
    )
    matrix_temp_minimal: float = Field(
        default=0.15,
        description="Profitability: sampling temp when research host down.",
    )
    matrix_temp_limited: float = Field(
        default=0.22,
        description="Profitability: temp when execution quality is weak.",
    )
    matrix_temp_full: float = Field(
        default=0.3,
        description="Profitability: temp when system is healthy.",
    )
    llm_temp_degraded_cap: float = Field(
        default=0.05,
        description="Profitability: hard temp cap in degraded quality.",
    )
    llm_temp_minimal_limited_cap: float = Field(
        default=0.18,
        description="Profitability: temp cap for minimal/limited tiers.",
    )
    llm_temp_conservative_cap: float = Field(
        default=0.12,
        description="Profitability: extra cap when force_conservative_reasoning.",
    )
    llm_tokens_degraded_cap: int = Field(
        default=3072,
        description="Profitability: completion cap when degraded.",
    )
    llm_tokens_limited_cap: int = Field(
        default=5500,
        description="Profitability: token ceiling for minimal/limited tiers.",
    )
    llm_top_p_conservative: float = Field(
        default=0.78,
        description="Profitability: narrow sampling when conservative.",
    )
    llm_top_p_normal: float = Field(
        default=0.93,
        description="Profitability: normal top_p when quality is full.",
    )

    @model_validator(mode="after")
    def _finalize_config(self) -> LoopConfig:
        object.__setattr__(
            self,
            "react_tool_feedback_max_chars",
            _env_int("AGENT_TOOL_FEEDBACK_MAX_CHARS", self.react_tool_feedback_max_chars),
        )
        object.__setattr__(
            self,
            "react_playbook_max_chars",
            _env_int("TOOL_PLAYBOOK_MAX_CHARS", self.react_playbook_max_chars),
        )
        object.__setattr__(
            self,
            "gap_guard_spy_move_pct",
            _env_float("OPEN_GAP_GUARD_PCT", self.gap_guard_spy_move_pct),
        )
        object.__setattr__(
            self,
            "gap_guard_delay_minutes",
            _env_int("OPEN_GUARD_DELAY_MINUTES", self.gap_guard_delay_minutes),
        )
        if self.min_rm_for_new_risk > self.limited_posture_rm_cap:
            raise ValueError("min_rm_for_new_risk must be <= limited_posture_rm_cap")
        if self.symbol_exec_quality_min >= self.symbol_exec_quality_max:
            raise ValueError("symbol_exec_quality_min must be < symbol_exec_quality_max")
        if self.react_cooldown_min_seconds > self.react_cooldown_max_seconds:
            raise ValueError("react_cooldown_min_seconds must be <= react_cooldown_max_seconds")
        return self

    def clamp_cooldown(self, seconds: int | float) -> int:
        """Clamp agent-requested cooldown to configured bounds."""
        try:
            val = int(seconds)
        except (TypeError, ValueError):
            val = self.react_json_reprompt_cooldown_hint
        return max(self.react_cooldown_min_seconds, min(val, self.react_cooldown_max_seconds))

    def grok_api_backoff_seconds(self, attempt_index: int) -> int:
        """Exponential backoff for API attempt ``attempt_index`` (0-based)."""
        return self.react_grok_api_backoff_base_seconds ** (attempt_index + 1)

    def scheduler_defaults(self) -> dict[str, int]:
        """Dict of CycleScheduler tunables (for tests that override instance attrs)."""
        return {
            "halted_poll_seconds": self.scheduler_halted_poll_seconds,
            "after_hours_sleep_seconds": self.scheduler_after_hours_sleep_seconds,
            "after_hours_threshold_hour": self.scheduler_after_hours_threshold_hour,
            "cycle_error_cooldown_seconds": self.scheduler_cycle_error_cooldown_seconds,
        }


_loop_config_override: LoopConfig | None = None


@lru_cache(maxsize=1)
def get_loop_config() -> LoopConfig:
    """Return the process-wide :class:`LoopConfig` singleton."""
    if _loop_config_override is not None:
        return _loop_config_override
    return LoopConfig()


def install_loop_config(cfg: LoopConfig | None) -> None:
    """Pin a profile-patched instance (``None`` clears the override)."""
    global _loop_config_override
    _loop_config_override = cfg
    get_loop_config.cache_clear()


def reload_loop_config() -> LoopConfig:
    """Clear cache and return a fresh config (tests / after env refresh)."""
    install_loop_config(None)
    return get_loop_config()


__all__ = [
    "LoopConfig",
    "get_loop_config",
    "install_loop_config",
    "reload_loop_config",
]
