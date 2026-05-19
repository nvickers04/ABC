"""Research-host view of master ProfitConfig (scorer, combiner, caches, heartbeat).

Built from :class:`~core.central_profit_config.ComposedProfitConfig` so the research host
and trader read the same profile overlays on memory, loop, risk, and tool registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.central_profit_config import ComposedProfitConfig
    from core.loop_config import LoopConfig
    from core.memory_config import MemoryConfig
    from core.risk_execution_config import RiskExecutionConfig
    from core.tool_registry import ToolRegistry

RESEARCH_HOST_PROFILE_KEY = "research_host_profit_profile"


@dataclass(frozen=True)
class ResearchSettings:
    """Active profitability knobs for the research host and signal pipeline."""

    profile_label: str

    # Scorer universe / pacing
    tier1_universe_size: int
    deep_scan_top_n: int
    trade_rec_top_n: int
    round_delay_secs: int
    focus_symbols_scorer_limit: int

    # Agent / session research caches (trader reads same loop + memory TTLs)
    multi_agent_research_ttl_seconds: int
    multi_agent_research_cache_max_entries: int
    research_ttl_ticker_seconds: int
    research_ttl_sector_seconds: int
    research_ttl_macro_seconds: int
    react_session_research_cache_max: int
    react_research_summary_max_chars: int

    # Context summary aggressiveness (continuity fed to Grok on trader)
    cycle_last_summary_chars: int
    cycle_guidance_max_chars: int
    cycle_wm_max_chars: int

    # Combiner / signal quality
    min_shared_periods_for_combination: int
    signal_weight_lookback_days: int
    composite_trade_threshold: float
    neff_circuit_threshold: float
    neff_circuit_streak: int
    neff_warn_below: float
    snr_floor: float
    dead_signal_periods: int
    ewma_halflife_fraction: float
    ewma_halflife_min: int
    category_weight_cap: float
    stratify_by_horizon_bucket: bool
    ic_window_days: int
    ic_min_obs: int
    ic_noise_threshold: float
    ic_retire_streak: int
    ic_attribution_top_k: int
    ir_gate_min: float

    # Template evolution
    template_evolution_train_pct: float
    template_evolution_min_trades: int
    evolution_cooldown_market_hours: int
    evolution_cooldown_off_hours: int
    evolution_mutations_per_template: int
    evolution_exploration_rate: float
    evolution_worse_accept_tolerance: float
    evolution_directed_mutation_prob: float

    # Researcher spend / heartbeat (risk rails)
    researcher_daily_token_cap: int
    heartbeat_default_stale_after_s: float

    # Tool registry weights (signal-facing tools)
    tool_weight_signal_breakdown: float
    tool_weight_briefing: float
    tool_weight_research: float

    def as_heartbeat_metadata(self) -> dict[str, Any]:
        return {
            "profit_profile": self.profile_label,
            "tier1_universe_size": self.tier1_universe_size,
            "deep_scan_top_n": self.deep_scan_top_n,
            "research_ttl_ticker_s": self.research_ttl_ticker_seconds,
            "combiner_ir_gate_min": self.ir_gate_min,
        }


def _tool_weight(tools: ToolRegistry, name: str) -> float:
    spec = tools.tools.get(name)
    return float(spec.profitability_weight) if spec else 0.0


def build_research_settings(
    *,
    memory: MemoryConfig,
    loop: LoopConfig,
    risk: RiskExecutionConfig,
    tools: ToolRegistry,
    profile_label: str | None = None,
) -> ResearchSettings:
    from core.profit_config_state import get_active_profile_label

    label = profile_label or get_active_profile_label()
    return ResearchSettings(
        profile_label=label,
        tier1_universe_size=memory.scorer_tier1_universe_size,
        deep_scan_top_n=memory.scorer_deep_scan_top_n,
        trade_rec_top_n=memory.scorer_trade_rec_top_n,
        round_delay_secs=memory.scorer_round_delay_secs,
        focus_symbols_scorer_limit=memory.focus_symbols_scorer_limit,
        multi_agent_research_ttl_seconds=memory.multi_agent_research_ttl_seconds,
        multi_agent_research_cache_max_entries=memory.multi_agent_research_cache_max_entries,
        research_ttl_ticker_seconds=memory.research_ttl_ticker_seconds,
        research_ttl_sector_seconds=memory.research_ttl_sector_seconds,
        research_ttl_macro_seconds=memory.research_ttl_macro_seconds,
        react_session_research_cache_max=loop.react_session_research_cache_max,
        react_research_summary_max_chars=loop.react_research_summary_max_chars,
        cycle_last_summary_chars=memory.cycle_last_summary_chars,
        cycle_guidance_max_chars=memory.cycle_guidance_max_chars,
        cycle_wm_max_chars=memory.cycle_wm_max_chars,
        min_shared_periods_for_combination=memory.combiner_min_shared_periods,
        signal_weight_lookback_days=memory.combiner_signal_weight_lookback_days,
        composite_trade_threshold=memory.combiner_composite_trade_threshold,
        neff_circuit_threshold=memory.combiner_neff_circuit_threshold,
        neff_circuit_streak=memory.combiner_neff_circuit_streak,
        neff_warn_below=memory.combiner_neff_warn_below,
        snr_floor=memory.combiner_snr_floor,
        dead_signal_periods=memory.combiner_dead_signal_periods,
        ewma_halflife_fraction=memory.combiner_ewma_halflife_fraction,
        ewma_halflife_min=memory.combiner_ewma_halflife_min,
        category_weight_cap=memory.combiner_category_weight_cap,
        stratify_by_horizon_bucket=memory.combiner_stratify_by_horizon_bucket,
        ic_window_days=memory.combiner_ic_window_days,
        ic_min_obs=memory.combiner_ic_min_obs,
        ic_noise_threshold=memory.combiner_ic_noise_threshold,
        ic_retire_streak=memory.combiner_ic_retire_streak,
        ic_attribution_top_k=memory.combiner_ic_attribution_top_k,
        ir_gate_min=memory.combiner_ir_gate_min,
        template_evolution_train_pct=memory.template_evolution_train_pct,
        template_evolution_min_trades=memory.template_evolution_min_trades,
        evolution_cooldown_market_hours=memory.template_evolution_cooldown_market_hours,
        evolution_cooldown_off_hours=memory.template_evolution_cooldown_off_hours,
        evolution_mutations_per_template=memory.template_evolution_mutations_per_template,
        evolution_exploration_rate=memory.template_evolution_exploration_rate,
        evolution_worse_accept_tolerance=memory.template_evolution_worse_accept_tolerance,
        evolution_directed_mutation_prob=memory.template_evolution_directed_mutation_prob,
        researcher_daily_token_cap=int(risk.researcher_daily_token_cap),
        heartbeat_default_stale_after_s=float(risk.heartbeat_default_stale_after_s),
        tool_weight_signal_breakdown=_tool_weight(tools, "signal_breakdown"),
        tool_weight_briefing=_tool_weight(tools, "briefing"),
        tool_weight_research=_tool_weight(tools, "research"),
    )


def build_research_settings_from_composed(
    composed: ComposedProfitConfig,
    *,
    profile_label: str | None = None,
) -> ResearchSettings:
    return build_research_settings(
        memory=composed.memory,
        loop=composed.loop,
        risk=composed.risk,
        tools=composed.tools,
        profile_label=profile_label,
    )


__all__ = [
    "RESEARCH_HOST_PROFILE_KEY",
    "ResearchSettings",
    "build_research_settings",
    "build_research_settings_from_composed",
]
