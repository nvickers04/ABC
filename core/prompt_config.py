"""Single source for LLM prompts, model slugs, sampling defaults, and quality thresholds.

Import via :func:`get_prompt_config` from agent, config, operating_context, and
quality_matrix only — do not duplicate these constants elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from core.risk_execution_config import MODE_DEFAULTS, TradingMode

_SYSTEM_PROMPT_TEMPLATE = (
    Path(__file__).with_name("_system_prompt_template.txt").read_text(encoding="utf-8")
)

# ── Mode-specific guidance (injected into system prompt) ─────────────────────

MODE_GUIDANCE_AGGRESSIVE_PAPER: str = (
    "PAPER TEST MODE — aggressive exploration. Paper capital, so try things: complex options\n"
    "(spreads, condors, calendars, straddles, diagonals), different order types, hedges, rolls.\n"
    "Fail fast, learn the tooling, stress-test the system."
)

MODE_GUIDANCE_PAPER: str = (
    "PAPER MODE — practice capital. Take good setups, manage risk, iterate."
)

MODE_GUIDANCE_LIVE: str = (
    "LIVE MODE — real money. Protect capital. Higher conviction bar, smaller size."
)

TOOL_SMOKE_PROMPT_BODY: str = """\
═══ TOOL SMOKE MODE (TOOL_SMOKE_MODE=1) ═══
You are in tool smoke mode. Goal: validate real tool wiring in paper mode by
actually using tools across live cycles, not by discussing them.

Protocol:
- Use real tools end-to-end with concrete params. Keep explanations short.
- Maintain a running checklist in memory and execute tools in this order.
- For execution-path tools, use tiny size (1 share / 1 contract) and confirm
  resulting state with account/positions/open_orders or refresh_state.
- Never call emergency-only tools (`flatten_limits`, `cancel_all_orphans`) unless
  the operator explicitly asks in this session.

Ordered smoke checklist (must progress in order; do not skip ahead unless a tool
is truly unavailable in this session):
1) Session/state baseline:
   market_hours, account, positions, open_orders, refresh_state
2) Core market data:
   quote, candles, atr, fundamentals, earnings, news, analysts
3) Extended research/context:
   iv_info, extended_fundamentals, institutional_data, insider_data,
   peer_comparison, economic_calendar, briefing, prior_research
4) Chart stack:
   chart_quick, chart_intraday, chart_swing, chart_full
5) Observability/introspection:
   budget, stats, daily_summary, review_trades, execution_status,
   open_hypotheses, signal_breakdown
6) Sizing/planning:
   calculate_size, instrument_selector, plan_order, enter_option,
   option_chain, option_quote, option_greeks, position_greeks
7) Memory and engine controls:
   update_working_memory, clear_working_memory_entry, trader_rules
   (status first; only then optional pause/resume/stop if needed)
8) Research stress (costly):
   research (run after cheap tools pass)
9) Controlled broker mutation checks (paper only, tiny size):
   a) Place one tiny stock order path (prefer limit_order OR market_order),
      then test stop management path (cancel_stops OR modify_stop if applicable).
   b) Place one tiny option path (buy_option OR vertical_spread with limit),
      then one close/roll path if a position exists (close_option/close_spread/roll_option).
   c) Optional advanced order-type checks when conditions allow:
      adaptive_order, midprice_order, relative_order, snap_mid_order,
      trailing_stop, trailing_stop_limit, stop_order, stop_limit, bracket_order,
      oca_order, moo_order, moc_order, loo_order, loc_order, gtd_order,
      fok_order, ioc_order, vwap_order, twap_order, iceberg_order, multi_leg,
      covered_call, cash_secured_put, protective_put, iron_condor,
      iron_butterfly, straddle, strangle, calendar_spread, diagonal_spread,
      butterfly, ratio_spread, jade_lizard, collar, cancel_order.

Cycle-end requirement:
- End each cycle with `done` and checklist fields:
  tested:[...], passed:[...], failed:[...], deferred:[...], next:[...].
- Keep `next` focused on the next 1-3 unchecked tools from the ordered list.
- Do not repeat already-passed tools in the next cycle unless needed to verify
  an execution-path side effect or data freshness issue.
"""

INDEPENDENT_MODE_CYCLE_GUIDANCE: str = """=== INDEPENDENT MODE (researcher unavailable) ===
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

CONNECTED_MODE_CYCLE_GUIDANCE: str = (
    "Start by calling briefing() to assess research status. "
    "Use quality_status() when you need the current host risk posture."
)


@dataclass(frozen=True)
class SystemPromptInputs:
    """Runtime values interpolated into :meth:`PromptConfig.build_system_prompt`."""

    trading_mode: TradingMode
    risk_per_trade: float
    max_daily_loss_pct: float
    eod_flatten_minutes: int
    max_daily_llm_cost: float
    cycle_sleep_seconds: int
    tool_smoke_mode: bool = False

    @property
    def risk_per_trade_pct(self) -> float:
        return self.risk_per_trade * 100.0


class PromptConfig(BaseModel):
    """Authoritative prompt, model, sampling, R:R, and QualityMatrix tuning."""

    # ── xAI model slugs ─────────────────────────────────────────────────────
    reasoning_model: str = Field(
        default="grok-4.3",
        description=(
            "Profitability: primary ReAct model on every cycle; stronger reasoning reduces "
            "bad entries but raises $/token — balance via QualityMatrix temperature caps."
        ),
    )
    multi_agent_model: str = Field(
        default="grok-4.20-multi-agent",
        description=(
            "Profitability: used only by research(); expensive multi-agent calls — "
            "slug choice directly affects research spend vs discovery quality."
        ),
    )

    # ── Default LLM sampling (agent baseline; matrix may override) ──────────
    llm_temperature: float = Field(
        default=0.0,
        description=(
            "Profitability: base sampling temperature; lower values reduce random "
            "entries in live mode (0 = deterministic tool routing)."
        ),
    )
    llm_seed: int = Field(
        default=42,
        description=(
            "Profitability: fixed seed for reproducible cycles when debugging "
            "loss clusters (does not affect market P&L, aids post-mortems)."
        ),
    )
    llm_max_tokens: int = Field(
        default=8192,
        description=(
            "Profitability: per-turn completion ceiling; higher allows richer "
            "plans but increases API cost on every ReAct step."
        ),
    )

    # ── Minimum reward:risk by trading mode (told to agent + sizing gates) ──
    min_rr_aggressive_paper: float = Field(
        default=float(MODE_DEFAULTS["aggressive_paper"]["rr"]),
        description=(
            "Profitability: lowest R:R bar (1.5:1) — more paper trades to learn "
            "execution; accepts lower edge per trade for exploration."
        ),
    )
    min_rr_paper: float = Field(
        default=float(MODE_DEFAULTS["paper"]["rr"]),
        description=(
            "Profitability: standard paper R:R (2:1); agent should skip setups "
            "below this unless management/hedge."
        ),
    )
    min_rr_live: float = Field(
        default=float(MODE_DEFAULTS["live"]["rr"]),
        description=(
            "Profitability: live R:R floor (2.5:1); raises entry bar so "
            "real-money trades need more upside vs stop risk."
        ),
    )

    # ── Confidence / conviction thresholds ──────────────────────────────────
    wm_default_entry_confidence: float = Field(
        default=0.85,
        description=(
            "Profitability: default WM entry confidence; higher values slow "
            "low-conviction theses from polluting prompt context."
        ),
    )
    high_conviction_confidence_floor: float = Field(
        default=0.75,
        description=(
            "Profitability: independent-mode guidance treats scores above this "
            "as warranting new risk when execution quality is strong."
        ),
    )
    idle_cash_slack_pct: float = Field(
        default=30.0,
        description=(
            "Profitability: system-prompt rule flags IDLE CASH above this % of "
            "NetLiq — forces candidate evaluation before cycle end (reduces cash drag)."
        ),
    )

    # ── QualityMatrix scoring weights (populate + policy) ───────────────────
    symbol_exec_quality_base: float = Field(
        default=0.75,
        description=(
            "Profitability: baseline symbol execution score before gap penalty; "
            "anchors per-symbol risk scaling to recent fill quality."
        ),
    )
    symbol_exec_quality_gap_coeff: float = Field(
        default=25.0,
        description=(
            "Profitability: multiplier on |avg execution_gap|; higher = faster "
            "downgrade of symbols with chronic slippage (protects edge)."
        ),
    )
    symbol_exec_quality_min: float = Field(
        default=0.05,
        description="Profitability: floor on symbol execution_quality so sizing never hits zero.",
    )
    symbol_exec_quality_max: float = Field(
        default=0.95,
        description="Profitability: cap prevents over-trusting one lucky week of fills.",
    )
    global_exec_quality_default: float = Field(
        default=0.5,
        description="Profitability: neutral book-wide execution prior when no feedback rows.",
    )
    global_exec_formula_base: float = Field(
        default=0.7,
        description="Profitability: starting point for portfolio execution score from gaps.",
    )
    global_exec_formula_gap_coeff: float = Field(
        default=20.0,
        description=(
            "Profitability: portfolio-wide gap sensitivity; tightens risk when "
            "aggregate slippage erodes simulated vs actual returns."
        ),
    )
    global_exec_min: float = Field(default=0.1, description="Profitability: minimum global execution score.")
    global_exec_max: float = Field(default=0.9, description="Profitability: maximum global execution score.")

    symbol_exec_poor_threshold: float = Field(
        default=0.35,
        description=(
            "Profitability: below this, recommended_policies caps risk_multiplier "
            "at symbol_risk_cap_multiplier — shrinks size on sloppy names."
        ),
    )
    symbol_exec_conservative_threshold: float = Field(
        default=0.4,
        description=(
            "Profitability: triggers force_conservative_reasoning per symbol; "
            "blocks aggressive entries on historically bad fills."
        ),
    )
    symbol_risk_cap_multiplier: float = Field(
        default=0.5,
        description="Profitability: max risk scale when symbol execution_quality is poor.",
    )
    global_exec_degraded_threshold: float = Field(
        default=0.35,
        description=(
            "Profitability: book-wide score triggering 'limited' posture — "
            "reduces new risk after bad execution week."
        ),
    )
    symbol_exec_bad_threshold: float = Field(
        default=0.3,
        description="Profitability: any symbol below this contributes to has_bad_exec flag.",
    )

    high_risk_entry_rm_threshold: float = Field(
        default=0.55,
        description=(
            "Profitability: plan/market/limit entries categorized as high_risk_entry "
            "when risk_multiplier below this — enables category blocks."
        ),
    )
    independent_mode_entry_rm_threshold: float = Field(
        default=0.55,
        description=(
            "Profitability: blocks new entries in independent mode when rm below "
            "this — avoids trading on stale research."
        ),
    )
    min_rm_for_new_risk: float = Field(
        default=0.40,
        description=(
            "Profitability: can_initiate_new_risk requires rm above this — "
            "hard gate on new positions when policy is defensive."
        ),
    )
    entry_scale_minimal_degraded: float = Field(
        default=0.25,
        description=(
            "Profitability: entry quantity multiplier cap in minimal/degraded "
            "quality — limits damage during bad regimes."
        ),
    )
    entry_scale_conservative: float = Field(
        default=0.6,
        description=(
            "Profitability: entry scale when force_conservative_reasoning — "
            "preserves participation but cuts size."
        ),
    )
    researcher_unavailable_rm_cap: float = Field(
        default=0.65,
        description=(
            "Profitability: max risk_multiplier when researcher drops offline — "
            "reduces size until heartbeat returns."
        ),
    )
    minimal_posture_rm_cap: float = Field(
        default=0.4,
        description="Profitability: risk cap when overall_quality is minimal.",
    )
    limited_posture_rm_cap: float = Field(
        default=0.65,
        description="Profitability: risk cap when execution quality is limited.",
    )

    matrix_default_suggested_temperature: float = Field(
        default=0.3,
        description="Profitability: full-quality LLM temperature — balanced creativity vs discipline.",
    )
    matrix_default_suggested_max_tokens: int = Field(
        default=2048,
        description="Profitability: default matrix max_tokens before quality-tier caps.",
    )
    matrix_temp_minimal: float = Field(
        default=0.15,
        description="Profitability: sampling temp when research host down — fewer impulsive entries.",
    )
    matrix_temp_limited: float = Field(
        default=0.22,
        description="Profitability: temp when execution quality is weak — cautious planning.",
    )
    matrix_temp_full: float = Field(
        default=0.3,
        description="Profitability: temp when system is healthy — normal planning bandwidth.",
    )
    llm_temp_degraded_cap: float = Field(
        default=0.05,
        description="Profitability: hard temp cap in degraded quality — near-deterministic exits-only mode.",
    )
    llm_temp_minimal_limited_cap: float = Field(
        default=0.18,
        description="Profitability: temp cap for minimal/limited overall_quality tiers.",
    )
    llm_temp_conservative_cap: float = Field(
        default=0.12,
        description="Profitability: extra cap when force_conservative_reasoning is on.",
    )
    llm_tokens_degraded_cap: int = Field(
        default=3072,
        description="Profitability: limits completion spend when quality is degraded.",
    )
    llm_tokens_limited_cap: int = Field(
        default=5500,
        description="Profitability: token ceiling for minimal/limited tiers.",
    )
    llm_top_p_conservative: float = Field(
        default=0.78,
        description="Profitability: narrows sampling when conservative — less tail-risk ideation.",
    )
    llm_top_p_normal: float = Field(
        default=0.93,
        description="Profitability: normal top_p when quality is full.",
    )
    fallback_matrix_temperature: float = Field(
        default=0.3,
        description="Profitability: operating_context fallback matrix default temperature.",
    )
    fallback_matrix_max_tokens: int = Field(
        default=8192,
        description="Profitability: fallback max_tokens when QualityMatrix import fails.",
    )
    fallback_conservative_temperature: float = Field(
        default=0.15,
        description="Profitability: fallback conservative sampling when core.quality missing.",
    )
    fallback_conservative_max_tokens: int = Field(
        default=3072,
        description="Profitability: fallback token cap in conservative fallback mode.",
    )

    # ── Mode guidance fragments (stored on config object) ─────────────────────
    mode_guidance_aggressive_paper: str = Field(
        default=MODE_GUIDANCE_AGGRESSIVE_PAPER,
        description="Profitability: tells agent to explore complex structures in stress paper.",
    )
    mode_guidance_paper: str = Field(default=MODE_GUIDANCE_PAPER, description="Profitability: balanced paper trading guidance.")
    mode_guidance_live: str = Field(
        default=MODE_GUIDANCE_LIVE,
        description="Profitability: emphasizes capital preservation and smaller size live.",
    )
    tool_smoke_prompt_body: str = Field(
        default=TOOL_SMOKE_PROMPT_BODY,
        description="Profitability: smoke-mode checklist; avoids untested tools in production.",
    )
    independent_mode_cycle_guidance: str = Field(
        default=INDEPENDENT_MODE_CYCLE_GUIDANCE,
        description="Profitability: steers agent to quality tools when research host is offline.",
    )
    connected_mode_cycle_guidance: str = Field(
        default=CONNECTED_MODE_CYCLE_GUIDANCE,
        description="Profitability: reminds agent to read briefing before acting when connected.",
    )

    def min_rr_for_mode(self, mode: TradingMode) -> float:
        """Return configured minimum R:R for a trading mode."""
        if mode == "aggressive_paper":
            return self.min_rr_aggressive_paper
        if mode == "live":
            return self.min_rr_live
        return self.min_rr_paper

    def mode_guidance(self, mode: TradingMode) -> str:
        """Return mode-specific system-prompt guidance text."""
        if mode == "aggressive_paper":
            return self.mode_guidance_aggressive_paper
        if mode == "live":
            return self.mode_guidance_live
        return self.mode_guidance_paper

    def tool_smoke_instructions(self, enabled: bool) -> str:
        """Return tool-smoke block or empty string."""
        return self.tool_smoke_prompt_body if enabled else ""

    def build_system_prompt(self, inputs: SystemPromptInputs) -> str:
        """Assemble the full trader system prompt from fragments + runtime rails."""
        return _SYSTEM_PROMPT_TEMPLATE.format(
            trading_mode=inputs.trading_mode,
            mode_description=self.mode_guidance(inputs.trading_mode),
            tool_smoke_instructions=self.tool_smoke_instructions(inputs.tool_smoke_mode),
            risk_per_trade_pct=inputs.risk_per_trade_pct,
            max_daily_loss_pct=inputs.max_daily_loss_pct,
            eod_flatten_minutes=inputs.eod_flatten_minutes,
            max_daily_llm_cost=inputs.max_daily_llm_cost,
            cycle_sleep_seconds=inputs.cycle_sleep_seconds,
        )

    def cycle_guidance_footer(self, *, independent_mode: bool) -> str:
        """Per-cycle user-prompt footer (operating context)."""
        if independent_mode:
            return self.independent_mode_cycle_guidance
        return self.connected_mode_cycle_guidance


_prompt_config_override: PromptConfig | None = None


@lru_cache(maxsize=1)
def get_prompt_config() -> PromptConfig:
    """Return the process-wide :class:`PromptConfig` singleton."""
    if _prompt_config_override is not None:
        return _prompt_config_override
    return PromptConfig()


def install_prompt_config(cfg: PromptConfig | None) -> None:
    """Pin a profile-patched instance (``None`` clears the override)."""
    global _prompt_config_override
    _prompt_config_override = cfg
    get_prompt_config.cache_clear()


def reload_prompt_config() -> PromptConfig:
    """Clear cache and return a fresh config (tests / after env refresh)."""
    install_prompt_config(None)
    return get_prompt_config()


__all__ = [
    "PromptConfig",
    "SystemPromptInputs",
    "get_prompt_config",
    "install_prompt_config",
    "reload_prompt_config",
]
