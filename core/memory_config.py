"""Single source for working memory, prompt budget, context quality, and research cache rules.

Import via :func:`get_memory_config` from runtime modules, ``memory.working_memory``,
``core.agent``, ``core.quality.quality_matrix``, and tools — do not duplicate these constants.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Symbol → sector map for portfolio concentration (state context).
_SECTOR_MAP_DEFAULT: dict[str, str] = {
    "NVDA": "tech",
    "META": "tech",
    "AMD": "tech",
    "AVGO": "tech",
    "CRWD": "software",
    "NET": "software",
    "PLTR": "software",
    "APP": "software",
    "SOFI": "fintech",
    "HOOD": "fintech",
    "DKNG": "discretionary",
    "CAVA": "discretionary",
    "LLY": "healthcare",
    "UNH": "healthcare",
    "GILD": "healthcare",
    "XOM": "energy",
    "OXY": "energy",
    "JPM": "financials",
    "GS": "financials",
    "CAT": "industrials",
    "UPS": "industrials",
    "COST": "staples",
    "WMT": "staples",
    "FCX": "materials",
    "BABA": "em",
    "SPY": "index_hedge",
    "QQQ": "index_hedge",
    "IWM": "index_hedge",
}

_WM_SECTIONS_DEFAULT: tuple[str, ...] = (
    "open_theses",
    "recent_verdicts",
    "watching_for",
    "regime_notes",
    "lessons_today",
)

_SECTION_CAPS_DEFAULT: dict[str, int] = {
    "open_theses": 8,
    "recent_verdicts": 12,
    "watching_for": 10,
    "regime_notes": 5,
    "lessons_today": 8,
}

_SECTION_DEFAULT_EXPIRY_DEFAULT: dict[str, str | int] = {
    "open_theses": "EOD",
    "recent_verdicts": 30,
    "watching_for": 60,
    "regime_notes": "EOD",
    "lessons_today": "EOD",
}

_DEFAULT_SECTION_SCORES_DEFAULT: dict[str, dict[str, Any]] = {
    "lessons_today": {"score": 0.92, "last_updated": None, "sample_size": 0},
    "open_theses": {"score": 0.70, "last_updated": None, "sample_size": 0},
    "watching_for": {"score": 0.78, "last_updated": None, "sample_size": 0},
    "regime_notes": {"score": 0.75, "last_updated": None, "sample_size": 0},
    "recent_verdicts": {"score": 0.85, "last_updated": None, "sample_size": 0},
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
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


class MemoryConfig(BaseModel):
    """Centralized memory, context, and prompt-budget knobs with P&L-oriented defaults."""

    model_config = {"frozen": True}

    # ── Working memory sections ─────────────────────────────────────────────
    wm_sections: tuple[str, ...] = Field(
        default=_WM_SECTIONS_DEFAULT,
        description="Profitability: section list defines what theses survive into prompts.",
    )
    section_caps: dict[str, int] = Field(
        default_factory=lambda: dict(_SECTION_CAPS_DEFAULT),
        description=(
            "Profitability: per-section caps limit stale narrative in prompts; "
            "lower caps = fewer tokens but less recall of prior theses."
        ),
    )
    section_default_expiry: dict[str, str | int] = Field(
        default_factory=lambda: dict(_SECTION_DEFAULT_EXPIRY_DEFAULT),
        description=(
            "Profitability: shorter expiry on verdicts/watch-fors reduces acting on "
            "stale intraday reads."
        ),
    )
    default_section_scores: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in _DEFAULT_SECTION_SCORES_DEFAULT.items()},
        description="Profitability: section trust scores bias which WM blocks the agent weights.",
    )
    wm_policy: str = Field(
        default="postgres_wins_no_merge",
        description=(
            "Profitability: on researcher reconnect, Postgres WM wins with no merge — "
            "avoids duplicate/conflicting theses that cause double risk."
        ),
    )
    wm_render_cycle_max_entries: int = Field(
        default=3,
        description=(
            "Profitability: per-cycle WM render cap; lower = cheaper prompts, "
            "higher = more thesis continuity (better entries, more tokens)."
        ),
    )
    wm_render_local_default_max_entries: int = Field(
        default=5,
        description="Profitability: local JSON WM render default; independent mode recall vs cost.",
    )
    wm_entry_text_max_chars: int = Field(
        default=400,
        description="Profitability: Postgres WM line cap; trims token spend per entry.",
    )
    wm_local_entry_text_max_chars: int = Field(
        default=300,
        description="Profitability: local fallback line cap when researcher is offline.",
    )
    wm_local_file: str = Field(
        default="data/local_working_memory.json",
        description="Profitability: offline WM path; survives restarts without Postgres theses.",
    )
    wm_completeness_independent: float = Field(
        default=0.85,
        description="Profitability: reported WM completeness in independent mode (quality gates).",
    )
    wm_completeness_connected: float = Field(
        default=0.95,
        description="Profitability: reported WM completeness when researcher + Postgres are healthy.",
    )
    wm_completeness_startup_degraded: float = Field(
        default=0.5,
        description=(
            "Profitability: startup completeness when researcher heartbeat is missing — "
            "signals reduced context until briefing/WM hydrate."
        ),
    )
    wm_completeness_limited_threshold: float = Field(
        default=0.3,
        description=(
            "Profitability: above this WM completeness, 'limited' quality tier applies "
            "with 0.65 legacy risk multiplier vs 0.4 minimal."
        ),
    )

    # ── Prompt budget / summary strategy ────────────────────────────────────
    chars_per_token: float = Field(default=4.0, description="Profitability: token estimate heuristic for budget logging.")
    truncate_marker: str = Field(
        default="\n…[trimmed]\n",
        description="Profitability: visible trim marker so the agent knows context was cut.",
    )
    truncate_min_room: int = Field(default=80, description="Profitability: minimum retained chars when truncating blocks.")
    cycle_wm_max_chars: int = Field(
        default=2400,
        description=(
            "Profitability: WM block char cap per cycle; lower saves LLM $, "
            "higher preserves thesis detail for better entries."
        ),
    )
    cycle_attention_max_chars: int = Field(
        default=1200,
        description="Profitability: attention block cap; triggers drive timely wakes on setups.",
    )
    cycle_intuition_top_n: int = Field(
        default=3,
        description="Profitability: intuition lines shown; more = richer pattern recall, more tokens.",
    )
    cycle_snapshot_max: int = Field(
        default=3,
        description="Profitability: rolling market snapshots in continuity; helps regime memory vs tokens.",
    )
    cycle_snapshot_chars: int = Field(
        default=100,
        description="Profitability: per-snapshot truncation in continuity block.",
    )
    cycle_last_summary_chars: int = Field(
        default=160,
        description="Profitability: last-cycle summary cap fed into next prompt.",
    )
    cycle_guidance_max_chars: int = Field(
        default=400,
        description="Profitability: independent/connected footer cap.",
    )
    continuity_wait_reason_max_chars: int = Field(
        default=80,
        description="Profitability: wait_reason in continuity holds agent accountable on idle cash.",
    )
    continuity_wake_reason_max_chars: int = Field(
        default=60,
        description="Profitability: wake_reason explains scorer-driven re-entry timing.",
    )
    cycle_attention_max_rows: int = Field(
        default=6,
        description="Profitability: attention rows per cycle; more rows = better trigger visibility.",
    )
    cycle_attention_max_source_chars: int = Field(
        default=48,
        description="Profitability: truncates attention source text in prompt.",
    )

    # ── State context / portfolio concentration ─────────────────────────────
    idle_cash_slack_pct: float = Field(
        default=30.0,
        description=(
            "Profitability: flags IDLE CASH in state when cash exceeds this % of NetLiq — "
            "forces candidate evaluation, reducing cash drag on edge days."
        ),
    )
    portfolio_concentration_top_n: int = Field(
        default=3,
        description="Profitability: top-N concentration line warns against single-name blowups.",
    )
    sector_map: dict[str, str] = Field(
        default_factory=lambda: dict(_SECTOR_MAP_DEFAULT),
        description="Profitability: sector exposure rollup steers diversification in state block.",
    )

    # ── Focus universe ──────────────────────────────────────────────────────
    focus_symbols_scorer_limit: int = Field(
        default=32,
        description=(
            "Profitability: max focus symbols scored every daemon round — "
            "prioritizes names the trader is engaged with."
        ),
    )
    focus_symbols_trade_universe_limit: int = Field(
        default=64,
        description="Profitability: executor allow-list union cap; blocks off-universe tickets.",
    )

    # ── Attention layer ─────────────────────────────────────────────────────
    attention_active_cap: int = Field(
        default=10,
        description="Profitability: max live triggers; cap prevents wake storms and DB churn.",
    )
    attention_render_window_s: float = Field(
        default=900.0,
        description="Profitability: show fired triggers 15m; balances recall vs prompt noise.",
    )
    attention_render_default_max_rows: int = Field(
        default=10,
        description="Profitability: default attention rows when caller omits max_rows.",
    )
    attention_render_default_max_source_chars: int = Field(
        default=80,
        description="Profitability: default source truncation in attention renderer.",
    )

    # ── Quality matrix population / retention (weights stay in PromptConfig) ─
    quality_populate_max_age_seconds: float = Field(
        default=60.0,
        description="Profitability: refresh cadence for execution-quality aggregates.",
    )
    quality_symbol_lookback_days: int = Field(
        default=7,
        description="Profitability: trade_feedback window for per-symbol execution scores.",
    )
    quality_symbol_sql_limit: int = Field(
        default=50,
        description="Profitability: max symbols in quality populate query.",
    )
    quality_matrix_enabled_threshold: float = Field(
        default=0.5,
        description="Profitability: research_config threshold to enable QualityMatrix gates.",
    )
    quality_max_recent_tools: int = Field(
        default=50,
        description="Profitability: provenance ring size for tool usage audit.",
    )
    quality_max_recent_provenance: int = Field(
        default=30,
        description="Profitability: decision provenance ring size at cycle boundaries.",
    )
    quality_session_reset_max_tools: int = Field(
        default=5,
        description="Profitability: session-boundary trim for tool usage (memory bound).",
    )
    quality_session_reset_max_provenance: int = Field(
        default=3,
        description="Profitability: session-boundary trim for provenance snapshots.",
    )

    # ── Research caches ─────────────────────────────────────────────────────
    multi_agent_research_ttl_seconds: int = Field(
        default=600,
        description=(
            "Profitability: research() multi-agent cache TTL; longer = fewer API $, "
            "staler narratives."
        ),
    )
    multi_agent_research_cache_max_entries: int = Field(
        default=20,
        description="Profitability: in-process research() cache size cap.",
    )
    research_ttl_ticker_seconds: int = Field(
        default=1800,
        description="Profitability: agent session cache TTL for ticker queries (30m).",
    )
    research_ttl_sector_seconds: int = Field(
        default=14400,
        description="Profitability: agent session cache TTL for sector queries (4h).",
    )
    research_ttl_macro_seconds: int = Field(
        default=0,
        description="Profitability: 0 = macro research expires on session boundary only.",
    )

    # ── Legacy risk multipliers (operating context, pre-matrix feedback) ────
    legacy_risk_multiplier_full: float = Field(
        default=1.0,
        description="Profitability: full context — no extra risk haircut.",
    )
    legacy_risk_multiplier_minimal: float = Field(
        default=0.4,
        description="Profitability: minimal context — strongest size reduction.",
    )
    legacy_risk_multiplier_limited: float = Field(
        default=0.65,
        description="Profitability: limited context (local WM with some completeness).",
    )
    legacy_risk_multiplier_degraded: float = Field(
        default=0.85,
        description="Profitability: researcher down but not minimal WM completeness.",
    )

    @model_validator(mode="after")
    def _apply_env_overrides(self) -> MemoryConfig:
        """Apply CYCLE_* and MEMORY_* env overrides (same names as legacy prompt_budget)."""
        object.__setattr__(
            self,
            "cycle_wm_max_chars",
            _env_int("CYCLE_WM_MAX_CHARS", self.cycle_wm_max_chars),
        )
        object.__setattr__(
            self,
            "cycle_attention_max_chars",
            _env_int("CYCLE_ATTENTION_MAX_CHARS", self.cycle_attention_max_chars),
        )
        object.__setattr__(
            self,
            "cycle_intuition_top_n",
            _env_int("CYCLE_INTUITION_TOP_N", self.cycle_intuition_top_n),
        )
        object.__setattr__(
            self,
            "cycle_snapshot_max",
            _env_int("CYCLE_SNAPSHOT_MAX", self.cycle_snapshot_max),
        )
        object.__setattr__(
            self,
            "cycle_snapshot_chars",
            _env_int("CYCLE_SNAPSHOT_CHARS", self.cycle_snapshot_chars),
        )
        object.__setattr__(
            self,
            "cycle_last_summary_chars",
            _env_int("CYCLE_LAST_SUMMARY_CHARS", self.cycle_last_summary_chars),
        )
        object.__setattr__(
            self,
            "cycle_guidance_max_chars",
            _env_int("CYCLE_GUIDANCE_MAX_CHARS", self.cycle_guidance_max_chars),
        )
        object.__setattr__(
            self,
            "multi_agent_research_ttl_seconds",
            _env_int("MEMORY_RESEARCH_TTL_SECONDS", self.multi_agent_research_ttl_seconds),
        )
        object.__setattr__(
            self,
            "focus_symbols_scorer_limit",
            _env_int("MEMORY_FOCUS_SCORER_LIMIT", self.focus_symbols_scorer_limit),
        )
        object.__setattr__(
            self,
            "focus_symbols_trade_universe_limit",
            _env_int("MEMORY_FOCUS_TRADE_LIMIT", self.focus_symbols_trade_universe_limit),
        )
        object.__setattr__(
            self,
            "idle_cash_slack_pct",
            _env_float("MEMORY_IDLE_CASH_PCT", self.idle_cash_slack_pct),
        )
        return self

    def sector_of(self, symbol: str) -> str:
        """Map a symbol to a sector bucket for concentration reporting."""
        return self.sector_map.get((symbol or "").upper(), "other")

    def legacy_risk_multiplier_for_quality(
        self,
        *,
        researcher_available: bool,
        overall_quality: str,
    ) -> float:
        """Context-only risk scale (mirrors OperatingContext.legacy_risk_multiplier)."""
        if researcher_available:
            return self.legacy_risk_multiplier_full
        if overall_quality == "minimal":
            return self.legacy_risk_multiplier_minimal
        if overall_quality == "limited":
            return self.legacy_risk_multiplier_limited
        return self.legacy_risk_multiplier_degraded

    def optimize_for_profit(self) -> None:
        """Print current settings and documented P&L impact (stdout)."""
        groups: list[tuple[str, list[str]]] = [
            (
                "Working memory",
                [
                    "wm_sections",
                    "section_caps",
                    "section_default_expiry",
                    "default_section_scores",
                    "wm_policy",
                    "wm_render_cycle_max_entries",
                    "wm_render_local_default_max_entries",
                    "wm_entry_text_max_chars",
                    "wm_local_entry_text_max_chars",
                    "wm_local_file",
                    "wm_completeness_independent",
                    "wm_completeness_connected",
                    "wm_completeness_startup_degraded",
                    "wm_completeness_limited_threshold",
                ],
            ),
            (
                "Prompt budget & summaries",
                [
                    "cycle_wm_max_chars",
                    "cycle_attention_max_chars",
                    "cycle_intuition_top_n",
                    "cycle_snapshot_max",
                    "cycle_snapshot_chars",
                    "cycle_last_summary_chars",
                    "cycle_guidance_max_chars",
                    "continuity_wait_reason_max_chars",
                    "continuity_wake_reason_max_chars",
                    "cycle_attention_max_rows",
                    "cycle_attention_max_source_chars",
                    "chars_per_token",
                    "truncate_marker",
                    "truncate_min_room",
                ],
            ),
            (
                "State & focus",
                [
                    "idle_cash_slack_pct",
                    "portfolio_concentration_top_n",
                    "sector_map",
                    "focus_symbols_scorer_limit",
                    "focus_symbols_trade_universe_limit",
                ],
            ),
            (
                "Attention",
                [
                    "attention_active_cap",
                    "attention_render_window_s",
                    "attention_render_default_max_rows",
                    "attention_render_default_max_source_chars",
                ],
            ),
            (
                "Quality matrix retention",
                [
                    "quality_populate_max_age_seconds",
                    "quality_symbol_lookback_days",
                    "quality_symbol_sql_limit",
                    "quality_matrix_enabled_threshold",
                    "quality_max_recent_tools",
                    "quality_max_recent_provenance",
                    "quality_session_reset_max_tools",
                    "quality_session_reset_max_provenance",
                ],
            ),
            (
                "Research cache",
                [
                    "multi_agent_research_ttl_seconds",
                    "multi_agent_research_cache_max_entries",
                    "research_ttl_ticker_seconds",
                    "research_ttl_sector_seconds",
                    "research_ttl_macro_seconds",
                ],
            ),
            (
                "Legacy risk (independent mode)",
                [
                    "legacy_risk_multiplier_full",
                    "legacy_risk_multiplier_minimal",
                    "legacy_risk_multiplier_limited",
                    "legacy_risk_multiplier_degraded",
                ],
            ),
        ]
        print("=== MemoryConfig — settings and P&L impact ===\n")
        for title, keys in groups:
            print(f"## {title}")
            for key in keys:
                val = getattr(self, key)
                if key == "sector_map":
                    val = f"<{len(self.sector_map)} symbols>"
                elif key in ("section_caps", "section_default_expiry", "default_section_scores"):
                    val = f"<{len(getattr(self, key))} sections>"
                field = self.model_fields[key]
                impact = (field.description or "").replace("Profitability: ", "").strip()
                print(f"  {key} = {val!r}")
                if impact:
                    print(f"    → P&L: {impact}")
            print()
        print(
            "Tip: override cycle caps via CYCLE_WM_MAX_CHARS, CYCLE_ATTENTION_MAX_CHARS, etc.; "
            "MEMORY_RESEARCH_TTL_SECONDS, MEMORY_FOCUS_* , MEMORY_IDLE_CASH_PCT."
        )


_memory_config_override: MemoryConfig | None = None


@lru_cache(maxsize=1)
def get_memory_config() -> MemoryConfig:
    """Return the process-wide :class:`MemoryConfig` singleton."""
    if _memory_config_override is not None:
        return _memory_config_override
    return MemoryConfig()


def install_memory_config(cfg: MemoryConfig | None) -> None:
    """Pin a profile-patched instance (``None`` clears the override)."""
    global _memory_config_override
    _memory_config_override = cfg
    get_memory_config.cache_clear()


def reload_memory_config() -> MemoryConfig:
    """Clear cache and return a fresh config (tests / after env refresh)."""
    install_memory_config(None)
    return get_memory_config()


__all__ = [
    "MemoryConfig",
    "get_memory_config",
    "install_memory_config",
    "reload_memory_config",
]
