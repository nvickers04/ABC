"""Single source for trading risk, IBKR execution, MDA pacing, and split-host heartbeat rules.

Import via :func:`get_risk_execution_config` from :mod:`core.config`, safety, MDA budget,
heartbeat, and execution helpers — do not duplicate these constants elsewhere.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Literal, Self

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TradingMode = Literal["aggressive_paper", "paper", "live"]
IbkrAccountType = Literal["paper", "live"]

MODE_DEFAULTS: dict[TradingMode, dict[str, float]] = {
    "aggressive_paper": {"risk": 5.0, "rr": 1.5},
    "paper": {"risk": 1.0, "rr": 2.0},
    "live": {"risk": 0.5, "rr": 2.5},
}

_TRADING_MODES: frozenset[str] = frozenset(MODE_DEFAULTS.keys())


def _strip_lower(v: Any) -> str:
    return str(v).strip().lower()


def _parse_bool(v: Any, *, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("0", "false", "no", "off")


class RiskExecutionConfig(BaseSettings):
    """Risk, execution, LLM spend caps, MDA pacing, and research-host heartbeat policy."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Trading mode & per-trade risk ─────────────────────────────────────────
    trading_mode: TradingMode = Field(
        default="paper",
        validation_alias=AliasChoices("TRADING_MODE"),
        description="Profitability: aggressive_paper / paper / live risk and R:R defaults.",
    )
    ibkr_account_type: IbkrAccountType = Field(
        default="paper",
        validation_alias=AliasChoices("IBKR_ACCOUNT_TYPE"),
        description="Profitability: must align with TRADING_MODE for live capital protection.",
    )
    risk_per_trade_pct: float | None = Field(
        default=None,
        validation_alias=AliasChoices("RISK_PER_TRADE"),
        description="Profitability: percent of cash per trade (1.0 = 1%%).",
    )
    min_rr_ratio: float | None = Field(
        default=None,
        validation_alias=AliasChoices("MIN_RR"),
        description="Profitability: minimum reward:risk before entries.",
    )
    live_risk_ramp_approved_fraction: float = Field(
        default=0.01,
        description="Profitability: per-trade risk after DB risk_ramp_approved (1%%).",
    )

    # ── Session safety rails (SafetyController) ───────────────────────────────
    max_daily_loss_pct: float = Field(
        default=15.0,
        validation_alias=AliasChoices("MAX_DAILY_LOSS_PCT"),
        gt=0,
        le=100,
        description="Profitability: flatten when daily NLV loss exceeds this %.",
    )
    intraday_drawdown_pct: float = Field(
        default=3.0,
        validation_alias=AliasChoices("INTRADAY_DRAWDOWN_PCT"),
        gt=0,
        le=100,
        description="Profitability: flatten on peak-to-trough session drawdown.",
    )
    eod_flatten_minutes: int = Field(
        default=5,
        validation_alias=AliasChoices("EOD_FLATTEN_MINUTES"),
        gt=0,
        description="Profitability: close positions this many minutes before the close.",
    )
    cycle_sleep_seconds: int = Field(
        default=30,
        validation_alias=AliasChoices("CYCLE_SLEEP_SECONDS"),
        gt=0,
        description="Profitability: default done-action cooldown / wake pacing.",
    )

    # ── LLM & multi-agent spend ───────────────────────────────────────────────
    max_daily_llm_cost: float = Field(
        default=4.5,
        validation_alias=AliasChoices("MAX_DAILY_LLM_COST"),
        gt=0,
        description="Profitability: halt agent when estimated daily LLM USD reaches cap.",
    )
    max_daily_multi_agent_research_usd: float = Field(
        default=0.75,
        validation_alias=AliasChoices("MAX_DAILY_MULTI_AGENT_RESEARCH_USD"),
        ge=0,
        description="Profitability: sub-cap for research() multi-agent tool.",
    )
    multi_agent_research_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("MULTI_AGENT_RESEARCH_ENABLED"),
        description="Profitability: 0 disables expensive research() calls.",
    )
    researcher_daily_token_cap: int = Field(
        default=100_000,
        validation_alias=AliasChoices("RESEARCHER_DAILY_TOKEN_CAP"),
        gt=0,
        description="Profitability: researcher host token budget per day.",
    )
    max_daily_llm_noncached_prompt_text_tokens: int = Field(
        default=350_000,
        validation_alias=AliasChoices("MAX_DAILY_LLM_NONCACHED_PROMPT_TEXT_TOKENS"),
        ge=0,
    )
    max_daily_llm_cached_prompt_text_tokens: int = Field(
        default=999_999_999,
        validation_alias=AliasChoices("MAX_DAILY_LLM_CACHED_PROMPT_TEXT_TOKENS"),
        ge=0,
    )
    max_daily_llm_prompt_image_tokens: int = Field(
        default=50_000,
        validation_alias=AliasChoices("MAX_DAILY_LLM_PROMPT_IMAGE_TOKENS"),
        ge=0,
    )
    max_daily_llm_completion_tokens: int = Field(
        default=80_000,
        validation_alias=AliasChoices("MAX_DAILY_LLM_COMPLETION_TOKENS"),
        ge=0,
    )
    max_daily_llm_reasoning_tokens: int = Field(
        default=120_000,
        validation_alias=AliasChoices("MAX_DAILY_LLM_REASONING_TOKENS"),
        ge=0,
    )
    max_daily_llm_output_priced_tokens: int = Field(
        default=150_000,
        validation_alias=AliasChoices("MAX_DAILY_LLM_OUTPUT_PRICED_TOKENS"),
        ge=0,
    )

    # ── IBKR connection & execution ─────────────────────────────────────────
    ibkr_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("IBKR_HOST"),
        description="Profitability: TWS/gateway host for order routing.",
    )
    ibkr_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("IBKR_PORT"),
        description="Profitability: explicit port override; mode defaults apply when unset.",
    )
    ibkr_paper_port: int = Field(default=7497, description="Profitability: paper TWS API port.")
    ibkr_live_port: int = Field(default=7496, description="Profitability: live TWS API port.")
    ibkr_client_id: int = Field(
        default=1,
        validation_alias=AliasChoices("IBKR_CLIENT_ID"),
        ge=0,
        description="Profitability: unique client id per simultaneous API session.",
    )
    ibkr_connect_max_attempts: int = Field(
        default=12,
        validation_alias=AliasChoices("IBKR_CONNECT_MAX_ATTEMPTS"),
        ge=1,
        le=50,
        description="Profitability: tries CLIENT_ID..CLIENT_ID+N-1 on connect.",
    )
    ibkr_account_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("IBKR_ACCOUNT_ID"),
        description="Profitability: explicit DU*/U* account when auto-detect fails.",
    )
    paper_mode_legacy: bool = Field(
        default=True,
        validation_alias=AliasChoices("PAPER_MODE"),
        description="Profitability: legacy safety toggle when TRADING_MODE unset.",
    )
    cash_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("CASH_ONLY"),
        description="Profitability: block margin/short stock paths in executor.",
    )
    ibkr_quotes_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("IBKR_QUOTES_ENABLED"),
        description="Profitability: IBKR quote line budget on trader (MDA on research host).",
    )
    ibkr_quote_line_budget: int = Field(
        default=90,
        validation_alias=AliasChoices("IBKR_QUOTE_LINE_BUDGET"),
        gt=0,
        description="Profitability: max concurrent IBKR market data lines.",
    )
    prescan_prompt_expensive_research: bool = Field(
        default=False,
        validation_alias=AliasChoices("PRESCAN_PROMPT_EXPENSIVE_RESEARCH"),
        description="Profitability: first-day pre-scan nudges research() vs cheap tools.",
    )
    researcher_mda_health_check_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RESEARCHER_MDA_HEALTH_CHECK_ENABLED"),
    )

    # ── Two-machine / heartbeat policy ──────────────────────────────────────
    trader_in_process_scorer: str = Field(
        default="auto",
        validation_alias=AliasChoices("TRADER_IN_PROCESS_SCORER"),
        description=(
            "Profitability: never = require fresh research host (no duplicate scorer); "
            "auto = in-process fallback when heartbeat stale."
        ),
    )
    heartbeat_default_stale_after_s: float = Field(
        default=180.0,
        gt=0,
        description="Profitability: floor for research host heartbeat staleness.",
    )
    heartbeat_cadence_multiplier: float = Field(
        default=3.0,
        gt=0,
        description="Profitability: stale threshold = max(floor, mult×cadence + offset).",
    )
    heartbeat_cadence_offset_s: float = Field(
        default=60.0,
        ge=0,
        description="Profitability: grace seconds added to cadence-based stale window.",
    )

    # ── MDA credit pacing (research host) ─────────────────────────────────────
    mda_soft_credit_fraction: float = Field(
        default=0.48,
        gt=0,
        lt=1,
        description="Profitability: below this remaining/limit → 1.35× sleep.",
    )
    mda_low_credit_fraction: float = Field(
        default=0.32,
        gt=0,
        lt=1,
        description="Profitability: below → 2× sleep between rounds.",
    )
    mda_critical_credit_fraction: float = Field(
        default=0.18,
        gt=0,
        lt=1,
        description="Profitability: below → 4× sleep (aggressive conservation).",
    )
    mda_skip_subdaily_fraction: float = Field(
        default=0.38,
        gt=0,
        lt=1,
        description="Profitability: skip 1m/5m/1h bundles to save credits.",
    )
    mda_max_sleep_multiplier: float = Field(
        default=8.0,
        ge=1.0,
        description="Profitability: hard cap on combined MDA sleep stretch.",
    )
    mda_cadence_mult_soft: float = Field(default=1.35, description="Profitability: soft-tier sleep multiplier.")
    mda_cadence_mult_low: float = Field(default=2.0, description="Profitability: low-tier sleep multiplier.")
    mda_cadence_mult_critical: float = Field(default=4.0, description="Profitability: critical-tier sleep multiplier.")
    mda_burn_sustainable_ratio: float = Field(
        default=1.18,
        gt=1.0,
        description="Profitability: burn pacing when spend rate exceeds sustainable runway.",
    )
    mda_burn_multiplier_cap: float = Field(default=5.0, ge=1.0)
    mda_burn_ratio_cap: float = Field(default=4.0, ge=1.0)
    mda_burn_min_dt_seconds: float = Field(default=0.5, gt=0)
    mda_runway_reset_skew_s: float = Field(default=120.0, ge=0)

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _normalize_trading_mode(cls, v: Any) -> str:
        raw = _strip_lower(v) if v is not None else "paper"
        if raw not in _TRADING_MODES:
            raise ValueError(
                f"TRADING_MODE={raw!r} is not one of {tuple(sorted(_TRADING_MODES))}"
            )
        return raw

    @field_validator("ibkr_account_type", mode="before")
    @classmethod
    def _normalize_ibkr_account(cls, v: Any) -> str:
        raw = _strip_lower(v) if v is not None else "paper"
        if raw not in ("paper", "live"):
            raise ValueError(f"IBKR_ACCOUNT_TYPE={raw!r} must be 'paper' or 'live'")
        return raw

    @field_validator("multi_agent_research_enabled", mode="before")
    @classmethod
    def _multi_agent_enabled(cls, v: Any) -> bool:
        if v is None:
            return True
        return _parse_bool(v, default=True)

    @field_validator("cash_only", "paper_mode_legacy", "ibkr_quotes_enabled", mode="before")
    @classmethod
    def _bool_flags(cls, v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, bool):
            return v
        return _parse_bool(v, default=True)

    @field_validator("researcher_mda_health_check_enabled", mode="before")
    @classmethod
    def _mda_health(cls, v: Any) -> bool:
        if v is None:
            return True
        return _parse_bool(v, default=True)

    @field_validator("prescan_prompt_expensive_research", mode="before")
    @classmethod
    def _prescan(cls, v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        return str(v).strip() == "1"

    @model_validator(mode="after")
    def _apply_mode_defaults(self) -> Self:
        defaults = MODE_DEFAULTS[self.trading_mode]
        if self.risk_per_trade_pct is None:
            self.risk_per_trade_pct = defaults["risk"]
        if self.min_rr_ratio is None:
            self.min_rr_ratio = defaults["rr"]
        return self

    @model_validator(mode="after")
    def _cross_field_rules(self) -> Self:
        risk = self.risk_per_trade_fraction
        if not (0.0 < risk <= 0.5):
            raise ValueError(
                f"RISK_PER_TRADE={self.risk_per_trade_pct}% must map to a fraction "
                f"in (0, 0.5] (got {risk:.4f})"
            )
        min_rr = self.min_rr_ratio
        if min_rr is None or min_rr <= 0:
            raise ValueError(f"MIN_RR={min_rr} must be positive")

        if self.trading_mode == "live" and risk > 0.02:
            raise ValueError(
                f"TRADING_MODE=live but RISK_PER_TRADE={self.risk_per_trade_pct}% "
                f"(>{2}% per trade) is not permitted"
            )
        if self.trading_mode == "live" and self.ibkr_account_type != "live":
            raise ValueError(
                "TRADING_MODE=live requires IBKR_ACCOUNT_TYPE=live "
                f"(got {self.ibkr_account_type!r})"
            )
        if self.trading_mode == "aggressive_paper" and self.ibkr_account_type == "live":
            raise ValueError(
                "TRADING_MODE=aggressive_paper requires a paper IBKR account"
            )
        if self.trading_mode in ("paper", "aggressive_paper") and self.ibkr_account_type == "live":
            raise ValueError(
                f"TRADING_MODE={self.trading_mode} is incompatible with "
                "IBKR_ACCOUNT_TYPE=live"
            )

        fracs = (
            self.mda_critical_credit_fraction,
            self.mda_low_credit_fraction,
            self.mda_soft_credit_fraction,
        )
        if not (fracs[0] < fracs[1] < fracs[2]):
            raise ValueError(
                "MDA credit fractions must satisfy critical < low < soft"
            )
        return self

    @property
    def risk_per_trade_fraction(self) -> float:
        assert self.risk_per_trade_pct is not None
        return float(self.risk_per_trade_pct) / 100.0

    @property
    def paper_aggressive(self) -> bool:
        return self.trading_mode == "aggressive_paper"

    @property
    def trader_in_process_scorer_never(self) -> bool:
        scorer = str(self.trader_in_process_scorer)
        return scorer.strip().lower() in (
            "never",
            "0",
            "false",
            "off",
            "remote_only",
            "no",
        )

    def resolve_ibkr_port(self) -> int:
        """Resolve TWS API port from mode, legacy PAPER_MODE, or IBKR_PORT override."""
        if self.ibkr_port is not None:
            return int(self.ibkr_port)
        if self.trading_mode == "live":
            return self.ibkr_live_port
        if self.trading_mode in ("paper", "aggressive_paper"):
            return self.ibkr_paper_port
        if self.paper_mode_legacy:
            return self.ibkr_paper_port
        return int(os.getenv("IBKR_PORT", str(self.ibkr_paper_port)))

    def heartbeat_stale_after_s(self, cadence_seconds: float) -> float:
        """Cadence-aware research host staleness threshold."""
        return max(
            self.heartbeat_default_stale_after_s,
            self.heartbeat_cadence_multiplier * float(cadence_seconds)
            + self.heartbeat_cadence_offset_s,
        )

    def mode_defaults(self, mode: TradingMode | None = None) -> dict[str, float]:
        """Return default risk/rr table row for a mode."""
        return dict(MODE_DEFAULTS[mode or self.trading_mode])


_risk_execution_config_override: RiskExecutionConfig | None = None


@lru_cache(maxsize=1)
def get_risk_execution_config() -> RiskExecutionConfig:
    """Return the process-wide :class:`RiskExecutionConfig` singleton."""
    if _risk_execution_config_override is not None:
        return _risk_execution_config_override
    return RiskExecutionConfig()


def install_risk_execution_config(cfg: RiskExecutionConfig | None) -> None:
    """Pin a profile-patched instance (``None`` clears the override)."""
    global _risk_execution_config_override
    _risk_execution_config_override = cfg
    get_risk_execution_config.cache_clear()


def reload_risk_execution_config() -> RiskExecutionConfig:
    """Clear cache and reload from current environment (CLI / tests)."""
    install_risk_execution_config(None)
    return get_risk_execution_config()


__all__ = [
    "IbkrAccountType",
    "MODE_DEFAULTS",
    "RiskExecutionConfig",
    "TradingMode",
    "get_risk_execution_config",
    "install_risk_execution_config",
    "reload_risk_execution_config",
]
