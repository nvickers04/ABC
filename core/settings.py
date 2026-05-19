"""Pydantic settings: .env loading, field validation, and cross-field rules.

Loaded once at import into :mod:`core.config` module-level constants.
:func:`core.config.validate_config` re-validates a runtime snapshot (supports
``monkeypatch`` in tests and CLI ``--account live`` via ``os.environ``).
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Self
from urllib.parse import urlparse

from pydantic import (
    AliasChoices,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

TradingMode = Literal["aggressive_paper", "paper", "live"]
IbkrAccountType = Literal["paper", "live"]

MODE_DEFAULTS: dict[TradingMode, dict[str, float]] = {
    "aggressive_paper": {"risk": 5.0, "rr": 1.5},
    "paper": {"risk": 1.0, "rr": 2.0},
    "live": {"risk": 0.5, "rr": 2.5},
}

_TRADING_MODES: frozenset[str] = frozenset(MODE_DEFAULTS.keys())
_POSTGRES_SCHEMES = frozenset({"postgresql", "postgres"})


def _strip_lower(v: Any) -> str:
    return str(v).strip().lower()


def _parse_bool(v: Any, *, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("0", "false", "no", "off")


_ENV_FIELD_LABELS: dict[str, str] = {
    "trading_mode": "TRADING_MODE",
    "ibkr_account_type": "IBKR_ACCOUNT_TYPE",
    "risk_per_trade": "RISK_PER_TRADE",
    "min_rr_ratio": "MIN_RR",
    "database_url": "DATABASE_URL",
    "max_daily_loss_pct": "MAX_DAILY_LOSS_PCT",
    "intraday_drawdown_pct": "INTRADAY_DRAWDOWN_PCT",
    "eod_flatten_minutes": "EOD_FLATTEN_MINUTES",
    "open_gap_guard_pct": "OPEN_GAP_GUARD_PCT",
    "open_guard_delay_minutes": "OPEN_GUARD_DELAY_MINUTES",
    "max_daily_llm_cost": "MAX_DAILY_LLM_COST",
    "max_daily_multi_agent_research_usd": "MAX_DAILY_MULTI_AGENT_RESEARCH_USD",
    "cycle_sleep_seconds": "CYCLE_SLEEP_SECONDS",
    "llm_temperature": "LLM_TEMPERATURE",
    "llm_max_tokens": "LLM_MAX_TOKENS",
    "tool_playbook_max_chars": "TOOL_PLAYBOOK_MAX_CHARS",
    "agent_tool_feedback_max_chars": "AGENT_TOOL_FEEDBACK_MAX_CHARS",
}


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Flatten a Pydantic :class:`ValidationError` into user-facing strings."""
    messages: list[str] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = str(loc[-1]) if loc else ""
        label = _ENV_FIELD_LABELS.get(field, field or "config")
        msg = err.get("msg", "invalid value")
        if str(msg).startswith("Value error"):
            detail = str(msg).split(",", 1)[-1].strip()
            messages.append(detail if field in ("trading_mode", "ibkr_account_type") else f"{label}: {detail}")
        elif field:
            messages.append(f"{label}: {msg}")
        else:
            messages.append(str(msg))
    return messages


class AppSettings(BaseSettings):
    """Application settings from environment and optional ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    trading_mode: TradingMode = Field(
        default="paper",
        validation_alias=AliasChoices("TRADING_MODE"),
    )
    ibkr_account_type: IbkrAccountType = Field(
        default="paper",
        validation_alias=AliasChoices("IBKR_ACCOUNT_TYPE"),
    )

    risk_per_trade_pct: float | None = Field(
        default=None,
        validation_alias=AliasChoices("RISK_PER_TRADE"),
        description="Per-trade risk as percent of cash (e.g. 1.0 = 1%%).",
    )
    min_rr_ratio: float | None = Field(
        default=None,
        validation_alias=AliasChoices("MIN_RR"),
    )

    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL"),
    )
    pghost: str | None = Field(default=None, validation_alias=AliasChoices("PGHOST"))
    pgport: str = Field(default="5432", validation_alias=AliasChoices("PGPORT"))
    pgdatabase: str | None = Field(default=None, validation_alias=AliasChoices("PGDATABASE"))
    pguser: str | None = Field(default=None, validation_alias=AliasChoices("PGUSER"))
    pgpassword: str | None = Field(default=None, validation_alias=AliasChoices("PGPASSWORD"))

    tool_smoke_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("TOOL_SMOKE_MODE"),
    )
    max_daily_llm_cost: float = Field(
        default=4.5,
        validation_alias=AliasChoices("MAX_DAILY_LLM_COST"),
        gt=0,
    )
    max_daily_multi_agent_research_usd: float = Field(
        default=0.75,
        validation_alias=AliasChoices("MAX_DAILY_MULTI_AGENT_RESEARCH_USD"),
        ge=0,
    )
    multi_agent_research_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("MULTI_AGENT_RESEARCH_ENABLED"),
    )
    researcher_daily_token_cap: int = Field(
        default=100_000,
        validation_alias=AliasChoices("RESEARCHER_DAILY_TOKEN_CAP"),
        gt=0,
    )
    researcher_mda_health_check_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RESEARCHER_MDA_HEALTH_CHECK_ENABLED"),
    )
    prescan_prompt_expensive_research: bool = Field(
        default=False,
        validation_alias=AliasChoices("PRESCAN_PROMPT_EXPENSIVE_RESEARCH"),
    )
    ibkr_quotes_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("IBKR_QUOTES_ENABLED"),
    )
    ibkr_quote_line_budget: int = Field(
        default=90,
        validation_alias=AliasChoices("IBKR_QUOTE_LINE_BUDGET"),
        gt=0,
    )
    trader_in_process_scorer: str = Field(
        default="auto",
        validation_alias=AliasChoices("TRADER_IN_PROCESS_SCORER"),
    )
    tool_playbook_max_chars: int = Field(
        default=1200,
        validation_alias=AliasChoices("TOOL_PLAYBOOK_MAX_CHARS"),
        gt=0,
    )
    agent_tool_feedback_max_chars: int = Field(
        default=4500,
        validation_alias=AliasChoices("AGENT_TOOL_FEEDBACK_MAX_CHARS"),
        ge=500,
    )
    briefing_min_template_trades: int = Field(
        default=5,
        validation_alias=AliasChoices("BRIEFING_MIN_TEMPLATE_TRADES"),
        gt=0,
    )
    briefing_template_leaderboard_k: int = Field(
        default=8,
        validation_alias=AliasChoices("BRIEFING_TEMPLATE_LEADERBOARD_K"),
        gt=0,
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

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _normalize_trading_mode(cls, v: Any) -> str:
        raw = _strip_lower(v) if v is not None else "paper"
        if raw not in _TRADING_MODES:
            raise ValueError(
                f"TRADING_MODE={raw!r} is not one of "
                f"{tuple(sorted(_TRADING_MODES))}"
            )
        return raw

    @field_validator("ibkr_account_type", mode="before")
    @classmethod
    def _normalize_ibkr_account(cls, v: Any) -> str:
        raw = _strip_lower(v) if v is not None else "paper"
        if raw not in ("paper", "live"):
            raise ValueError(f"IBKR_ACCOUNT_TYPE={raw!r} must be 'paper' or 'live'")
        return raw

    @field_validator("tool_smoke_mode", mode="before")
    @classmethod
    def _bool_tool_smoke(cls, v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        return str(v).strip() == "1"

    @field_validator("ibkr_quotes_enabled", mode="before")
    @classmethod
    def _ibkr_quotes(cls, v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        return str(v).strip() == "1"

    @field_validator("multi_agent_research_enabled", mode="before")
    @classmethod
    def _multi_agent_enabled(cls, v: Any) -> bool:
        if v is None:
            return True
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

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        parsed = urlparse(v)
        if parsed.scheme not in _POSTGRES_SCHEMES:
            raise ValueError(
                "DATABASE_URL must use postgresql:// or postgres:// scheme"
            )
        if not parsed.hostname:
            raise ValueError("DATABASE_URL must include a host")
        if not (parsed.path or "").lstrip("/"):
            raise ValueError("DATABASE_URL must include a database name in the path")
        return v

    @field_validator("pghost", "pgdatabase", "pguser", "pgpassword", mode="before")
    @classmethod
    def _strip_pg_fields(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

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
                "TRADING_MODE=aggressive_paper requires a paper IBKR account "
                "(IBKR_ACCOUNT_TYPE=live is not allowed)"
            )
        if self.trading_mode in ("paper", "aggressive_paper") and self.ibkr_account_type == "live":
            raise ValueError(
                f"TRADING_MODE={self.trading_mode} is incompatible with "
                "IBKR_ACCOUNT_TYPE=live; use TRADING_MODE=live or paper gateway"
            )

        self._validate_postgres_config()
        return self

    def _validate_postgres_config(self) -> None:
        """When any Postgres env is set, require a complete, valid configuration."""
        pg_parts = (self.pghost, self.pgdatabase, self.pguser, self.pgpassword)
        if self.database_url:
            return
        if not any(pg_parts):
            return
        missing = [
            name
            for name, val in (
                ("PGHOST", self.pghost),
                ("PGDATABASE", self.pgdatabase),
                ("PGUSER", self.pguser),
                ("PGPASSWORD", self.pgpassword),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                "Incomplete Postgres env: set DATABASE_URL or all of "
                f"PGHOST, PGDATABASE, PGUSER, PGPASSWORD (missing: {', '.join(missing)})"
            )
        if self.pgport and not re.fullmatch(r"\d{1,5}", str(self.pgport).strip()):
            raise ValueError(f"PGPORT={self.pgport!r} must be a numeric port")

    @property
    def risk_per_trade_fraction(self) -> float:
        """Per-trade risk as a fraction of cash (0.01 = 1%)."""
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

    def resolve_database_dsn(self) -> str:
        """Return a Postgres DSN or raise if configuration is missing."""
        if self.database_url:
            return self.database_url
        if all([self.pghost, self.pgdatabase, self.pguser, self.pgpassword]):
            return (
                f"postgresql://{self.pguser}:{self.pgpassword}"
                f"@{self.pghost}:{self.pgport}/{self.pgdatabase}"
            )
        raise RuntimeError(
            "PostgreSQL is required. Set DATABASE_URL or "
            "PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
        )


class RuntimeConfigSnapshot(BaseSettings):
    """Validate module-level / CLI-overridden values (tests use ``monkeypatch``)."""

    model_config = SettingsConfigDict(extra="forbid")

    trading_mode: TradingMode
    ibkr_account_type: IbkrAccountType
    risk_per_trade: float = Field(gt=0, le=0.5)
    min_rr_ratio: float = Field(gt=0)
    max_daily_loss_pct: float = Field(gt=0, le=100)
    intraday_drawdown_pct: float = Field(gt=0, le=100)
    eod_flatten_minutes: int = Field(gt=0)
    open_gap_guard_pct: float = Field(ge=0)
    open_guard_delay_minutes: int = Field(ge=0)
    max_daily_llm_cost: float = Field(gt=0)
    max_daily_multi_agent_research_usd: float = Field(ge=0)
    cycle_sleep_seconds: int = Field(gt=0)
    llm_temperature: float = Field(ge=0, le=2)
    llm_max_tokens: int = Field(gt=0)
    tool_playbook_max_chars: int = Field(gt=0)
    agent_tool_feedback_max_chars: int = Field(ge=500)
    max_daily_llm_noncached_prompt_text_tokens: int = Field(ge=0)
    max_daily_llm_cached_prompt_text_tokens: int = Field(ge=0)
    max_daily_llm_prompt_image_tokens: int = Field(ge=0)
    max_daily_llm_completion_tokens: int = Field(ge=0)
    max_daily_llm_reasoning_tokens: int = Field(ge=0)
    max_daily_llm_output_priced_tokens: int = Field(ge=0)
    database_url: str | None = None

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _mode(cls, v: Any) -> str:
        raw = _strip_lower(v)
        if raw not in _TRADING_MODES:
            raise ValueError(
                f"TRADING_MODE={raw!r} is not one of {tuple(sorted(_TRADING_MODES))}"
            )
        return raw

    @field_validator("ibkr_account_type", mode="before")
    @classmethod
    def _account(cls, v: Any) -> str:
        raw = _strip_lower(v)
        if raw not in ("paper", "live"):
            raise ValueError(f"IBKR_ACCOUNT_TYPE={raw!r} must be 'paper' or 'live'")
        return raw

    @field_validator("database_url", mode="before")
    @classmethod
    def _db_url(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("database_url")
    @classmethod
    def _db_url_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        parsed = urlparse(v)
        if parsed.scheme not in _POSTGRES_SCHEMES:
            raise ValueError("DATABASE_URL must use postgresql:// or postgres://")
        if not parsed.hostname:
            raise ValueError("DATABASE_URL must include a host")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> Self:
        if self.trading_mode == "live" and self.risk_per_trade > 0.02:
            raise ValueError(
                f"TRADING_MODE=live but RISK_PER_TRADE={self.risk_per_trade} > 2%; "
                "live trading with >2% per-trade risk is not permitted"
            )
        if self.trading_mode == "live" and self.ibkr_account_type != "live":
            raise ValueError(
                "TRADING_MODE=live requires IBKR_ACCOUNT_TYPE=live "
                f"(got {self.ibkr_account_type!r})"
            )
        if self.trading_mode == "aggressive_paper" and self.ibkr_account_type == "live":
            raise ValueError(
                "TRADING_MODE=aggressive_paper requires IBKR_ACCOUNT_TYPE=paper"
            )
        if self.trading_mode in ("paper", "aggressive_paper") and self.ibkr_account_type == "live":
            raise ValueError(
                f"TRADING_MODE={self.trading_mode} is incompatible with "
                "IBKR_ACCOUNT_TYPE=live"
            )
        return self


def load_app_settings() -> AppSettings:
    """Load settings from the current environment and ``.env``."""
    return AppSettings()


def effective_env(name: str, module_default: str) -> str:
    """Prefer ``os.environ`` (CLI overrides) then module-level default."""
    raw = os.getenv(name)
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip()
    return module_default
