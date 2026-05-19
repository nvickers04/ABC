"""

Core Configuration — module-level constants + .env via Pydantic.



TRADING_MODE and risk/execution knobs live in :mod:`core.risk_execution_config`.

Database and non-risk app toggles live in :mod:`core.settings.AppSettings`.

"""



from __future__ import annotations



import os

from typing import Literal, cast



# ── Load .env before Pydantic settings (graceful fallback) ───────────────────

try:

    from dotenv import load_dotenv



    load_dotenv()

except ImportError:

    pass



from pydantic import ValidationError



from core.prompt_config import SystemPromptInputs, get_prompt_config

from core.risk_execution_config import (

    MODE_DEFAULTS,

    TradingMode,

    get_risk_execution_config,

    reload_risk_execution_config,

)

from core.settings import (

    AppSettings,

    IbkrAccountType,

    RuntimeConfigSnapshot,

    effective_env,

    format_validation_errors,

    load_app_settings,

)



__all__ = [

    "ConfigError",

    "TradingMode",

    "MODE_DEFAULTS",

    "assert_config_valid",

    "get_database_dsn",

    "refresh_trading_identity_from_environ",

    "validate_config",

]



_risk = get_risk_execution_config()

_settings: AppSettings = load_app_settings()



# ── Risk & execution (single source: RiskExecutionConfig) ─────────────────

TRADING_MODE: TradingMode = _risk.trading_mode

IBKR_ACCOUNT_TYPE: Literal["paper", "live"] = _risk.ibkr_account_type

PAPER_AGGRESSIVE: bool = _risk.paper_aggressive



RISK_PER_TRADE: float = _risk.risk_per_trade_fraction

MIN_RR_RATIO: float = float(_risk.min_rr_ratio)  # type: ignore[arg-type]

MAX_RISK_PER_TRADE = RISK_PER_TRADE



MAX_DAILY_LOSS_PCT: float = _risk.max_daily_loss_pct

INTRADAY_DRAWDOWN_PCT: float = _risk.intraday_drawdown_pct

EOD_FLATTEN_MINUTES: int = _risk.eod_flatten_minutes

CYCLE_SLEEP_SECONDS: int = _risk.cycle_sleep_seconds



MAX_DAILY_LLM_COST: float = _risk.max_daily_llm_cost

MAX_DAILY_MULTI_AGENT_RESEARCH_USD: float = _risk.max_daily_multi_agent_research_usd

MULTI_AGENT_RESEARCH_ENABLED: bool = _risk.multi_agent_research_enabled

RESEARCHER_DAILY_TOKEN_CAP: int = _risk.researcher_daily_token_cap

RESEARCHER_MDA_HEALTH_CHECK_ENABLED: bool = _risk.researcher_mda_health_check_enabled

PRESCAN_PROMPT_EXPENSIVE_RESEARCH: bool = _risk.prescan_prompt_expensive_research

IBKR_QUOTES_ENABLED: bool = _risk.ibkr_quotes_enabled

IBKR_QUOTE_LINE_BUDGET: int = _risk.ibkr_quote_line_budget

TRADER_IN_PROCESS_SCORER_NEVER: bool = _risk.trader_in_process_scorer_never

CASH_ONLY: bool = _risk.cash_only



MAX_DAILY_LLM_NONCACHED_PROMPT_TEXT_TOKENS: int = _risk.max_daily_llm_noncached_prompt_text_tokens

MAX_DAILY_LLM_CACHED_PROMPT_TEXT_TOKENS: int = _risk.max_daily_llm_cached_prompt_text_tokens

MAX_DAILY_LLM_PROMPT_IMAGE_TOKENS: int = _risk.max_daily_llm_prompt_image_tokens

MAX_DAILY_LLM_COMPLETION_TOKENS: int = _risk.max_daily_llm_completion_tokens

MAX_DAILY_LLM_REASONING_TOKENS: int = _risk.max_daily_llm_reasoning_tokens

MAX_DAILY_LLM_OUTPUT_PRICED_TOKENS: int = _risk.max_daily_llm_output_priced_tokens



# IBKR (re-export for execution layer)

IBKR_HOST: str = _risk.ibkr_host

IBKR_PORT: int = _risk.resolve_ibkr_port()

IBKR_CLIENT_ID: int = _risk.ibkr_client_id

IBKR_CONNECT_MAX_ATTEMPTS: int = _risk.ibkr_connect_max_attempts

IBKR_ACCOUNT_ID: str | None = _risk.ibkr_account_id

PAPER_MODE: bool = _risk.paper_mode_legacy



# Gap guard (from loop_config; re-exported for validate_config / settings parity)

def _loop_gap_exports() -> tuple[float, int]:

    from core.loop_config import get_loop_config



    lc = get_loop_config()

    return lc.gap_guard_spy_move_pct, lc.gap_guard_delay_minutes





OPEN_GAP_GUARD_PCT, OPEN_GUARD_DELAY_MINUTES = _loop_gap_exports()



# ── App / DB settings (non-risk) ────────────────────────────────────────────

DATABASE_URL: str | None = _settings.database_url

TOOL_SMOKE_MODE: bool = _settings.tool_smoke_mode

TOOL_PLAYBOOK_MAX_CHARS: int = _settings.tool_playbook_max_chars

AGENT_TOOL_FEEDBACK_MAX_CHARS: int = _settings.agent_tool_feedback_max_chars

BRIEFING_MIN_TEMPLATE_TRADES: int = _settings.briefing_min_template_trades

BRIEFING_TEMPLATE_LEADERBOARD_K: int = _settings.briefing_template_leaderboard_k





def get_effective_risk_per_trade() -> float:

    """Return RISK_PER_TRADE with possible DB-driven ramp-up for live mode."""

    if TRADING_MODE != "live":

        return RISK_PER_TRADE

    try:

        from memory import get_research_config



        approved = get_research_config("risk_ramp_approved", 0.0)

        if approved >= 1.0:

            return get_risk_execution_config().live_risk_ramp_approved_fraction

    except Exception as e:

        import logging as _logging



        _logging.getLogger(__name__).debug("Risk ramp lookup failed: %s", e)

    return RISK_PER_TRADE





def _system_prompt_inputs() -> SystemPromptInputs:

    return SystemPromptInputs(

        trading_mode=TRADING_MODE,

        risk_per_trade=RISK_PER_TRADE,

        max_daily_loss_pct=MAX_DAILY_LOSS_PCT,

        eod_flatten_minutes=EOD_FLATTEN_MINUTES,

        max_daily_llm_cost=MAX_DAILY_LLM_COST,

        cycle_sleep_seconds=CYCLE_SLEEP_SECONDS,

        tool_smoke_mode=TOOL_SMOKE_MODE,

    )





def rebuild_prompt_exports() -> None:

    global SYSTEM_PROMPT, MODE_DESCRIPTION, LLM_TEMPERATURE, LLM_SEED, LLM_MAX_TOKENS



    pc = get_prompt_config()

    LLM_TEMPERATURE = pc.llm_temperature

    LLM_SEED = pc.llm_seed

    LLM_MAX_TOKENS = pc.llm_max_tokens

    MODE_DESCRIPTION = pc.mode_guidance(TRADING_MODE)

    SYSTEM_PROMPT = pc.build_system_prompt(_system_prompt_inputs())





LLM_TEMPERATURE: float

LLM_SEED: int

LLM_MAX_TOKENS: int

MODE_DESCRIPTION: str

SYSTEM_PROMPT: str



rebuild_prompt_exports()



TOOL_SMOKE_INSTRUCTIONS = get_prompt_config().tool_smoke_instructions(TOOL_SMOKE_MODE)





class ConfigError(ValueError):

    """Raised when ``core.config`` is misconfigured at startup."""





def get_database_dsn() -> str:

    return _settings.resolve_database_dsn()





def _sync_risk_module_exports(risk: object) -> None:

    """Refresh module-level risk aliases after env/CLI reload."""

    global TRADING_MODE, IBKR_ACCOUNT_TYPE, PAPER_AGGRESSIVE, RISK_PER_TRADE, MIN_RR_RATIO

    global MAX_RISK_PER_TRADE, MAX_DAILY_LOSS_PCT, INTRADAY_DRAWDOWN_PCT, EOD_FLATTEN_MINUTES

    global CYCLE_SLEEP_SECONDS, MAX_DAILY_LLM_COST, MAX_DAILY_MULTI_AGENT_RESEARCH_USD

    global MULTI_AGENT_RESEARCH_ENABLED, RESEARCHER_DAILY_TOKEN_CAP

    global RESEARCHER_MDA_HEALTH_CHECK_ENABLED, PRESCAN_PROMPT_EXPENSIVE_RESEARCH

    global IBKR_QUOTES_ENABLED, IBKR_QUOTE_LINE_BUDGET, TRADER_IN_PROCESS_SCORER_NEVER, CASH_ONLY

    global MAX_DAILY_LLM_NONCACHED_PROMPT_TEXT_TOKENS, MAX_DAILY_LLM_CACHED_PROMPT_TEXT_TOKENS

    global MAX_DAILY_LLM_PROMPT_IMAGE_TOKENS, MAX_DAILY_LLM_COMPLETION_TOKENS

    global MAX_DAILY_LLM_REASONING_TOKENS, MAX_DAILY_LLM_OUTPUT_PRICED_TOKENS

    global IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_CONNECT_MAX_ATTEMPTS, IBKR_ACCOUNT_ID, PAPER_MODE



    from core.risk_execution_config import RiskExecutionConfig



    r = cast(RiskExecutionConfig, risk)

    TRADING_MODE = r.trading_mode

    IBKR_ACCOUNT_TYPE = r.ibkr_account_type

    PAPER_AGGRESSIVE = r.paper_aggressive

    RISK_PER_TRADE = r.risk_per_trade_fraction

    MIN_RR_RATIO = float(r.min_rr_ratio)  # type: ignore[arg-type]

    MAX_RISK_PER_TRADE = RISK_PER_TRADE

    MAX_DAILY_LOSS_PCT = r.max_daily_loss_pct

    INTRADAY_DRAWDOWN_PCT = r.intraday_drawdown_pct

    EOD_FLATTEN_MINUTES = r.eod_flatten_minutes

    CYCLE_SLEEP_SECONDS = r.cycle_sleep_seconds

    MAX_DAILY_LLM_COST = r.max_daily_llm_cost

    MAX_DAILY_MULTI_AGENT_RESEARCH_USD = r.max_daily_multi_agent_research_usd

    MULTI_AGENT_RESEARCH_ENABLED = r.multi_agent_research_enabled

    RESEARCHER_DAILY_TOKEN_CAP = r.researcher_daily_token_cap

    RESEARCHER_MDA_HEALTH_CHECK_ENABLED = r.researcher_mda_health_check_enabled

    PRESCAN_PROMPT_EXPENSIVE_RESEARCH = r.prescan_prompt_expensive_research

    IBKR_QUOTES_ENABLED = r.ibkr_quotes_enabled

    IBKR_QUOTE_LINE_BUDGET = r.ibkr_quote_line_budget

    TRADER_IN_PROCESS_SCORER_NEVER = r.trader_in_process_scorer_never

    CASH_ONLY = r.cash_only

    MAX_DAILY_LLM_NONCACHED_PROMPT_TEXT_TOKENS = r.max_daily_llm_noncached_prompt_text_tokens

    MAX_DAILY_LLM_CACHED_PROMPT_TEXT_TOKENS = r.max_daily_llm_cached_prompt_text_tokens

    MAX_DAILY_LLM_PROMPT_IMAGE_TOKENS = r.max_daily_llm_prompt_image_tokens

    MAX_DAILY_LLM_COMPLETION_TOKENS = r.max_daily_llm_completion_tokens

    MAX_DAILY_LLM_REASONING_TOKENS = r.max_daily_llm_reasoning_tokens

    MAX_DAILY_LLM_OUTPUT_PRICED_TOKENS = r.max_daily_llm_output_priced_tokens

    IBKR_HOST = r.ibkr_host

    IBKR_PORT = r.resolve_ibkr_port()

    IBKR_CLIENT_ID = r.ibkr_client_id

    IBKR_CONNECT_MAX_ATTEMPTS = r.ibkr_connect_max_attempts

    IBKR_ACCOUNT_ID = r.ibkr_account_id

    PAPER_MODE = r.paper_mode_legacy





def refresh_trading_identity_from_environ() -> None:

    """Sync trading/risk module constants after CLI sets ``os.environ``."""

    from core.central_profit_config import reload_profit_config

    reload_profit_config()





def _runtime_database_url() -> str | None:

    raw = os.getenv("DATABASE_URL")

    if raw is not None and str(raw).strip():

        return str(raw).strip()

    return DATABASE_URL





def validate_config() -> list[str]:

    errors: list[str] = []

    try:

        RuntimeConfigSnapshot(

            trading_mode=cast(TradingMode, effective_env("TRADING_MODE", TRADING_MODE)),

            ibkr_account_type=cast(

                IbkrAccountType,

                effective_env("IBKR_ACCOUNT_TYPE", IBKR_ACCOUNT_TYPE),

            ),

            risk_per_trade=RISK_PER_TRADE,

            min_rr_ratio=MIN_RR_RATIO,

            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,

            intraday_drawdown_pct=INTRADAY_DRAWDOWN_PCT,

            eod_flatten_minutes=EOD_FLATTEN_MINUTES,

            open_gap_guard_pct=OPEN_GAP_GUARD_PCT,

            open_guard_delay_minutes=OPEN_GUARD_DELAY_MINUTES,

            max_daily_llm_cost=MAX_DAILY_LLM_COST,

            max_daily_multi_agent_research_usd=MAX_DAILY_MULTI_AGENT_RESEARCH_USD,

            cycle_sleep_seconds=CYCLE_SLEEP_SECONDS,

            llm_temperature=LLM_TEMPERATURE,

            llm_max_tokens=LLM_MAX_TOKENS,

            tool_playbook_max_chars=TOOL_PLAYBOOK_MAX_CHARS,

            agent_tool_feedback_max_chars=AGENT_TOOL_FEEDBACK_MAX_CHARS,

            max_daily_llm_noncached_prompt_text_tokens=MAX_DAILY_LLM_NONCACHED_PROMPT_TEXT_TOKENS,

            max_daily_llm_cached_prompt_text_tokens=MAX_DAILY_LLM_CACHED_PROMPT_TEXT_TOKENS,

            max_daily_llm_prompt_image_tokens=MAX_DAILY_LLM_PROMPT_IMAGE_TOKENS,

            max_daily_llm_completion_tokens=MAX_DAILY_LLM_COMPLETION_TOKENS,

            max_daily_llm_reasoning_tokens=MAX_DAILY_LLM_REASONING_TOKENS,

            max_daily_llm_output_priced_tokens=MAX_DAILY_LLM_OUTPUT_PRICED_TOKENS,

            database_url=_runtime_database_url(),

        )

    except ValidationError as exc:

        for msg in format_validation_errors(exc):

            if msg not in errors:

                errors.append(msg)



    return errors





def assert_config_valid() -> None:

    errors = validate_config()

    if errors:

        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


