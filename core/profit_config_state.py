"""Active ProfitConfig context, validation helpers, and error logging."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from datetime import date, datetime, timedelta
from typing import Any

from core.profit_profiles import PROFIT_PROFILE_ENV

# Default ~1.4 years calendar; override via ABC_MAX_BACKTEST_CALENDAR_DAYS
DEFAULT_MAX_BACKTEST_CALENDAR_DAYS = 400

_active_profile_label: ContextVar[str | None] = ContextVar("profit_profile_label", default=None)


class ProfitConfigError(Exception):
    """Base error for profitability config / simulation / logging paths."""


class InvalidProfitProfileError(ProfitConfigError, ValueError):
    """Unknown or malformed profit profile name."""


class BacktestWindowError(ProfitConfigError, ValueError):
    """Backtest date range invalid or too large."""


class SimulationDataError(ProfitConfigError):
    """Historical market data unavailable for simulation."""


def max_backtest_calendar_days() -> int:
    raw = os.getenv("ABC_MAX_BACKTEST_CALENDAR_DAYS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_BACKTEST_CALENDAR_DAYS


def set_active_profile_label(label: str | None) -> None:
    _active_profile_label.set(label)


def get_active_profile_label() -> str:
    label = _active_profile_label.get()
    if label:
        return label
    return os.getenv(PROFIT_PROFILE_ENV, "balanced").strip().lower() or "balanced"


def resolve_profile_label(
    *,
    candidate_id: str | None = None,
    base_profile: str | None = None,
) -> str:
    if candidate_id:
        return candidate_id
    if base_profile:
        return base_profile
    return get_active_profile_label()


def validate_backtest_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """Parse and validate inclusive YYYY-MM-DD backtest bounds."""
    try:
        start = datetime.strptime(start_date.strip(), "%Y-%m-%d").date()
        end = datetime.strptime(end_date.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise BacktestWindowError(
            f"Invalid backtest dates: start={start_date!r} end={end_date!r} (use YYYY-MM-DD)"
        ) from exc
    if end < start:
        raise BacktestWindowError(f"end_date {end_date} is before start_date {start_date}")
    span = (end - start).days + 1
    max_days = max_backtest_calendar_days()
    if span > max_days:
        raise BacktestWindowError(
            f"Backtest window {span} calendar days exceeds limit {max_days} "
            f"(set ABC_MAX_BACKTEST_CALENDAR_DAYS to raise)"
        )
    return start.isoformat(), end.isoformat()


def summarize_profit_config(cfg: Any) -> dict[str, Any]:
    """Compact dict for structured logs (no secrets)."""
    try:
        from core.profit_cycle_logger import snapshot_profit_config

        return snapshot_profit_config(cfg)
    except Exception:
        return {
            "profit_profile": get_active_profile_label(),
            "error": "snapshot_profit_config failed",
        }


def log_active_profit_config(
    logger: logging.Logger,
    message: str,
    *,
    cfg: Any | None = None,
    level: int = logging.ERROR,
    exc: BaseException | None = None,
    **extra: Any,
) -> None:
    """Log message plus active ProfitConfig snapshot for post-mortems."""
    if cfg is None:
        try:
            from core.central_profit_config import get_profit_config

            cfg = get_profit_config()
        except Exception:
            cfg = None
    snap = summarize_profit_config(cfg) if cfg is not None else {"profit_profile": get_active_profile_label()}
    payload = {
        "profit_config": snap,
        "profile_label": get_active_profile_label(),
        **extra,
    }
    if exc is not None:
        logger.log(level, "%s: %s | context=%s", message, exc, payload, exc_info=exc)
    else:
        logger.log(level, "%s | context=%s", message, payload)


def safe_normalize_profit_profile(profile_name: str) -> str:
    """Normalize built-in or evolved profile; raise :class:`InvalidProfitProfileError` on failure."""
    from core.profit_profiles import normalize_profit_profile

    try:
        return str(normalize_profit_profile(profile_name))
    except ValueError as exc:
        raise InvalidProfitProfileError(str(exc)) from exc
