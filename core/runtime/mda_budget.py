"""
Market Data App credit monitoring and adaptive pacing for the research daemon.

MDA exposes daily credits via response headers (``X-Api-Ratelimit-*``). The
singleton :class:`data.marketdata_client.MarketDataClient` tracks the latest
values; this module turns them into cadence multipliers and fetch-skips so a
long-running daemon stays inside budget instead of hitting the circuit breaker.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from research.config import (
    MDA_CRITICAL_CREDIT_FRACTION,
    MDA_LOW_CREDIT_FRACTION,
    MDA_SKIP_SUBDAILY_FRACTION,
)

logger = logging.getLogger(__name__)


def credit_fraction(usage: dict[str, Any]) -> Optional[float]:
    """Return remaining/limit, or None if not yet known from API headers."""
    rem = usage.get("mda_credits_remaining")
    lim = usage.get("mda_credits_limit")
    if rem is None or lim is None or lim <= 0:
        return None
    return max(0.0, min(1.0, float(rem) / float(lim)))


def mda_cadence_multiplier(usage: dict[str, Any]) -> float:
    """Scale sleep between scoring rounds: 1.0 = normal, 2x / 4x when low."""
    frac = credit_fraction(usage)
    if frac is None:
        return 1.0
    if frac < MDA_CRITICAL_CREDIT_FRACTION:
        return 4.0
    if frac < MDA_LOW_CREDIT_FRACTION:
        return 2.0
    return 1.0


def should_skip_subdaily_candle_fetches(usage: dict[str, Any]) -> bool:
    """If True, scorer should not fetch 1m/5m/1h bundles (daily only)."""
    frac = credit_fraction(usage)
    if frac is None:
        return False
    return frac < MDA_SKIP_SUBDAILY_FRACTION


def persist_mda_snapshot_to_db(usage: dict[str, Any], cadence_mult: float) -> None:
    """Write latest MDA counters to ``research_config`` for external monitoring (quiet)."""
    from memory.repos.config_repo import set_research_config

    rem = usage.get("mda_credits_remaining")
    lim = usage.get("mda_credits_limit")
    rst = usage.get("mda_credits_reset_epoch")
    br = usage.get("mda_breaker_open")

    if rem is not None:
        set_research_config("mda_credits_remaining", float(rem), "mda snapshot", log=False)
    if lim is not None:
        set_research_config("mda_credits_limit", float(lim), "mda snapshot", log=False)
    if rst is not None:
        set_research_config("mda_credits_reset_epoch", float(rst), "mda snapshot", log=False)
    if br is not None:
        set_research_config("mda_breaker_open", 1.0 if br else 0.0, "mda snapshot", log=False)
    set_research_config("mda_cadence_multiplier", float(cadence_mult), "mda snapshot", log=False)


def log_mda_round_status(
    round_num: int,
    usage: dict[str, Any],
    cadence_mult: float,
    skip_subdaily: bool,
) -> None:
    """One INFO line per round when headers are known; WARN when breaker active."""
    rem = usage.get("mda_credits_remaining")
    lim = usage.get("mda_credits_limit")
    br = usage.get("mda_breaker_open")
    frac = credit_fraction(usage)

    if frac is not None and lim is not None:
        logger.info(
            "Round %d MDA: %s / %s credits (%.1f%% left), breaker=%s, "
            "cadence_mult=%.1fx, skip_subdaily=%s",
            round_num,
            f"{int(rem):,}" if rem is not None else "?",
            f"{int(lim):,}",
            100.0 * frac,
            br,
            cadence_mult,
            skip_subdaily,
        )
    elif br:
        logger.warning(
            "Round %d MDA: circuit breaker ON — credits unknown or exhausted; "
            "check logs / dashboard; next HTTP may refresh headers after reset",
            round_num,
        )
    else:
        logger.debug(
            "Round %d MDA: no rate-limit headers yet (first successful API response will populate)",
            round_num,
        )
