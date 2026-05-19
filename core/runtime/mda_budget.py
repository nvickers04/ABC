"""
Market Data App credit monitoring and adaptive pacing for the research host.

MDA exposes daily credits via response headers (``X-Api-Ratelimit-*``). The
singleton :class:`data.marketdata_client.MarketDataClient` tracks the latest
values; this module turns them into cadence multipliers and fetch-skips so a
long-running daemon stays inside budget instead of hitting the circuit breaker.

Beyond simple ``remaining / limit`` tiers, we also use:

* **Runway** — ``X-Api-Ratelimit-Reset`` (Unix epoch when the bucket refills) vs
  wall-clock: if few hours remain but credits are already low, stretch sleeps
  earlier than fraction-only logic would.
* **Burn rate** — compare credits lost per wall second since the last round to
  ``remaining / seconds_until_reset`` (sustainable spend). If we're spending
  faster than that runway, add extra delay even while ``remaining`` looks OK.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from research.config import (
    MDA_CRITICAL_CREDIT_FRACTION,
    MDA_LOW_CREDIT_FRACTION,
    MDA_MAX_SLEEP_MULTIPLIER,
    MDA_SKIP_SUBDAILY_FRACTION,
    MDA_SOFT_CREDIT_FRACTION,
)

logger = logging.getLogger(__name__)

# Last observed credits + wall time (for burn-rate vs runway).
_prev_mda_rem: Optional[float] = None
_prev_mda_ts: float = 0.0


def credit_fraction(usage: dict[str, Any]) -> Optional[float]:
    """Return remaining/limit, or None if not yet known from API headers."""
    rem = usage.get("mda_credits_remaining")
    lim = usage.get("mda_credits_limit")
    if rem is None or lim is None or lim <= 0:
        return None
    return max(0.0, min(1.0, float(rem) / float(lim)))


def seconds_until_mda_reset(usage: dict[str, Any]) -> Optional[float]:
    """Seconds until quota reset, from ``mda_credits_reset_epoch``, or None."""
    r = usage.get("mda_credits_reset_epoch")
    if r is None:
        return None
    try:
        rt = float(r)
    except (TypeError, ValueError):
        return None
    now = time.time()
    if rt > 1e12:  # ms epoch (defensive)
        rt /= 1000.0
    diff = rt - now
    if diff >= -120.0:  # allow small clock/API skew
        return max(0.0, diff)
    # Some APIs send a short offset; treat small positive values as seconds left.
    if 0.0 < rt <= 172800.0:
        return rt
    return None


def mda_cadence_multiplier(usage: dict[str, Any]) -> float:
    """Scale sleep from **balance only** (remaining / limit): 1.0 / 1.35 / 2 / 4."""
    frac = credit_fraction(usage)
    if frac is None:
        return 1.0
    if frac < MDA_CRITICAL_CREDIT_FRACTION:
        return 4.0
    if frac < MDA_LOW_CREDIT_FRACTION:
        return 2.0
    if frac < MDA_SOFT_CREDIT_FRACTION:
        return 1.35
    return 1.0


def mda_runway_multiplier(usage: dict[str, Any]) -> float:
    """Extra stretch when little time remains before reset but credits are already low."""
    sec = seconds_until_mda_reset(usage)
    frac = credit_fraction(usage)
    if sec is None or frac is None:
        return 1.0
    hours = sec / 3600.0
    m = 1.0
    if hours < 1.0 and frac < 0.28:
        m = max(m, 2.0)
    elif hours < 3.0 and frac < 0.38:
        m = max(m, 1.6)
    elif hours < 8.0 and frac < 0.48:
        m = max(m, 1.3)
    elif hours < 14.0 and frac < 0.55:
        m = max(m, 1.15)
    return m


def _mda_burn_multiplier_and_note(usage: dict[str, Any]) -> tuple[float, Optional[str]]:
    """Compare recent spend rate to sustainable ``remaining / seconds_until_reset``."""
    global _prev_mda_rem, _prev_mda_ts
    now = time.time()
    rem_raw = usage.get("mda_credits_remaining")
    note: Optional[str] = None
    mult = 1.0

    if not isinstance(rem_raw, (int, float)) or rem_raw < 0:
        return mult, note

    rem = float(rem_raw)
    if _prev_mda_rem is not None and _prev_mda_ts > 0.0:
        spent = _prev_mda_rem - rem
        dt = now - _prev_mda_ts
        if dt > 0.5 and spent > 0.0:
            cps = spent / dt
            sec = seconds_until_mda_reset(usage)
            if sec is not None and sec > 120.0:
                sustainable = rem / sec
                if sustainable > 0 and cps > sustainable * 1.18:
                    ratio = min(4.0, cps / sustainable)
                    mult = min(5.0, 1.0 + 0.35 * (ratio - 1.0))
                    note = (
                        f"MDA burn {cps:.0f} cr/s vs runway budget {sustainable:.0f} cr/s "
                        f"({sec / 3600:.1f}h to reset)"
                    )
    _prev_mda_rem = rem
    _prev_mda_ts = now
    return mult, note


def mda_total_sleep_multiplier(usage: dict[str, Any]) -> tuple[float, Optional[str]]:
    """Combined multiplier for inter-round sleep (balance × runway × burn), capped."""
    bal = mda_cadence_multiplier(usage)
    rwy = mda_runway_multiplier(usage)
    burn, note = _mda_burn_multiplier_and_note(usage)
    total = max(1.0, min(float(MDA_MAX_SLEEP_MULTIPLIER), bal * rwy * burn))
    return total, note


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
    sec = seconds_until_mda_reset(usage)
    if sec is not None:
        set_research_config("mda_seconds_until_reset", float(sec), "mda snapshot", log=False)


def log_mda_round_status(
    round_num: int,
    usage: dict[str, Any],
    cadence_mult: float,
    skip_subdaily: bool,
    burn_note: Optional[str] = None,
) -> None:
    """One INFO line per round when headers are known; WARN when breaker active."""
    rem = usage.get("mda_credits_remaining")
    lim = usage.get("mda_credits_limit")
    br = usage.get("mda_breaker_open")
    frac = credit_fraction(usage)
    sec = seconds_until_mda_reset(usage)

    if frac is not None and lim is not None:
        eta = ""
        if sec is not None:
            if sec <= 0:
                eta = ", reset_imminent"
            elif sec < 86400:
                eta = f", ~{sec / 3600:.1f}h_to_reset"
            else:
                eta = f", ~{sec / 3600:.0f}h_to_reset"
        logger.info(
            "Round %d MDA: %s / %s credits (%.1f%% left)%s, breaker=%s, "
            "sleep_mult=%.2fx, skip_subdaily=%s",
            round_num,
            f"{int(rem):,}" if rem is not None else "?",
            f"{int(lim):,}",
            100.0 * frac,
            eta,
            br,
            cadence_mult,
            skip_subdaily,
        )
        if burn_note:
            logger.info("Round %d MDA pacing: %s", round_num, burn_note)
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
