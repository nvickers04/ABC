"""In-memory cache for composed :class:`ProfitConfig` per built-in profile name.

Used by grid search / ``build_candidate_grid`` to avoid repeated ``build_profit_config``
reloads. Invalidates when profitability-related environment variables change (fingerprint)
or when :func:`clear_profit_profile_cache` is called.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.profit_profiles import PROFIT_PROFILE_ENV

if TYPE_CHECKING:
    from core.central_profit_config import ComposedProfitConfig

logger = logging.getLogger(__name__)

# Env keys that affect composed ProfitConfig (excluding PROFIT_PROFILE — part of cache key).
_FINGERPRINT_ENV_KEYS: tuple[str, ...] = (
    "TRADING_MODE",
    "IBKR_ACCOUNT_TYPE",
    "RISK_PER_TRADE",
    "MIN_RR",
    "MAX_DAILY_LOSS_PCT",
    "INTRADAY_DRAWDOWN_PCT",
    "EOD_FLATTEN_MINUTES",
    "CYCLE_SLEEP_SECONDS",
    "MAX_DAILY_LLM_COST",
    "MAX_DAILY_MULTI_AGENT_RESEARCH_USD",
    "MULTI_AGENT_RESEARCH_ENABLED",
    "CASH_ONLY",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "CYCLE_WM_MAX_CHARS",
    "CYCLE_ATTENTION_MAX_CHARS",
    "TOOL_REGISTRY_DISABLE",
    "TOOL_REGISTRY_WEIGHTS",
)


@dataclass
class ProfitProfileCacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "total": self.total}


_profile_cache: dict[tuple[str, str, bool], ComposedProfitConfig] = {}
_stats = ProfitProfileCacheStats()
_last_fingerprint: str | None = None


def _env_fingerprint() -> str:
    """Stable signature of env knobs shared across profile builds (not PROFIT_PROFILE)."""
    parts = [f"{k}={os.environ.get(k, '')!r}" for k in _FINGERPRINT_ENV_KEYS]
    return "|".join(parts)


def _maybe_invalidate_on_fingerprint_change() -> None:
    global _last_fingerprint
    fp = _env_fingerprint()
    if _last_fingerprint is not None and fp != _last_fingerprint:
        logger.debug(
            "profit profile cache: env fingerprint changed, clearing %d entries",
            len(_profile_cache),
        )
        _profile_cache.clear()
    _last_fingerprint = fp


def clear_profit_profile_cache() -> None:
    """Drop all cached profiles and reset hit/miss stats."""
    global _last_fingerprint
    _profile_cache.clear()
    _stats.hits = 0
    _stats.misses = 0
    _last_fingerprint = None
    logger.debug("profit profile cache cleared")


def get_profit_profile_cache_stats() -> ProfitProfileCacheStats:
    """Return a snapshot of cache hit/miss counters."""
    return ProfitProfileCacheStats(hits=_stats.hits, misses=_stats.misses)


def load_cached_profit_profile(
    profile: str,
    *,
    dotenv: bool = False,
) -> ComposedProfitConfig:
    """Load composed config for ``profile`` (cache hit avoids rebuild)."""
    from core.central_profit_config import build_profit_config
    from core.profit_profiles import normalize_profit_profile

    _maybe_invalidate_on_fingerprint_change()

    if dotenv:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            pass
        _maybe_invalidate_on_fingerprint_change()

    norm = str(normalize_profit_profile(profile))
    fp = _env_fingerprint()
    key = (norm, fp, bool(dotenv))

    cached = _profile_cache.get(key)
    if cached is not None:
        _stats.hits += 1
        logger.debug("profit profile cache HIT profile=%s (hits=%d)", norm, _stats.hits)
        return cached

    _stats.misses += 1
    logger.debug("profit profile cache MISS profile=%s (misses=%d)", norm, _stats.misses)

    saved_profile = os.environ.get(PROFIT_PROFILE_ENV)
    try:
        os.environ[PROFIT_PROFILE_ENV] = norm
        cfg = build_profit_config()
        _profile_cache[key] = cfg
        return cfg
    finally:
        if saved_profile is None:
            os.environ.pop(PROFIT_PROFILE_ENV, None)
        else:
            os.environ[PROFIT_PROFILE_ENV] = saved_profile


def log_profit_profile_cache_summary(*, prefix: str = "") -> None:
    """Emit one INFO line with cache hit rate (for optimizer / grid builds)."""
    s = get_profit_profile_cache_stats()
    if s.total == 0:
        return
    pct = 100.0 * s.hits / s.total if s.total else 0.0
    msg = (
        f"ProfitConfig profile cache: {s.hits} hits, {s.misses} misses "
        f"({pct:.0f}% hit rate, {len(_profile_cache)} profiles stored)"
    )
    if prefix:
        msg = f"{prefix}: {msg}"
    logger.info(msg)


__all__ = [
    "ProfitProfileCacheStats",
    "clear_profit_profile_cache",
    "get_profit_profile_cache_stats",
    "load_cached_profit_profile",
    "log_profit_profile_cache_summary",
]
