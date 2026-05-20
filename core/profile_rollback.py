"""Automatic ProfitConfig profile rollback when a trial profile breaches live drawdown.

When ``TRADING_MODE=live`` and the active profile differs from the last *known-good*
profile, peak-to-trough drawdown since adoption is tracked. If drawdown reaches
``ABC_PROFILE_ROLLBACK_DRAWDOWN_PCT`` (default 10%%), ``PROFIT_PROFILE`` reverts to
the known-good profile and :func:`~core.central_profit_config.get_profit_config` is
reloaded.

State is persisted in ``data/profile_rollback_state.json`` (trader host).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.log_context import get_logger
from core.profit_config_state import get_active_profile_label, log_active_profit_config
from core.profit_profiles import PROFIT_PROFILE_ENV, normalize_profit_profile

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STATE_PATH = _REPO_ROOT / "data" / "profile_rollback_state.json"
_EVENTS_PATH = _REPO_ROOT / "logs" / "profile_rollback_events.jsonl"
_LOCK = threading.RLock()

# research_config float keys (optional mirror for ops dashboards)
_CFG_TRIAL_PEAK_NLV = "profile_rollback_trial_peak_nlv"
_CFG_LAST_ROLLBACK_TS = "profile_rollback_last_ts"


@dataclass
class ProfileRollbackState:
    known_good_profile: str = "balanced"
    trial_profile: str = ""
    trial_peak_nlv: float = 0.0
    trial_started_at: str = ""
    last_applied_profile: str = ""
    rollback_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProfileRollbackState:
        return cls(
            known_good_profile=str(raw.get("known_good_profile") or "balanced"),
            trial_profile=str(raw.get("trial_profile") or ""),
            trial_peak_nlv=float(raw.get("trial_peak_nlv") or 0),
            trial_started_at=str(raw.get("trial_started_at") or ""),
            last_applied_profile=str(raw.get("last_applied_profile") or ""),
            rollback_count=int(raw.get("rollback_count") or 0),
            history=list(raw.get("history") or [])[-20:],
        )


def _drawdown_threshold_pct() -> float:
    raw = os.getenv("ABC_PROFILE_ROLLBACK_DRAWDOWN_PCT", "10").strip()
    try:
        return max(1.0, min(50.0, float(raw)))
    except ValueError:
        return 10.0


def is_live_rollback_enabled() -> bool:
    """True when rollback guard should run (live trading, not disabled)."""
    if os.getenv("ABC_PROFILE_ROLLBACK_ENABLED", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    if os.getenv("ABC_PROFILE_ROLLBACK_FORCE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    try:
        from core.risk_execution_config import get_risk_execution_config

        return str(get_risk_execution_config().trading_mode) == "live"
    except Exception:
        return False


def load_state() -> ProfileRollbackState:
    with _LOCK:
        if not _STATE_PATH.is_file():
            return ProfileRollbackState()
        try:
            raw = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return ProfileRollbackState.from_dict(raw)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("profile_rollback state unreadable: %s", e)
        return ProfileRollbackState()


def save_state(state: ProfileRollbackState) -> None:
    with _LOCK:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(state)
        tmp = _STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, _STATE_PATH)


def _append_event(event: dict[str, Any]) -> None:
    _EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, default=str)
    with _LOCK:
        with open(_EVENTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _current_nlv(gateway: Any) -> float:
    if gateway is None:
        return 0.0
    nlv = float(getattr(gateway, "net_liquidation", 0) or 0)
    if nlv <= 0:
        nlv = float(getattr(gateway, "cash_value", 0) or 0)
    return max(0.0, nlv)


def _normalize_profile_label(profile: str) -> str:
    try:
        return str(normalize_profit_profile(profile))
    except ValueError:
        return str(profile or "balanced").strip().lower() or "balanced"


def initialize_profile_rollback_state(*, gateway: Any | None = None) -> ProfileRollbackState:
    """Ensure state file exists; align ``last_applied`` with current env profile."""
    current = _normalize_profile_label(get_active_profile_label())
    state = load_state()
    if not state.last_applied_profile:
        state.last_applied_profile = current
    if not state.known_good_profile:
        state.known_good_profile = current
    nlv = _current_nlv(gateway)
    state = _sync_trial_state(state, current, nlv)
    save_state(state)
    return state


def on_profile_applied(old_profile: str, new_profile: str, *, gateway: Any | None = None) -> None:
    """Record an explicit profile switch (CLI, optimize_for_profit, daily summary)."""
    if not is_live_rollback_enabled():
        return
    old_n = _normalize_profile_label(old_profile)
    new_n = _normalize_profile_label(new_profile)
    if old_n == new_n:
        return
    state = load_state()
    if state.known_good_profile == new_n:
        state.trial_profile = ""
        state.trial_peak_nlv = 0.0
        state.trial_started_at = ""
    else:
        state.known_good_profile = old_n
        state.trial_profile = new_n
        nlv = _current_nlv(gateway)
        state.trial_peak_nlv = nlv if nlv > 0 else state.trial_peak_nlv
        state.trial_started_at = datetime.now(timezone.utc).isoformat()
    state.last_applied_profile = new_n
    save_state(state)
    logger.info(
        "profile_rollback: profile switch recorded known_good=%s trial=%s new=%s",
        state.known_good_profile,
        state.trial_profile or "(none)",
        new_n,
    )


def _sync_trial_state(
    state: ProfileRollbackState,
    current: str,
    nlv: float,
) -> ProfileRollbackState:
    """Update trial tracking when env profile differs from known-good."""
    if current == state.known_good_profile:
        state.trial_profile = ""
        state.trial_peak_nlv = 0.0
        state.trial_started_at = ""
        state.last_applied_profile = current
        return state

    if state.trial_profile != current:
        state.trial_profile = current
        state.trial_peak_nlv = nlv if nlv > 0 else 0.0
        state.trial_started_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "profile_rollback: new trial profile %r (known_good=%r) baseline_nlv=%.2f",
            current,
            state.known_good_profile,
            state.trial_peak_nlv,
        )
    elif nlv > 0:
        state.trial_peak_nlv = max(state.trial_peak_nlv, nlv)

    state.last_applied_profile = current
    return state


def _trial_drawdown_pct(state: ProfileRollbackState, nlv: float) -> float:
    peak = state.trial_peak_nlv
    if peak <= 0 or nlv <= 0 or nlv >= peak:
        return 0.0
    return (peak - nlv) / peak * 100.0


def execute_profile_rollback(
    *,
    state: ProfileRollbackState,
    trial_profile: str,
    known_good: str,
    drawdown_pct: float,
    peak_nlv: float,
    current_nlv: float,
) -> dict[str, Any]:
    """Revert ``PROFIT_PROFILE`` to known-good and reload master ProfitConfig."""
    threshold = _drawdown_threshold_pct()
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "profile_rollback",
        "trial_profile": trial_profile,
        "known_good_profile": known_good,
        "drawdown_pct": round(drawdown_pct, 4),
        "threshold_pct": threshold,
        "trial_peak_nlv": peak_nlv,
        "current_nlv": current_nlv,
        "trading_mode": "live",
    }

    logger.critical("=" * 72)
    logger.critical(
        "PROFILE ROLLBACK: trial profile %r exceeded %.1f%% drawdown (actual %.2f%%)",
        trial_profile,
        threshold,
        drawdown_pct,
    )
    logger.critical(
        "  Reverting PROFIT_PROFILE: %r -> %r  (peak NLV $%s -> current $%s)",
        trial_profile,
        known_good,
        f"{peak_nlv:,.2f}",
        f"{current_nlv:,.2f}",
    )
    logger.critical("  Action: export %s=%s", PROFIT_PROFILE_ENV, known_good)
    logger.critical("=" * 72)
    log_active_profit_config(
        logger,
        "profile rollback executed",
        level=30,  # WARNING
        rollback_event=event,
    )

    os.environ[PROFIT_PROFILE_ENV] = known_good
    from core.central_profit_config import get_profit_config, sync_research_host_from_profit_config

    get_profit_config().reload(dotenv=False)
    sync_research_host_from_profit_config(publish_heartbeat=True)

    state.rollback_count += 1
    state.history.append(event)
    state.trial_profile = ""
    state.trial_peak_nlv = 0.0
    state.trial_started_at = ""
    state.last_applied_profile = known_good
    save_state(state)
    _append_event(event)

    try:
        from memory import set_research_config

        set_research_config(_CFG_LAST_ROLLBACK_TS, time.time(), reason="profile_rollback", log=True)
        set_research_config(_CFG_TRIAL_PEAK_NLV, 0.0, reason="profile_rollback_cleared", log=False)
    except Exception as e:
        logger.debug("profile_rollback research_config mirror skipped: %s", e)

    return event


def check_profile_rollback_live(gateway: Any) -> dict[str, Any] | None:
    """If trial profile drawdown breaches threshold, rollback and return event dict."""
    try:
        from core.profile_ab_test import is_ab_test_active

        if is_ab_test_active():
            return None
    except Exception:
        pass
    if not is_live_rollback_enabled():
        return None

    nlv = _current_nlv(gateway)
    if nlv <= 0:
        return None

    current = _normalize_profile_label(get_active_profile_label())
    state = load_state()
    if not state.known_good_profile:
        state.known_good_profile = current

    state = _sync_trial_state(state, current, nlv)
    save_state(state)

    if current == state.known_good_profile or not state.trial_profile:
        return None

    dd = _trial_drawdown_pct(state, nlv)
    threshold = _drawdown_threshold_pct()

    try:
        from memory import set_research_config

        set_research_config(
            _CFG_TRIAL_PEAK_NLV,
            float(state.trial_peak_nlv),
            reason="profile_rollback_trial_peak",
            log=False,
        )
    except Exception:
        pass

    if dd < threshold:
        if dd >= threshold * 0.85:
            logger.warning(
                "profile_rollback: trial %r drawdown %.2f%% approaching %.1f%% limit "
                "(peak=$%.2f current=$%.2f known_good=%r)",
                state.trial_profile,
                dd,
                threshold,
                state.trial_peak_nlv,
                nlv,
                state.known_good_profile,
            )
        return None

    return execute_profile_rollback(
        state=state,
        trial_profile=state.trial_profile,
        known_good=state.known_good_profile,
        drawdown_pct=dd,
        peak_nlv=state.trial_peak_nlv,
        current_nlv=nlv,
    )


__all__ = [
    "ProfileRollbackState",
    "check_profile_rollback_live",
    "execute_profile_rollback",
    "initialize_profile_rollback_state",
    "is_live_rollback_enabled",
    "load_state",
    "on_profile_applied",
    "save_state",
]
