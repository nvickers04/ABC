"""
Cycle helpers extracted from ``core.agent.TradingAgent.run_cycle``.

This module currently hosts the open-gap volatility guard. Behavior is
byte-identical to the previously-inline implementation: same prompt strings,
same state mutation on ``agent._gap_guard_until``, same silent-on-error
semantics for the broad outer exception handler.

The function takes the agent as its state container (it reads/writes
``agent._current_session`` and ``agent._gap_guard_until``) so the call site in
``run_cycle`` reduces to a single line.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from core.config import OPEN_GAP_GUARD_PCT, OPEN_GUARD_DELAY_MINUTES
from data.data_provider import get_data_provider
from data.market_hours import get_market_hours_provider

logger = logging.getLogger(__name__)


async def evaluate_gap_guard(agent: Any) -> str:
    """Compute the gap-guard prompt and update ``agent._gap_guard_until``.

    Returns the prompt text (possibly empty). Mirrors the inline logic
    previously embedded in ``TradingAgent.run_cycle``: only active during
    the regular session, only checks the gap in the first 5 minutes after
    the open, and silently swallows any unexpected error in the outer
    block (preserving prior behavior).
    """
    gap_guard_prompt = ""
    if agent._current_session != "regular":
        return gap_guard_prompt
    try:
        if agent._gap_guard_until and datetime.now(timezone.utc) < agent._gap_guard_until:
            mins_left = int(
                (agent._gap_guard_until - datetime.now(timezone.utc)).total_seconds() / 60
            )
            gap_guard_prompt = (
                f"\n⚠️ GAP GUARD ACTIVE: Large overnight gap detected. "
                f"Wait {mins_left} more minutes before new entries. "
                f"Manage existing positions only.\n"
            )
        elif agent._gap_guard_until is None:
            # First regular-session cycle: check for gap
            _mh = get_market_hours_provider()
            _info_gap = _mh.get_session_info()
            mins_since_open = _info_gap.get("minutes_to_close")
            # Only check gap in first 5 minutes of regular session
            if mins_since_open is not None:
                total_regular = 390  # minutes in regular session
                mins_elapsed = total_regular - mins_since_open
                if mins_elapsed <= 5:
                    try:
                        dp = get_data_provider()
                        spy_quote = await dp.get_quote("SPY")
                        if spy_quote:
                            last = spy_quote.get("last", 0) or spy_quote.get("close", 0)
                            prev_close = spy_quote.get("previous_close", 0) or spy_quote.get("close", 0)
                            if last > 0 and prev_close > 0:
                                gap_pct = abs(last - prev_close) / prev_close * 100
                                if gap_pct >= OPEN_GAP_GUARD_PCT:
                                    agent._gap_guard_until = (
                                        datetime.now(timezone.utc)
                                        + timedelta(minutes=OPEN_GUARD_DELAY_MINUTES)
                                    )
                                    gap_guard_prompt = (
                                        f"\n⚠️ GAP GUARD: SPY gapped {gap_pct:.1f}% overnight. "
                                        f"Waiting {OPEN_GUARD_DELAY_MINUTES} min before new entries. "
                                        f"Manage existing positions only.\n"
                                    )
                                    logger.warning(f"Gap guard triggered: SPY gap {gap_pct:.1f}%")
                                else:
                                    agent._gap_guard_until = datetime.min.replace(tzinfo=timezone.utc)
                    except Exception as e:
                        logger.debug(f"Gap guard quote check failed: {e}")
                        agent._gap_guard_until = datetime.min.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return gap_guard_prompt
