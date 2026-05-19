"""SafetyController — extracted loss / drawdown / cost guardrails.

This module hosts the *evaluation* portion of the safety logic that previously
lived inline on :class:`core.agent.TradingAgent`:

* ``_capture_start_of_day_cash``
* ``_check_daily_loss``
* ``_check_intraday_drawdown``
* ``_check_llm_cost``

Behavior parity with the original implementation is locked by
``tests/test_runtime_characterization.py``. The numeric formulas, fallback
chains, and return contracts (``Optional[float]`` percent or ``bool``) are
preserved verbatim so that the agent's call sites can delegate without any
observable change.

The actual ``flatten_all`` / halt action is *not* performed here — the agent
keeps the side-effecting ``_emergency_flatten`` so that future work
(``EmergencyActions`` per the stabilization plan) can split orchestration
from policy without disturbing this slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.log_context import get_logger
from core.runtime.interfaces import BrokerGatewayProtocol, CostTrackerProtocol

logger = get_logger(__name__)


@dataclass
class SafetyVerdict:
    """Aggregated result of a safety evaluation pass.

    ``triggered`` is True when *any* check breached its limit. The first
    breaching reason is exposed as ``reason`` for use in
    ``_emergency_flatten``. Individual check values are kept for logging
    / diagnostics.
    """

    triggered: bool = False
    reason: str = ""
    daily_loss_pct: Optional[float] = None
    drawdown_pct: Optional[float] = None
    llm_cost_breached: bool = False


class SafetyController:
    """Pure-function safety evaluator (no broker side-effects).

    State retained across cycles:

    * ``start_of_day_cash`` — baseline NLV captured at session open.
    * ``session_high_water`` — running peak NLV for drawdown.

    All thresholds are injected so tests can pin specific values without
    importing ``core.config``.
    """

    def __init__(
        self,
        gateway: BrokerGatewayProtocol,
        cost_tracker: CostTrackerProtocol,
        *,
        max_daily_loss_pct: float,
        intraday_drawdown_pct: float,
        max_daily_llm_cost: float,
    ) -> None:
        self.gateway = gateway
        self.cost_tracker = cost_tracker
        self.max_daily_loss_pct = max_daily_loss_pct
        self.intraday_drawdown_pct = intraday_drawdown_pct
        self.max_daily_llm_cost = max_daily_llm_cost

        self.start_of_day_cash: Optional[float] = None
        self.session_high_water: Optional[float] = None

    # ── State capture ────────────────────────────────────────────

    def capture_start_of_day_cash(self) -> Optional[float]:
        """Capture NLV (or cash fallback) baseline once per session.

        Returns the captured baseline, or ``None`` if both sources are
        unavailable. Subsequent calls are no-ops while ``start_of_day_cash``
        is set (matches original guard).
        """
        if self.start_of_day_cash is not None:
            return self.start_of_day_cash

        net_liq = self.gateway.net_liquidation if self.gateway else 0
        cash = self.gateway.cash_value if self.gateway else 0
        baseline = net_liq if net_liq > 0 else cash
        if baseline > 0:
            self.start_of_day_cash = baseline
            logger.info(
                f"Start-of-day Net Liq: ${baseline:,.2f} (cash: ${cash:,.2f}) "
                f"(loss limit: -{self.max_daily_loss_pct}% = "
                f"${baseline * self.max_daily_loss_pct / 100:,.2f})"
            )
            return baseline
        return None

    # ── Individual checks ────────────────────────────────────────

    def check_daily_loss(self) -> Optional[float]:
        """Return loss% if the daily-loss limit is breached, else ``None``.

        Uses NLV (with cash fallback) so that ordinary stock purchases —
        which decrease cash but leave NLV unchanged — do not generate a
        false flatten.
        """
        if not self.start_of_day_cash or self.start_of_day_cash <= 0:
            return None
        current = self.gateway.net_liquidation if self.gateway else 0
        if current <= 0:
            current = self.gateway.cash_value if self.gateway else 0
        if current <= 0:
            return None
        loss_pct = (self.start_of_day_cash - current) / self.start_of_day_cash * 100
        return loss_pct if loss_pct >= self.max_daily_loss_pct else None

    def check_intraday_drawdown(self) -> Optional[float]:
        """Return drawdown% if peak-to-trough breach, else ``None``."""
        current = self.gateway.net_liquidation if self.gateway else 0
        if current <= 0:
            current = self.gateway.cash_value if self.gateway else 0
        if current <= 0:
            return None

        if self.session_high_water is None or current > self.session_high_water:
            self.session_high_water = current

        drawdown_pct = (self.session_high_water - current) / self.session_high_water * 100
        return drawdown_pct if drawdown_pct >= self.intraday_drawdown_pct else None

    def check_llm_cost(self) -> bool:
        """``True`` when today's LLM spend has reached its ceiling."""
        summary = self.cost_tracker.get_budget_summary()
        return summary.today_llm_cost >= self.max_daily_llm_cost

    # ── Aggregate evaluation ─────────────────────────────────────

    def evaluate(self) -> SafetyVerdict:
        """Run all checks; return a :class:`SafetyVerdict`.

        Order of priority (matches inline agent logic):
        daily-loss → intraday-drawdown → LLM token buckets → LLM-cost.
        """
        verdict = SafetyVerdict()

        verdict.daily_loss_pct = self.check_daily_loss()
        if verdict.daily_loss_pct is not None:
            verdict.triggered = True
            verdict.reason = f"Daily loss: -{verdict.daily_loss_pct:.1f}%"
            return verdict

        verdict.drawdown_pct = self.check_intraday_drawdown()
        if verdict.drawdown_pct is not None:
            verdict.triggered = True
            verdict.reason = (
                f"Intraday drawdown: -{verdict.drawdown_pct:.1f}% from session high"
            )
            return verdict

        check_tok = getattr(self.cost_tracker, "check_daily_token_limits", None)
        if callable(check_tok):
            tok = check_tok()
            if tok:
                verdict.triggered = True
                verdict.reason = f"LLM token daily limit: {tok}"
                return verdict

        if self.check_llm_cost():
            verdict.triggered = True
            verdict.llm_cost_breached = True
            verdict.reason = (
                f"LLM cost ceiling ${self.max_daily_llm_cost} reached"
            )
        return verdict
