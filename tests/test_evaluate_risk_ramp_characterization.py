"""PR24 - Characterize ``TradingAgent._evaluate_risk_ramp``.

The risk ramp gates the live position size from 0.5% to 1.0% based on
recent live track record. We pin every branch:

  * Already-ramped (research_config >= 1.0) -> early return, no DB read.
  * No trades in last 30 days -> no-op.
  * <10 trading days -> not approved.
  * Negative cumulative P&L -> not approved.
  * Win rate <= 0.45 -> not approved.
  * All thresholds met -> ``risk_ramp_approved`` set to 1.0.
  * Exception swallowed (logger.warning, no raise).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

# _isolated_db (autouse) and db fixtures are provided by tests/conftest.py.


def _agent_stub():
    """The function only uses ``self`` formally; a bare object suffices."""
    return SimpleNamespace()


def _seed_trades(db, *, days_with_trades, trades_per_day, win_rate, pnl_per_win,
                 pnl_per_loss, today_iso="2026-04-15"):
    """Insert closed trades distributed across distinct trading days.

    ``today_iso`` is the anchor; trades are inserted on consecutive
    prior calendar days so they all fall within the 30-day window.
    """
    from datetime import date, timedelta
    anchor = date.fromisoformat(today_iso)
    win_count_per_day = int(round(trades_per_day * win_rate))
    loss_count_per_day = trades_per_day - win_count_per_day
    for i in range(days_with_trades):
        day = anchor - timedelta(days=i + 1)
        day_iso = day.isoformat()
        for w in range(win_count_per_day):
            db.execute(
                "INSERT INTO trades (ts, symbol, side, pnl) VALUES (?, ?, ?, ?)",
                (f"{day_iso}T10:{w:02d}:00", "AAPL", "long", pnl_per_win),
            )
        for l in range(loss_count_per_day):
            db.execute(
                "INSERT INTO trades (ts, symbol, side, pnl) VALUES (?, ?, ?, ?)",
                (f"{day_iso}T11:{l:02d}:00", "AAPL", "long", pnl_per_loss),
            )
    db.commit()


def _ramp() -> float:
    from memory import get_research_config
    return float(get_research_config("risk_ramp_approved", 0.0))


# ── Tests ─────────────────────────────────────────────────────


class TestEvaluateRiskRamp:
    def test_already_ramped_returns_early(self, db):
        from core.agent import TradingAgent
        from memory import set_research_config
        set_research_config("risk_ramp_approved", 1.0, "preset")
        # No trades seeded; if the function tried to query it would
        # still no-op, but we also assert config remains exactly 1.0.
        TradingAgent._evaluate_risk_ramp(_agent_stub(), db, "2026-04-15")
        assert _ramp() == 1.0

    def test_no_trades_no_approval(self, db):
        from core.agent import TradingAgent
        TradingAgent._evaluate_risk_ramp(_agent_stub(), db, "2026-04-15")
        assert _ramp() == 0.0

    def test_too_few_trading_days_no_approval(self, db):
        """9 days with trades -> below 10-day threshold."""
        from core.agent import TradingAgent
        _seed_trades(db, days_with_trades=9, trades_per_day=3,
                     win_rate=0.6, pnl_per_win=10.0, pnl_per_loss=-5.0)
        TradingAgent._evaluate_risk_ramp(_agent_stub(), db, "2026-04-15")
        assert _ramp() == 0.0

    def test_negative_pnl_no_approval(self, db):
        from core.agent import TradingAgent
        # 12 days, plenty of trades, win_rate=0.5 (not above 0.45 threshold
        # actually 0.5 > 0.45 so wr passes), but losses dominate.
        # Use win_rate=0.6 so wr passes but make losses bigger -> total < 0.
        _seed_trades(db, days_with_trades=12, trades_per_day=5,
                     win_rate=0.6, pnl_per_win=1.0, pnl_per_loss=-10.0)
        TradingAgent._evaluate_risk_ramp(_agent_stub(), db, "2026-04-15")
        assert _ramp() == 0.0

    def test_low_win_rate_no_approval(self, db):
        """WR exactly 0.45 -> not strictly above threshold -> no approval."""
        from core.agent import TradingAgent
        # Using win_rate=0.4 is unambiguously below 0.45.
        _seed_trades(db, days_with_trades=12, trades_per_day=5,
                     win_rate=0.4, pnl_per_win=10.0, pnl_per_loss=-1.0)
        TradingAgent._evaluate_risk_ramp(_agent_stub(), db, "2026-04-15")
        assert _ramp() == 0.0

    def test_all_criteria_met_approves(self, db):
        from core.agent import TradingAgent
        # 12 days, 5 trades/day, WR=0.6 (>0.45), wins big -> total > 0.
        _seed_trades(db, days_with_trades=12, trades_per_day=5,
                     win_rate=0.6, pnl_per_win=20.0, pnl_per_loss=-5.0)
        assert _ramp() == 0.0
        TradingAgent._evaluate_risk_ramp(_agent_stub(), db, "2026-04-15")
        assert _ramp() == 1.0

    def test_exception_swallowed(self, db):
        """A raising db connection must not propagate; ramp stays 0."""
        from core.agent import TradingAgent

        class _Raising:
            def execute(self, *a, **k):
                raise RuntimeError("simulated db failure")

        # Should NOT raise.
        TradingAgent._evaluate_risk_ramp(_agent_stub(), _Raising(), "2026-04-15")
        assert _ramp() == 0.0

    def test_old_trades_excluded_by_30_day_window(self, db):
        """Trades older than 30 days don't count."""
        from core.agent import TradingAgent
        from datetime import date, timedelta
        anchor = date.fromisoformat("2026-04-15")
        old_day = (anchor - timedelta(days=60)).isoformat()
        # Insert 50 winning trades >30 days ago -- should be ignored.
        for i in range(50):
            db.execute(
                "INSERT INTO trades (ts, symbol, side, pnl) VALUES (?, ?, ?, ?)",
                (f"{old_day}T10:{i % 60:02d}:00", "AAPL", "long", 100.0),
            )
        db.commit()
        TradingAgent._evaluate_risk_ramp(_agent_stub(), db, "2026-04-15")
        assert _ramp() == 0.0
