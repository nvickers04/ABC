"""End-of-day review extracted from ``core.agent.TradingAgent``.

Pure refactor (PR21) — moves the body of ``_run_daily_review`` into a free
async function that takes the agent as its first argument. ``TradingAgent``
keeps a thin method that delegates here so the rest of the codebase is
unaffected.

Byte-identical behavior is preserved: same imports, same SQL, same logging
strings, same error swallowing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.config import TRADING_MODE

logger = logging.getLogger(__name__)


async def run_daily_review(agent) -> None:
    """End-of-day review: aggregate performance, run execution analysis if data threshold met.

    Called once when session transitions to POSTMARKET. Computes execution
    gaps, triggers execution analysis when enough snapshots exist.
    """
    try:
        from memory import get_db, get_new_snapshot_count
        db = get_db()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 1. Aggregate today's trade_feedback for execution gap
        rows = db.execute(
            """SELECT slot, symbol,
                      AVG(execution_gap) as avg_gap,
                      COUNT(*) as n,
                      SUM(actual_pnl) as total_pnl,
                      SUM(simulated_return) as total_sim
               FROM trade_feedback
               WHERE ts >= ? || 'T00:00:00'
               GROUP BY slot""",
            (today,),
        ).fetchall()

        for r in rows:
            gap = r["avg_gap"] if r["avg_gap"] else 0
            if abs(gap) > 0.005:  # >0.5% gap is noteworthy
                agent._emit_hypothesis(
                    hypothesis_type="execution_gap",
                    description=f"Slot {r['slot']} avg execution gap {gap:+.3f} on {r['n']} trades",
                    suggested_action=f"Factor {gap:+.1%} execution cost into slot {r['slot']} sizing",
                    priority=4,
                    related_slot=r["slot"],
                )

        today_trades = db.execute(
            "SELECT COUNT(*) as n, SUM(pnl) as total_pnl FROM trades WHERE ts >= ? || 'T00:00:00'",
            (today,),
        ).fetchone()
        day_trade_count = today_trades["n"] if today_trades and today_trades["n"] else 0

        # 2. Execution analysis: triggered by data threshold
        new_snaps = get_new_snapshot_count()
        if new_snaps >= 10:
            await agent._run_execution_analysis()

        # 3. Risk ramp-up evaluation (live mode only: 0.5% → 1.0%)
        if TRADING_MODE == "live":
            evaluate_risk_ramp(db, today)

        logger.info(f"Daily review complete for {today}: {day_trade_count} trades, {new_snaps} new snapshots")
    except Exception as e:
        logger.warning(f"Daily review failed: {e}")


def evaluate_risk_ramp(db, today: str) -> None:
    """Check if live trading performance warrants risk ramp-up 0.5% → 1.0%.

    Criteria: 10+ trading days with trades, cumulative P&L > 0, win rate > 45%.
    """
    try:
        from memory import get_research_config, set_research_config
        if get_research_config("risk_ramp_approved", 0.0) >= 1.0:
            return  # Already ramped

        # Count trading days with at least 1 trade in last 30 days
        stats = db.execute(
            """SELECT COUNT(DISTINCT date(ts)) as trading_days,
                      COUNT(*) as total_trades,
                      SUM(pnl) as total_pnl,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
               FROM trades
               WHERE ts >= date(?, '-30 days') || 'T00:00:00'""",
            (today,),
        ).fetchone()

        if not stats or not stats["total_trades"]:
            return

        trading_days = stats["trading_days"] or 0
        total_trades = stats["total_trades"] or 0
        total_pnl = stats["total_pnl"] or 0
        wins = stats["wins"] or 0
        win_rate = wins / total_trades if total_trades > 0 else 0

        if trading_days >= 10 and total_pnl > 0 and win_rate > 0.45:
            set_research_config(
                "risk_ramp_approved", 1.0,
                f"Auto-approved: {trading_days} days, {total_trades} trades, "
                f"P&L=${total_pnl:.2f}, WR={win_rate:.0%}"
            )
            logger.info(
                f"RISK RAMP-UP APPROVED: {trading_days} days, "
                f"{total_trades} trades, P&L=${total_pnl:.2f}, WR={win_rate:.0%} → 1.0%"
            )
        else:
            logger.info(
                f"Risk ramp check: {trading_days}/10 days, "
                f"P&L=${total_pnl:.2f}, WR={win_rate:.0%} — not yet"
            )
    except Exception as e:
        logger.warning(f"Risk ramp evaluation failed: {e}")

