"""
Profit Protection Metrics — tracks how well the agent retains gains and contains losses.

Three core KPIs:
1. Profit Retention Rate  — realized winner PnL / peak unrealized winner PnL
2. Loss Containment Rate  — planned stop distance / actual realized loss
3. Session Giveback %      — (peak equity - closing equity) / peak equity

These answer the question: "Are we capturing enough of what the market gives us?"
PnL alone shows the outcome; these show *why*.
"""

import logging
import threading
import time
from datetime import datetime, timezone, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PositionHighWaterTracker:
    """
    Tracks per-position high water marks (peak unrealized P&L) in real time.

    Updated every time LiveState refreshes position data. When a position
    closes, the caller reads the peak and compares to realized PnL.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # symbol -> { peak_pnl, peak_pnl_pct, peak_price, entry_price, samples }
        self._hwm: Dict[str, dict] = {}

    def update(self, symbol: str, unrealized_pnl: float, pnl_pct: float,
               market_price: float, entry_price: float) -> None:
        """Record a P&L observation. Updates peak if new high."""
        with self._lock:
            existing = self._hwm.get(symbol)
            if existing is None:
                self._hwm[symbol] = {
                    "peak_pnl": unrealized_pnl,
                    "peak_pnl_pct": pnl_pct,
                    "peak_price": market_price,
                    "entry_price": entry_price,
                    "samples": 1,
                    "first_seen": datetime.now(timezone.utc).isoformat(),
                    "peak_time": datetime.now(timezone.utc).isoformat(),
                }
            else:
                existing["samples"] += 1
                if unrealized_pnl > existing["peak_pnl"]:
                    existing["peak_pnl"] = unrealized_pnl
                    existing["peak_pnl_pct"] = pnl_pct
                    existing["peak_price"] = market_price
                    existing["peak_time"] = datetime.now(timezone.utc).isoformat()

    def get_peak(self, symbol: str) -> Optional[dict]:
        """Return the HWM record for a symbol, or None."""
        with self._lock:
            return self._hwm.get(symbol)

    def pop_peak(self, symbol: str) -> Optional[dict]:
        """Return and remove the HWM record (used when position closes)."""
        with self._lock:
            return self._hwm.pop(symbol, None)

    def snapshot(self) -> Dict[str, dict]:
        """Thread-safe snapshot of all HWM records."""
        with self._lock:
            return dict(self._hwm)


class SessionEquityTracker:
    """
    Tracks session-level equity high water mark for Giveback % calculation.

    Fed by LiveState on every account update (net_liq + unrealized).
    At end of day, compare closing equity to peak → giveback %.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._peak_equity: float = 0.0
        self._peak_time: Optional[str] = None
        self._first_equity: Optional[float] = None
        self._latest_equity: float = 0.0
        self._samples: int = 0
        self._date: str = date.today().isoformat()

    def update(self, equity: float) -> None:
        """Record a net liquidation observation."""
        if equity <= 0:
            return
        with self._lock:
            # Reset if new day
            today = date.today().isoformat()
            if today != self._date:
                self._peak_equity = 0.0
                self._peak_time = None
                self._first_equity = None
                self._samples = 0
                self._date = today

            self._latest_equity = equity
            self._samples += 1

            if self._first_equity is None:
                self._first_equity = equity

            if equity > self._peak_equity:
                self._peak_equity = equity
                self._peak_time = datetime.now(timezone.utc).isoformat()

    def get_giveback(self) -> Optional[dict]:
        """Compute current giveback % from peak."""
        with self._lock:
            if self._peak_equity <= 0 or self._samples < 2:
                return None
            giveback_dollars = self._peak_equity - self._latest_equity
            giveback_pct = (giveback_dollars / self._peak_equity) * 100 if self._peak_equity > 0 else 0
            return {
                "peak_equity": round(self._peak_equity, 2),
                "peak_time": self._peak_time,
                "current_equity": round(self._latest_equity, 2),
                "first_equity": round(self._first_equity, 2) if self._first_equity else None,
                "giveback_dollars": round(giveback_dollars, 2),
                "giveback_pct": round(giveback_pct, 2),
                "samples": self._samples,
            }


def compute_profit_retention(closed_trades: List[dict], hwm_tracker: PositionHighWaterTracker) -> dict:
    """
    Compute Profit Retention Rate from today's closed trades.

    For winning trades only:
      retention = realized_pnl / peak_unrealized_pnl

    A value of 0.70 means the agent captured 70% of the max profit
    the market offered on those positions. Below 0.50 = giving back
    too much; above 0.80 = excellent exit timing.
    """
    winners = []
    total_realized = 0.0
    total_peak = 0.0
    details = []

    for t in closed_trades:
        pnl = t.get("pnl", 0)
        if pnl <= 0:
            continue
        symbol = t.get("symbol", "")
        hwm = hwm_tracker.pop_peak(symbol)  # already popped by _record_closed_trade; check fallback
        peak_pnl = hwm["peak_pnl"] if hwm and hwm.get("peak_pnl", 0) > 0 else pnl

        retention = pnl / peak_pnl if peak_pnl > 0 else 1.0
        total_realized += pnl
        total_peak += peak_pnl

        details.append({
            "symbol": symbol,
            "realized_pnl": round(pnl, 2),
            "peak_pnl": round(peak_pnl, 2),
            "retention": round(retention, 2),
        })
        winners.append(retention)

    overall_retention = (total_realized / total_peak) if total_peak > 0 else None

    return {
        "winner_count": len(winners),
        "overall_retention": round(overall_retention, 2) if overall_retention is not None else None,
        "avg_retention": round(sum(winners) / len(winners), 2) if winners else None,
        "total_realized_winner_pnl": round(total_realized, 2),
        "total_peak_winner_pnl": round(total_peak, 2),
        "per_trade": details,
    }


def compute_loss_containment(closed_trades: List[dict]) -> dict:
    """
    Compute Loss Containment Rate from today's closed trades.

    For losing trades:
      containment = planned_stop_loss / actual_loss

    If actual loss <= planned stop loss → containment >= 1.0 (good).
    If actual loss > planned stop loss → containment < 1.0 (slippage/hesitation).

    Falls back to entry_price * 2% as the "planned" stop if no explicit
    stop distance is recorded (assumes standard 2% risk per trade).
    """
    losers = []
    details = []

    for t in closed_trades:
        pnl = t.get("pnl", 0)
        if pnl >= 0:
            continue
        actual_loss = abs(pnl)
        symbol = t.get("symbol", "")
        entry = t.get("entry_price", 0)
        qty = t.get("quantity", 0)

        # Planned stop distance: use explicit stop_distance if recorded,
        # else fallback to 2% of entry as the assumed target stop.
        planned_stop_distance = t.get("planned_stop_distance")
        if planned_stop_distance and planned_stop_distance > 0:
            planned_loss = planned_stop_distance * qty
        else:
            # Default: 2% of entry * qty as the standard risk budget
            planned_loss = entry * 0.02 * qty if entry > 0 and qty > 0 else actual_loss

        containment = planned_loss / actual_loss if actual_loss > 0 else 1.0

        details.append({
            "symbol": symbol,
            "actual_loss": round(actual_loss, 2),
            "planned_loss": round(planned_loss, 2),
            "containment": round(containment, 2),
        })
        losers.append(containment)

    return {
        "loser_count": len(losers),
        "avg_containment": round(sum(losers) / len(losers), 2) if losers else None,
        "contained_count": sum(1 for c in losers if c >= 1.0),
        "breached_count": sum(1 for c in losers if c < 1.0),
        "per_trade": details,
    }


def compute_session_metrics(
    closed_trades: List[dict],
    hwm_tracker: PositionHighWaterTracker,
    equity_tracker: SessionEquityTracker,
) -> dict:
    """
    Combined profit protection dashboard for the current session.

    Injected into stats / daily_summary so the agent and user
    see profit retention quality every session.
    """
    retention = compute_profit_retention(closed_trades, hwm_tracker)
    containment = compute_loss_containment(closed_trades)
    giveback = equity_tracker.get_giveback()

    return {
        "profit_retention": retention,
        "loss_containment": containment,
        "session_giveback": giveback,
    }


def format_profit_metrics_for_agent(
    closed_trades: List[dict],
    hwm_tracker: PositionHighWaterTracker,
    equity_tracker: SessionEquityTracker,
) -> List[str]:
    """Format profit protection metrics as lines for agent context injection."""
    lines: List[str] = []

    # Only show if we have meaningful data
    if not closed_trades:
        return lines

    retention = compute_profit_retention(closed_trades, hwm_tracker)
    containment = compute_loss_containment(closed_trades)
    giveback = equity_tracker.get_giveback()

    winners = retention["winner_count"]
    losers = containment["loser_count"]

    if winners == 0 and losers == 0:
        return lines

    lines.append("=== PROFIT PROTECTION SCORECARD ===")

    # Retention
    if winners > 0 and retention["overall_retention"] is not None:
        ret_pct = retention["overall_retention"] * 100
        if ret_pct >= 80:
            grade = "🟢 EXCELLENT"
        elif ret_pct >= 60:
            grade = "🟡 GOOD"
        elif ret_pct >= 40:
            grade = "🟠 WEAK"
        else:
            grade = "🔴 POOR"
        lines.append(
            f"PROFIT RETENTION: {ret_pct:.0f}% {grade} — "
            f"captured ${retention['total_realized_winner_pnl']:,.2f} "
            f"of ${retention['total_peak_winner_pnl']:,.2f} peak ({winners} winners)"
        )

    # Containment
    if losers > 0 and containment["avg_containment"] is not None:
        cont = containment["avg_containment"]
        breached = containment["breached_count"]
        if cont >= 1.0:
            grade = "🟢 CONTAINED"
        elif cont >= 0.8:
            grade = "🟡 MINOR BREACH"
        else:
            grade = "🔴 STOPS NOT WORKING"
        lines.append(
            f"LOSS CONTAINMENT: {cont:.2f}x {grade} — "
            f"{breached}/{losers} losses exceeded planned stop"
        )

    # Giveback
    if giveback and giveback["samples"] >= 5:
        gb_pct = giveback["giveback_pct"]
        gb_dollars = giveback["giveback_dollars"]
        if gb_pct <= 5:
            grade = "🟢 TIGHT"
        elif gb_pct <= 20:
            grade = "🟡 MODERATE"
        elif gb_pct <= 40:
            grade = "🟠 ELEVATED"
        else:
            grade = "🔴 HIGH"
        lines.append(
            f"SESSION GIVEBACK: {gb_pct:.1f}% {grade} — "
            f"${gb_dollars:+,.2f} from peak ${giveback['peak_equity']:,.2f}"
        )

    lines.append("")
    return lines
