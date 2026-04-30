"""Execution / trades / costs repository.

Owns the SQL bodies that have been moved out of ``memory.__init__``
during PR13. Functions that depend on cross-module mutable state
(``_pending_order_context``, ``_pending_graduated_params``,
``_calibration_version``) — i.e. the snapshot writers and
``upsert_calibrated_slippage`` — intentionally remain in
``memory.__init__`` for now. They will be moved in a follow-up once
the mutable-state coupling is broken.

Migrated this PR (read-side):
    * record_iv_snapshot, compute_iv_rank_percentile
    * get_execution_cost
    * get_calibrated_slippage

Still in memory.__init__ (writer / mutable-state coupled):
    * record_trade + _match_trade_to_signal
    * insert_execution_snapshot, update_execution_snapshot_fill,
      cancel_execution_snapshot
    * get_new_snapshot_count, get_filled_snapshots
    * get_snapshots_for_param_review
    * upsert_calibrated_slippage  (mutates _calibration_version)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from memory.repos.schema import get_db

logger = logging.getLogger(__name__)


# Minimum number of historical IV snapshots before trailing percentile
# is considered meaningful.  Below this we return None so callers fall
# back to the chain-derived approximation.
_IV_HISTORY_MIN_SAMPLES = 30


# ── IV history ──────────────────────────────────────────────────


def record_iv_snapshot(
    symbol: str, iv_current: float | None, source: str = "marketdata"
) -> None:
    """Record one daily IV snapshot for a symbol.

    Idempotent within the same UTC day: a second call on the same day
    overwrites the prior snapshot via ON CONFLICT(symbol, ts) DO UPDATE,
    where ts is the UTC date as a unix timestamp at midnight.
    """
    if iv_current is None:
        return
    try:
        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_ts = today.timestamp()
        db = get_db()
        db.execute(
            "INSERT INTO iv_history (symbol, ts, iv_current, source) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (symbol, ts) DO UPDATE SET "
            "iv_current = EXCLUDED.iv_current, source = EXCLUDED.source",
            (symbol.upper(), day_ts, float(iv_current), source),
        )
        db.commit()
    except Exception as e:
        logger.debug(f"record_iv_snapshot failed for {symbol}: {e}")


def compute_iv_rank_percentile(
    symbol: str,
    iv_current: float | None = None,
    lookback_days: int = 252,
) -> float | None:
    """Return IV rank as the percentile of `iv_current` versus trailing
    `lookback_days` of `iv_history` snapshots for `symbol`.

    Returns None when fewer than _IV_HISTORY_MIN_SAMPLES snapshots exist.
    If `iv_current` is None the most recent snapshot is used.
    Result is in [0, 100].
    """
    try:
        db = get_db()
        cutoff_ts = (
            datetime.now(timezone.utc).timestamp() - lookback_days * 86400
        )
        rows = db.execute(
            "SELECT iv_current FROM iv_history "
            "WHERE symbol = ? AND ts >= ? ORDER BY ts ASC",
            (symbol.upper(), cutoff_ts),
        ).fetchall()
        if not rows or len(rows) < _IV_HISTORY_MIN_SAMPLES:
            return None
        values = [float(r["iv_current"]) for r in rows]
        if iv_current is None:
            iv_current = values[-1]
        # Percentile = % of historical samples strictly below current IV.
        below = sum(1 for v in values if v < iv_current)
        return max(0.0, min(100.0, 100.0 * below / len(values)))
    except Exception as e:
        logger.debug(f"compute_iv_rank_percentile failed for {symbol}: {e}")
        return None


# ── Execution cost (read-side aggregation) ─────────────────────


def get_execution_cost(symbol: str | None = None) -> dict:
    """Query aggregated execution gaps from trade_feedback.

    Returns per-symbol cost model from last 30 days:
      {symbol -> {avg_gap, n, total_pnl, total_sim}}

    If symbol given, returns single entry.
    If not, returns top-10 symbols by trade count.
    """
    db = get_db()
    cutoff = "(NOW() - INTERVAL '30 days')"
    if symbol:
        row = db.execute(
            f"""SELECT symbol,
                       AVG(execution_gap) as avg_gap,
                       COUNT(*) as n,
                       SUM(actual_pnl) as total_pnl,
                       SUM(simulated_return) as total_sim
                FROM trade_feedback
                WHERE symbol = ? AND ts >= {cutoff}""",
            (symbol.upper(),),
        ).fetchone()
        if row and row["n"]:
            return {
                "symbol": row["symbol"],
                "avg_gap_pct": round(row["avg_gap"] or 0, 4),
                "trades": row["n"],
                "total_pnl": round(row["total_pnl"] or 0, 2),
                "total_sim": round(row["total_sim"] or 0, 4),
            }
        return {}
    else:
        rows = db.execute(
            f"""SELECT symbol,
                       AVG(execution_gap) as avg_gap,
                       COUNT(*) as n
                FROM trade_feedback
                WHERE ts >= {cutoff}
                GROUP BY symbol
                ORDER BY n DESC LIMIT 10"""
        ).fetchall()
        return {
            r["symbol"]: {"avg_gap_pct": round(r["avg_gap"] or 0, 4), "trades": r["n"]}
            for r in rows
        }


# ── Calibrated slippage (read-side) ────────────────────────────


def get_calibrated_slippage() -> dict[tuple[str, str, str], float]:
    """Get all calibrated slippage values as
    {(order_type, time_bucket, atr_bucket): median_bps}."""
    db = get_db()
    rows = db.execute(
        "SELECT order_type, time_bucket, atr_bucket, median_slippage_bps "
        "FROM calibrated_slippage"
    ).fetchall()
    return {
        (r["order_type"], r["time_bucket"], r["atr_bucket"]): r["median_slippage_bps"]
        for r in rows
    }


# ── Re-exports for legacy import paths (still owned by memory.__init__) ──
# These names are imported elsewhere via this repo for forward-compat;
# the actual implementations remain in `memory.__init__` until the
# coupled mutable-state items are extracted.
from memory import (  # noqa: E402,F401
    cancel_execution_snapshot,
    get_filled_snapshots,
    get_new_snapshot_count,
    get_snapshots_for_param_review,
    insert_execution_snapshot,
    record_trade,
    update_execution_snapshot_fill,
    upsert_calibrated_slippage,
)


__all__ = [
    "_IV_HISTORY_MIN_SAMPLES",
    "cancel_execution_snapshot",
    "compute_iv_rank_percentile",
    "get_calibrated_slippage",
    "get_execution_cost",
    "get_filled_snapshots",
    "get_new_snapshot_count",
    "get_snapshots_for_param_review",
    "insert_execution_snapshot",
    "record_iv_snapshot",
    "record_trade",
    "update_execution_snapshot_fill",
    "upsert_calibrated_slippage",
]
