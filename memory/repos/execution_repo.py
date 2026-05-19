"""Execution / trades / costs repository.

Canonical home for trade recording, execution snapshots, IV history,
calibrated slippage, and execution-cost aggregates. ``from memory import …``
re-exports via thin shims in ``memory.__init__``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import memory.session_state as session_state
from memory.repos.config_repo import get_research_config
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


def record_trade(
    symbol: str,
    side: str,
    pnl: float,
    held_minutes: int = 0,
    signal_id: int | None = None,
) -> int | None:
    """Record a closed trade into the trades table.

    This feeds the feedback loop: _match_trade_to_signal() matches these
    rows to template_recommendations by symbol + time proximity and writes
    trade_feedback rows tracking execution gaps.

    Returns the trade row id, or None on failure.
    """
    try:
        db = get_db()
        cur = db.execute(
            """INSERT INTO trades (ts, symbol, side, pnl, signal_id, held_minutes)
               VALUES (?, ?, ?, ?, ?, ?)
               RETURNING id""",
            (
                datetime.now(timezone.utc).isoformat(),
                symbol.upper(),
                side,
                round(pnl, 2),
                signal_id,
                held_minutes,
            ),
        )
        db.commit()
        row = cur.fetchone()
        trade_id = int(row["id"]) if row else None
        logger.info(f"Trade recorded: {symbol} {side} pnl=${pnl:.2f} id={trade_id}")

        # Immediately try to match this trade to a signal for feedback
        try:
            _match_trade_to_signal(db, trade_id, symbol)
        except Exception as match_err:
            logger.debug(f"Trade-signal matching failed: {match_err}")

        return trade_id
    except Exception as e:
        logger.warning(f"Failed to record trade: {e}")
        return None


def _match_trade_to_signal(db, trade_id: int, symbol: str) -> None:
    """Match a closed trade to the most recent template_recommendation for the
    symbol and write a trade_feedback row.

    The legacy `live_signals` ⨝ `signals` join was removed when those tables
    were retired; we now match against the signal-engine's recommendations.
    """
    trade = db.execute(
        "SELECT id, symbol, side, pnl, ts FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    if not trade:
        return

    # Convert the trade's ISO timestamp to a unix epoch float so we can
    # compare against the REAL `ts` column on template_recommendations.
    try:
        trade_epoch = datetime.fromisoformat(trade["ts"]).timestamp()
    except Exception:
        trade_epoch = None

    rec = None
    if trade_epoch is not None:
        # Within a 7-day window pick the closest recommendation by ts.
        rec = db.execute(
            """SELECT rowid AS id, template_name, direction, entry_price,
                      composite_score, ts
               FROM template_recommendations
               WHERE symbol = ?
                 AND ABS(ts - ?) < 7.0 * 86400
               ORDER BY ABS(ts - ?) ASC
               LIMIT 1""",
            (symbol.upper(), trade_epoch, trade_epoch),
        ).fetchone()

    if not rec:
        logger.debug(f"No matching recommendation for trade {trade_id} ({symbol})")
        return

    sim_return = rec["composite_score"] or 0.0
    actual_pnl = trade["pnl"] or 0
    entry_price = rec["entry_price"]

    if entry_price and entry_price > 0:
        actual_return_pct = (actual_pnl / entry_price) * 100
        execution_gap = actual_return_pct - sim_return
    else:
        execution_gap = None

    db.execute(
        """INSERT INTO trade_feedback
           (ts, trade_id, signal_id, slot, simulated_return,
            actual_pnl, execution_gap, symbol, template_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            trade_id,
            rec["id"],
            None,                  # legacy slot column — unused for new rows
            sim_return,
            actual_pnl,
            execution_gap,
            symbol.upper(),
            rec["template_name"],
        ),
    )
    db.commit()
    gap_str = f"{execution_gap:.2f}%" if execution_gap is not None else "unknown"
    logger.info(
        f"Trade feedback: trade {trade_id} -> template {rec['template_name']} "
        f"gap={gap_str}"
    )


# ═══════════════════════════════════════════════════════════════
# EXECUTION AUTORESEARCH
# ═══════════════════════════════════════════════════════════════

def _time_bucket(ts_iso: str | None) -> str:
    """Classify a timestamp into a time-of-day bucket (ET).

    open     = 09:30–09:45
    morning  = 09:45–12:00
    midday   = 12:00–15:45
    close    = 15:45–16:00
    extended = everything else
    """
    if not ts_iso:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_iso)
        et = dt.astimezone(ZoneInfo("America/New_York"))
        minutes = et.hour * 60 + et.minute
        if 570 <= minutes < 585:
            return "open"
        elif 585 <= minutes < 720:
            return "morning"
        elif 720 <= minutes < 945:
            return "midday"
        elif 945 <= minutes < 960:
            return "close"
        else:
            return "extended"
    except Exception:
        return "unknown"


def _atr_bucket(atr_pct: float | None) -> str:
    """Classify ATR % into volatility bucket."""
    if atr_pct is None:
        return "unknown"
    if atr_pct < 1.5:
        return "low"
    elif atr_pct < 3.0:
        return "medium"
    else:
        return "high"


# ── Canonical order type mapping ─────────────────────────────
# IBKR order type string → canonical name used in param_key schema.
# Many IBKR types share the same base order_type (e.g. MKT) but differ
# by algoStrategy or tif, so a bare mapping is insufficient.
# `_normalize_order_type` accepts optional context to disambiguate.
_IBKR_TO_CANONICAL: dict[str, str] = {
    "MKT": "market",
    "LMT": "limit",
    "STP": "stop_entry",
    "STP LMT": "stop_entry",
    "TRAIL": "trailing_stop",
    "TRAIL LIMIT": "trailing_stop",
    "MIDPRICE": "midprice",
    "MOC": "moc",
    "LOC": "loc",
    "REL": "relative",
    "SNAP MID": "snap_mid",
}

# Algo strategies that override the base IBKR type.
_ALGO_TO_CANONICAL: dict[str, str] = {
    "Adaptive": "adaptive",
    "Vwap": "vwap",
    "Twap": "twap",
}


def _normalize_order_type(
    raw: str,
    *,
    algo_strategy: str | None = None,
    tif: str | None = None,
    order_name: str | None = None,
) -> str:
    """Normalize IBKR order type to canonical name for consistent analysis.

    Priority: algo_strategy > tif-based special types > base mapping > pass-through.
    """
    # 1. Algo strategies override the base type entirely
    if algo_strategy and algo_strategy in _ALGO_TO_CANONICAL:
        return _ALGO_TO_CANONICAL[algo_strategy]
    # 2. tif='OPG' with MKT=moo, LMT=loo (opening-auction orders)
    if tif == "OPG":
        if raw == "MKT":
            return "moo"
        if raw == "LMT":
            return "loo"
    # 3. Standard IBKR type mapping
    return _IBKR_TO_CANONICAL.get(raw, raw)


def insert_execution_snapshot(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    intent: str,
    bid: float | None,
    ask: float | None,
    mid: float | None,
    spread: float | None,
    volume: int | None,
    atr: float | None,
    atr_pct: float | None,
    graduated_param_id: int | None = None,
    algo_strategy: str | None = None,
    order_tif: str | None = None,
) -> int | None:
    """Insert a snapshot at order submission time. Returns snapshot id."""
    # Normalize IBKR order type to canonical name using full context
    order_type = _normalize_order_type(
        order_type, algo_strategy=algo_strategy, tif=order_tif,
    )
    # Auto-consume pending graduated param if not explicitly provided
    if graduated_param_id is None:
        graduated_param_id = session_state.pending_graduated_params.pop(symbol, None)
    # Auto-consume pending order context (intent, atr_pct) from _plan_order
    _ctx = session_state.pending_order_context.pop(symbol, {})
    if intent == 'unknown' and 'intent' in _ctx:
        intent = _ctx['intent']
    if atr_pct is None and 'atr_pct' in _ctx:
        atr_pct = _ctx['atr_pct']
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO execution_snapshots
               (ts, symbol, side, quantity, order_type, intent,
                bid_at_submit, ask_at_submit, mid_at_submit, spread_at_submit,
                volume_at_submit, atr_at_submit,
                time_bucket, atr_bucket, graduated_param_id, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted')
               RETURNING id""",
            (
                ts, symbol, side, quantity, order_type, intent,
                bid, ask, mid, spread,
                volume, atr,
                _time_bucket(ts), _atr_bucket(atr_pct),
                graduated_param_id,
            ),
        )
        db.commit()
        row = cur.fetchone()
        return int(row["id"]) if row else None
    except Exception as e:
        logger.warning(f"Failed to insert execution snapshot: {e}")
        return None


def update_execution_snapshot_fill(
    snapshot_id: int,
    fill_price: float,
    fill_time: str,
    commission: float,
    partial_fills: int = 0,
) -> None:
    """Update a snapshot with fill data and compute derived metrics."""
    db = get_db()
    try:
        row = db.execute(
            "SELECT mid_at_submit, ts FROM execution_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if not row:
            return

        mid = row["mid_at_submit"]
        submit_ts = row["ts"]

        # Compute slippage in bps (positive = worse than mid)
        slippage_bps = None
        if mid and mid > 0:
            slippage_bps = round((fill_price - mid) / mid * 10000, 2)

        # Compute latency
        latency_ms = None
        try:
            from datetime import datetime as _dt
            submit_dt = _dt.fromisoformat(submit_ts)
            fill_dt = _dt.fromisoformat(fill_time)
            latency_ms = round((fill_dt - submit_dt).total_seconds() * 1000, 0)
        except Exception:
            pass

        db.execute(
            """UPDATE execution_snapshots
               SET fill_price = ?, fill_time = ?, commission = ?,
                   partial_fills = ?, slippage_bps = ?, latency_ms = ?,
                   status = 'filled'
               WHERE id = ?""",
            (fill_price, fill_time, commission, partial_fills, slippage_bps, latency_ms, snapshot_id),
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to update execution snapshot fill: {e}")


def cancel_execution_snapshot(snapshot_id: int) -> None:
    """Mark a snapshot as cancelled."""
    try:
        db = get_db()
        db.execute(
            "UPDATE execution_snapshots SET status = 'cancelled' WHERE id = ?",
            (snapshot_id,),
        )
        db.commit()
    except Exception:
        pass


def get_new_snapshot_count() -> int:
    """Count filled snapshots since last analysis run."""
    db = get_db()
    last_analysis = get_research_config("last_analysis_snapshot_id", 0.0)
    row = db.execute(
        "SELECT COUNT(*) as n FROM execution_snapshots WHERE status = 'filled' AND id > ?",
        (int(last_analysis),),
    ).fetchone()
    return row["n"] if row else 0


def get_filled_snapshots(since_id: int = 0, limit: int = 500) -> list[dict]:
    """Get filled snapshots since a given id."""
    db = get_db()
    rows = db.execute(
        """SELECT * FROM execution_snapshots
           WHERE status = 'filled' AND id > ?
           ORDER BY id ASC LIMIT ?""",
        (since_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_snapshots_for_param_review(param_id: int, activated_ts: str) -> dict:
    """Get fill stats before and after a graduated param was activated.

    Returns {"before": [slippage_bps, ...], "after": [slippage_bps, ...]}.
    The 'before' window matches the same (order_type, time_bucket, atr_bucket)
    from snapshots created before the param's activation timestamp.
    The 'after' window uses snapshots linked to this param_id or created after activation.
    """
    db = get_db()
    try:
        # Get the param's target buckets from its key
        param = db.execute(
            "SELECT param_key FROM graduated_params WHERE id = ?", (param_id,)
        ).fetchone()
        if not param:
            return {"before": [], "after": []}

        key = param["param_key"]
        parts = key.split(".")
        # Structured key: order_type.intent.time_bucket.atr_bucket
        if len(parts) >= 4:
            ot, intent, tb, ab = parts[0], parts[1], parts[2], parts[3]
        else:
            return {"before": [], "after": []}

        # Build WHERE clause for matching snapshots (before window)
        conditions = ["status = 'filled'", "slippage_bps IS NOT NULL"]
        params_list: list = []
        if ot != "all":
            conditions.append("order_type = ?")
            params_list.append(ot)
        if intent != "all":
            conditions.append("intent = ?")
            params_list.append(intent)
        if tb != "all":
            conditions.append("time_bucket = ?")
            params_list.append(tb)
        if ab != "all":
            conditions.append("atr_bucket = ?")
            params_list.append(ab)

        where = " AND ".join(conditions)

        # Before: matching snapshots before activation (same order type + buckets)
        before_rows = db.execute(
            f"SELECT slippage_bps FROM execution_snapshots WHERE {where} AND ts < ? ORDER BY id DESC LIMIT 50",
            (*params_list, activated_ts),
        ).fetchall()

        # After: only snapshots directly linked to this graduated param.
        # Bucket-only matching would introduce noise from unrelated orders
        # (e.g., exits or different order types in the same time/atr bucket).
        after_rows = db.execute(
            "SELECT slippage_bps FROM execution_snapshots "
            "WHERE graduated_param_id = ? AND status = 'filled' AND slippage_bps IS NOT NULL "
            "ORDER BY id DESC LIMIT 50",
            (param_id,),
        ).fetchall()

        return {
            "before": [abs(r["slippage_bps"]) for r in before_rows],
            "after": [abs(r["slippage_bps"]) for r in after_rows],
        }
    except Exception as e:
        logger.warning(f"Failed to get param review snapshots: {e}")
        return {"before": [], "after": []}


def upsert_calibrated_slippage(
    order_type: str,
    time_bucket: str,
    atr_bucket: str,
    median_bps: float,
    sample_count: int,
    p25_bps: float | None = None,
    p75_bps: float | None = None,
) -> None:
    """Upsert a calibrated slippage entry."""
    db = get_db()
    try:
        db.execute(
            """INSERT INTO calibrated_slippage
               (ts, order_type, time_bucket, atr_bucket, median_slippage_bps,
                sample_count, p25_bps, p75_bps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(order_type, time_bucket, atr_bucket) DO UPDATE
                 SET ts = excluded.ts,
                     median_slippage_bps = excluded.median_slippage_bps,
                     sample_count = excluded.sample_count,
                     p25_bps = excluded.p25_bps,
                     p75_bps = excluded.p75_bps""",
            (
                datetime.now(timezone.utc).isoformat(),
                order_type, time_bucket, atr_bucket,
                median_bps, sample_count, p25_bps, p75_bps,
            ),
        )
        db.commit()
        session_state.bump_calibration_version()
    except Exception as e:
        logger.warning(f"Failed to upsert calibrated slippage: {e}")


__all__ = [
    "_IV_HISTORY_MIN_SAMPLES",
    "_atr_bucket",
    "_time_bucket",
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

