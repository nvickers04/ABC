"""Research-config / graduated-params repository.

Owns the SQL for the ``research_config`` and ``graduated_params``
tables. The legacy ``memory.__init__`` re-exports these names via thin
shims so existing callers (``from memory import get_research_config``)
keep working byte-for-byte.

The monotonic ``calibration_version`` counter is intentionally kept in
``memory.__init__`` — it is mutated by ``upsert_calibrated_slippage``
(execution domain) and read here, so collocating it with one side
would have been arbitrary. ``get_calibration_version`` is a thin
re-read of that module-level int.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from memory.repos.schema import get_db

logger = logging.getLogger(__name__)


# ── research_config ─────────────────────────────────────────────


def get_research_config(key: str, default: float) -> float:
    """Read a tunable config value.  Returns DB override if set, else default."""
    row = get_db().execute(
        "SELECT value FROM research_config WHERE key = ?", (key,)
    ).fetchone()
    return float(row["value"]) if row else default


def set_research_config(key: str, value: float, reason: str = "", *, log: bool = True) -> None:
    """Write (upsert) a tunable config value with an audit trail."""
    ts = datetime.now(timezone.utc).isoformat()
    get_db().execute(
        """INSERT INTO research_config (key, value, updated_ts, reason)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(key) DO UPDATE
             SET value = excluded.value,
                 updated_ts = excluded.updated_ts,
                 reason = excluded.reason""",
        (key, value, ts, reason),
    )
    get_db().commit()
    if log:
        logger.info(f"research_config[{key}] = {value}  ({reason})")


def get_all_research_config() -> dict[str, float]:
    """Return all DB-stored tunable config values as {key: value}."""
    rows = get_db().execute("SELECT key, value FROM research_config").fetchall()
    return {r["key"]: float(r["value"]) for r in rows}


# ── graduated_params ────────────────────────────────────────────

# Param key schema — structured: {order_type}.{intent}.{time_bucket}.{atr_bucket}
_VALID_ORDER_TYPES = {
    "market", "limit", "stop_entry", "bracket", "trailing_stop",
    "oca_exit", "midprice", "adaptive", "vwap", "twap",
    "relative", "snap_mid", "moc", "moo", "loc", "loo", "all",
}
_VALID_INTENTS = {"entry", "exit", "stop", "all"}
_VALID_TIME_BUCKETS = {"open", "morning", "midday", "close", "extended", "all"}
_VALID_ATR_BUCKETS = {"low", "medium", "high", "all"}


def validate_param_key(key: str) -> str | None:
    """Validate a graduated param key matches the structured schema.

    Returns None if valid, or an error message if invalid.
    """
    parts = key.split(".")
    if len(parts) != 4:
        return f"Expected 4 dot-separated parts, got {len(parts)}: {key}"
    ot, intent, tb, ab = parts
    if ot not in _VALID_ORDER_TYPES:
        return f"Invalid order_type '{ot}' in key: {key}"
    if intent not in _VALID_INTENTS:
        return f"Invalid intent '{intent}' in key: {key}"
    if tb not in _VALID_TIME_BUCKETS:
        return f"Invalid time_bucket '{tb}' in key: {key}"
    if ab not in _VALID_ATR_BUCKETS:
        return f"Invalid atr_bucket '{ab}' in key: {key}"
    return None


def get_graduated_params(active_only: bool = True) -> list[dict]:
    """Get graduated parameter overrides."""
    db = get_db()
    where = "WHERE active = 1" if active_only else ""
    rows = db.execute(
        f"SELECT * FROM graduated_params {where} ORDER BY ts DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def deactivate_graduated_param(param_id: int, reason: str) -> None:
    """Deactivate a graduated param and record why."""
    db = get_db()
    try:
        db.execute(
            "UPDATE graduated_params SET active = 0, rollback_reason = ? WHERE id = ?",
            (reason, param_id),
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to deactivate graduated param {param_id}: {e}")


def insert_graduated_param(
    param_key: str,
    param_value: str,
    previous_value: str | None,
    evidence_json: str,
    snapshots_analyzed: int,
    improvement_bps: float,
    p_value: float,
) -> int | None:
    """Insert a new graduated parameter override."""
    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO graduated_params
               (ts, param_key, param_value, previous_value, evidence_json,
                snapshots_analyzed, improvement_bps, p_value, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
               RETURNING id""",
            (
                datetime.now(timezone.utc).isoformat(),
                param_key, param_value, previous_value, evidence_json,
                snapshots_analyzed, improvement_bps, p_value,
            ),
        )
        db.commit()
        row = cur.fetchone()
        return int(row["id"]) if row else None
    except Exception as e:
        logger.warning(f"Failed to insert graduated param: {e}")
        return None


# ── calibration version (state owned by memory/__init__) ───────


def get_calibration_version() -> int:
    """Return current calibration version (monotonic counter).

    The counter itself is a module-level int in ``memory.__init__``,
    mutated by ``upsert_calibrated_slippage``. Read it through the module
    so we always see the latest value rather than capturing a stale ref.
    """
    import memory as _memory
    return _memory._calibration_version


__all__ = [
    "deactivate_graduated_param",
    "get_all_research_config",
    "get_calibration_version",
    "get_graduated_params",
    "get_research_config",
    "insert_graduated_param",
    "set_research_config",
    "validate_param_key",
]
