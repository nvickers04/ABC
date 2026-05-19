"""QualityMatrix provenance persistence (tool usage + decision snapshots)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from memory.repos.schema import get_db

logger = logging.getLogger(__name__)


def insert_tool_usage_log(
    tool_name: str,
    symbol: str | None = None,
    success: bool = True,
    latency_ms: float = 0.0,
    cycle_id: int = 0,
    decision_context: str | None = None,
) -> int | None:
    """Insert a tool usage record. Returns new id or None."""
    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO tool_usage_log
               (ts, cycle_id, tool_name, symbol, success, latency_ms, decision_context)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               RETURNING id""",
            (
                datetime.now(timezone.utc).isoformat(),
                cycle_id,
                tool_name,
                symbol,
                1 if success else 0,
                latency_ms,
                decision_context,
            ),
        )
        db.commit()
        row = cur.fetchone()
        return int(row["id"]) if row else None
    except Exception as e:
        logger.warning(f"insert_tool_usage_log failed: {e}")
        return None


def insert_decision_provenance(
    decision_type: str,
    cycle_id: int = 0,
    symbol: str | None = None,
    tools_json: str | None = None,
    quality_state_json: str | None = None,
    context_quality: str = "full",
    outcome: str | None = None,
    notes: str = "",
) -> int | None:
    """Insert a decision provenance snapshot."""
    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO decision_provenance
               (ts, cycle_id, decision_type, symbol, tools_json, quality_state_json,
                context_quality, outcome, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               RETURNING id""",
            (
                datetime.now(timezone.utc).isoformat(),
                cycle_id,
                decision_type,
                symbol,
                tools_json,
                quality_state_json,
                context_quality,
                outcome,
                notes,
            ),
        )
        db.commit()
        row = cur.fetchone()
        return int(row["id"]) if row else None
    except Exception as e:
        logger.warning(f"insert_decision_provenance failed: {e}")
        return None


def get_recent_tool_usage(limit: int = 50) -> list[dict]:
    db = get_db()
    try:
        rows = db.execute(
            "SELECT * FROM tool_usage_log ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_recent_decision_provenance(limit: int = 20) -> list[dict]:
    db = get_db()
    try:
        rows = db.execute(
            "SELECT * FROM decision_provenance ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


__all__ = [
    "get_recent_decision_provenance",
    "get_recent_tool_usage",
    "insert_decision_provenance",
    "insert_tool_usage_log",
]
