"""Trade-feedback / hypothesis repository.

Owns the ``trader_hypotheses`` table reads/writes. The legacy
``memory.__init__`` re-exports these names for back-compat so existing
callers (``from memory import get_open_hypotheses``) keep working.
"""

from __future__ import annotations

from memory.repos.schema import get_db


def get_open_hypotheses(slot: int | None = None, limit: int = 10) -> list[dict]:
    """Return open trader hypotheses, optionally filtered by slot."""
    db = get_db()
    if slot is not None:
        rows = db.execute(
            """SELECT hypothesis_type, description, suggested_action, priority, related_slot
               FROM trader_hypotheses
               WHERE status = 'open' AND (related_slot = ? OR related_slot IS NULL)
               ORDER BY priority ASC, ts DESC LIMIT ?""",
            (slot, limit),
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT hypothesis_type, description, suggested_action, priority, related_slot
               FROM trader_hypotheses
               WHERE status = 'open'
               ORDER BY priority ASC, ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_hypothesis_incorporated(hypothesis_type: str, description: str) -> None:
    """Mark matching open hypotheses as incorporated."""
    db = get_db()
    db.execute(
        """UPDATE trader_hypotheses SET status = 'incorporated', kept = 1
           WHERE status = 'open' AND hypothesis_type = ? AND description = ?""",
        (hypothesis_type, description),
    )
    db.commit()


__all__ = ["get_open_hypotheses", "mark_hypothesis_incorporated"]
