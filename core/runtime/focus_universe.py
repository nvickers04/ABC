"""
Focus universe — the symbols the trader is actively engaged with.

Used by the research host to score "things the trader cares about
right now" every round, while the static base universe (research/config.py
RESEARCH_UNIVERSE) is sampled less frequently.

Source of truth:  ``attention_triggers`` table, state='active'.

The trader registers attention triggers for:
    * symbols it's currently holding
    * symbols it's watching for an entry/exit
    * symbols flagged from working_memory (auto-synced)

So the active-trigger set is a clean proxy for "trader interest right
now" without the research host having to open an IBKR connection of its own.

This module is pure read-only on the SQLite layer; no IBKR, no MDA.
"""

from __future__ import annotations

from typing import Iterable, List

from core.log_context import get_logger
from core.memory_config import get_memory_config

logger = get_logger(__name__)


def get_focus_symbols(conn, *, limit: int | None = None) -> List[str]:
    """Return up to ``limit`` distinct symbols with active attention triggers.

    Ordered by most-recently-created trigger first so newer interests
    take priority if we ever bump up against the limit.  Returns an
    empty list (never None) on any DB failure — the caller treats an
    empty focus set as "score base universe only."
    """
    if limit is None:
        limit = get_memory_config().focus_symbols_scorer_limit
    try:
        cur = conn.execute(
            "SELECT DISTINCT symbol FROM attention_triggers "
            "WHERE state='active' "
            "ORDER BY created_ts DESC "
            "LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall()
    except Exception as e:
        logger.debug("focus_universe: read failed: %s", e)
        return []

    out: List[str] = []
    seen: set[str] = set()
    for row in rows:
        sym = row[0] if not isinstance(row, dict) else row.get("symbol")
        if not isinstance(sym, str):
            continue
        s = sym.strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def merge_universes(
    base: Iterable[str],
    focus: Iterable[str],
    *,
    include_base: bool,
) -> List[str]:
    """Combine the base research universe with the focus set.

    When ``include_base`` is False, returns just the focus set (so the
    daemon can score focus-only on most rounds and pay base credits
    only every Nth round).  Order: focus first (newest interest first),
    then base symbols not already in focus.
    """
    seen: set[str] = set()
    out: List[str] = []
    for sym in focus:
        if not isinstance(sym, str):
            continue
        s = sym.strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    if include_base:
        for sym in base:
            if not isinstance(sym, str):
                continue
            s = sym.strip().upper()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out
