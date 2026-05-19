"""
Resolve the active Working Memory store for the current operating context.

Postgres-backed WorkingMemory is the default when the researcher path is healthy.
Local JSON fallback is used in Independent Mode or when Postgres WM is unreachable.

All agent and tool write/read paths should use get_active_working_memory() so
routing stays consistent.

Recovery policy (researcher reconnect): see docs/policy-independent-mode-memory.md
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.runtime.operating_context import get_operating_context
from core.runtime.local_memory_fallback import get_local_working_memory

logger = logging.getLogger(__name__)


def get_active_working_memory(*, prefer_local: bool = False) -> Any:
    """Return the WM store appropriate for the current trader context.

    ``prefer_local`` forces the JSON fallback (used in tests).
    """
    ctx = get_operating_context()
    if prefer_local or ctx.is_independent_mode:
        return get_local_working_memory()

    try:
        from memory.working_memory import get_working_memory

        return get_working_memory()
    except Exception as exc:
        logger.warning("Postgres working memory unavailable, using local fallback: %s", exc)
        ctx.set_researcher_unavailable()
        return get_local_working_memory()


def count_live_wm_entries(wm: Any, *, now_ts: float | None = None) -> dict[str, Any]:
    """Count non-expired entries per section for any WM store with snapshot() or get_all()."""
    from memory.working_memory import SECTIONS

    now = float(now_ts if now_ts is not None else time.time())
    by_section: dict[str, int] = {}
    total = 0

    for section in SECTIONS:
        entries: list[Any] = []
        if hasattr(wm, "snapshot"):
            try:
                snap = wm.snapshot()
                raw = snap.get(section, []) if isinstance(snap, dict) else []
                entries = list(raw) if isinstance(raw, list) else []
            except Exception:
                entries = []
        elif hasattr(wm, "get_all"):
            try:
                entries = list(wm.get_all(section))
            except Exception:
                entries = []

        live = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            exp = e.get("expires_ts", 0)
            try:
                if float(exp) > now:
                    live += 1
            except (TypeError, ValueError):
                pass
        by_section[section] = live
        total += live

    return {"total": total, "by_section": by_section}


def summarize_wm_stores() -> dict[str, Any]:
    """Snapshot live entry counts for local JSON and Postgres WM (read-only)."""
    summary: dict[str, Any] = {
        "policy": "postgres_wins_no_merge",
        "local": {"total": 0, "by_section": {}},
        "postgres": {"total": 0, "by_section": {}},
    }

    try:
        local_wm = get_local_working_memory()
        summary["local"] = count_live_wm_entries(local_wm)
    except Exception as exc:
        summary["local"] = {"total": 0, "by_section": {}, "error": str(exc)}

    try:
        from memory.working_memory import get_working_memory

        summary["postgres"] = count_live_wm_entries(get_working_memory())
    except Exception as exc:
        summary["postgres"] = {"total": 0, "by_section": {}, "error": str(exc)}

    return summary


def log_wm_recovery_on_reconnect(*, had_local_fallback: bool) -> dict[str, Any]:
    """Log WM state when the researcher becomes available again (Option A: no merge).

    Called from OperatingContext.set_researcher_available(). Local entries written
    during Independent Mode remain in data/local_working_memory.json; Postgres is
    authoritative for new writes. See docs/policy-independent-mode-memory.md.
    """
    summary = summarize_wm_stores()
    local_n = int(summary["local"].get("total", 0))
    pg = summary["postgres"]
    pg_n = int(pg.get("total", 0)) if "error" not in pg else -1

    if had_local_fallback and local_n > 0:
        logger.info(
            "WM recovery: researcher back online | policy=postgres_wins_no_merge | "
            "local_live=%d postgres_live=%s | local archived at data/local_working_memory.json "
            "(not auto-merged) | local_sections=%s",
            local_n,
            pg_n if pg_n >= 0 else "unavailable",
            summary["local"].get("by_section"),
        )
    elif had_local_fallback:
        logger.info(
            "WM recovery: researcher back online | policy=postgres_wins_no_merge | "
            "local_live=0 | postgres_live=%s",
            pg_n if pg_n >= 0 else "unavailable",
        )
    else:
        logger.debug(
            "WM recovery: researcher available (no prior local fallback) | postgres_live=%s",
            pg_n if pg_n >= 0 else "unavailable",
        )

    summary["had_local_fallback"] = had_local_fallback
    return summary
