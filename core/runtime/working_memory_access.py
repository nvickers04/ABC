"""Working memory store routing (Postgres vs local JSON fallback).

All trader WM reads/writes should go through :func:`get_active_working_memory` so
Independent Mode and Postgres outages route consistently.

Routing rules:

* **Postgres** — default when :class:`~core.runtime.operating_context.OperatingContext`
  reports researcher available and ``memory.working_memory`` is reachable.
* **Local JSON** — ``data/local_working_memory.json`` via
  :class:`~core.runtime.local_memory_fallback.LocalWorkingMemoryStore` when in
  Independent Mode or Postgres fails (then context flips to unavailable).

Recovery on researcher reconnect: Option A (no merge) — see
``docs/operations/independent-mode.md``.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

from core.log_context import get_logger
from core.runtime.operating_context import get_operating_context

logger = get_logger(__name__)


@runtime_checkable
class WorkingMemoryStore(Protocol):
    """Minimal WM surface used by tools and the agent loop.

    Implemented by :class:`memory.working_memory.WorkingMemory` (Postgres) and
    :class:`~core.runtime.local_memory_fallback.LocalWorkingMemoryStore` (JSON).
    """

    def add(
        self,
        section: str,
        entry: str,
        *,
        expires_in_minutes: int | float | None = None,
        metadata: dict[str, Any] | None = None,
        now_ts: float | None = None,
    ) -> int:
        """Append an entry; return new entry id."""
        ...

    def clear(self, section: str, entry_id: int | None = None) -> int:
        """Remove one entry or clear a section; return count removed."""
        ...

    def get_all(self, section: str) -> list[dict[str, Any]]:
        """Return all entries in a section (including expired unless curated)."""
        ...

    def curate(self) -> int:
        """Drop expired entries; return count removed."""
        ...

    def render(
        self,
        *,
        now_ts: float | None = None,
        max_entries_per_section: int | None = None,
    ) -> str:
        """Render WM text for prompts."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable state (sections + metadata)."""
        ...


class WmSectionCounts(TypedDict, total=False):
    """Live entry counts for one WM backend."""

    total: int
    by_section: dict[str, int]
    error: str


class WmStoreSummary(TypedDict, total=False):
    """Result of :func:`summarize_wm_stores`."""

    policy: str
    local: WmSectionCounts
    postgres: WmSectionCounts
    had_local_fallback: bool


def get_active_working_memory(*, prefer_local: bool = False) -> WorkingMemoryStore:
    """Return the WM store for the current trader operating context.

    Args:
        prefer_local: When True, force the JSON fallback (tests).

    Returns:
        A :class:`WorkingMemoryStore` — Postgres-backed or local JSON.

    Side effects:
        On Postgres failure, marks researcher unavailable via
        :meth:`~core.runtime.operating_context.OperatingContext.set_researcher_unavailable`.
    """
    from core.runtime.local_memory_fallback import get_local_working_memory

    ctx = get_operating_context()
    if prefer_local or ctx.is_independent_mode:
        return get_local_working_memory()

    try:
        from memory.working_memory import get_working_memory

        return cast(WorkingMemoryStore, get_working_memory())
    except Exception as exc:
        logger.warning("Postgres working memory unavailable, using local fallback: %s", exc)
        ctx.set_researcher_unavailable()
        return get_local_working_memory()


def count_live_wm_entries(
    wm: WorkingMemoryStore,
    *,
    now_ts: float | None = None,
) -> WmSectionCounts:
    """Count non-expired entries per section for any compatible WM store.

    Args:
        wm: Store implementing :class:`WorkingMemoryStore`.
        now_ts: Optional epoch seconds; defaults to ``time.time()``.

    Returns:
        Dict with ``total`` and ``by_section`` keys.
    """
    from memory.working_memory import SECTIONS

    now = float(now_ts if now_ts is not None else time.time())
    by_section: dict[str, int] = {}
    total = 0

    for section in SECTIONS:
        entries: list[dict[str, Any]] = []
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


def summarize_wm_stores() -> WmStoreSummary:
    """Compare live entry counts in local JSON vs Postgres WM (read-only).

    Returns:
        Summary dict with ``policy``, ``local``, and ``postgres`` counts.
        Postgres section may include ``error`` when unreachable.
    """
    summary: WmStoreSummary = {
        "policy": "postgres_wins_no_merge",
        "local": {"total": 0, "by_section": {}},
        "postgres": {"total": 0, "by_section": {}},
    }

    from core.runtime.local_memory_fallback import get_local_working_memory

    try:
        local_wm = get_local_working_memory()
        summary["local"] = count_live_wm_entries(local_wm)
    except Exception as exc:
        summary["local"] = {"total": 0, "by_section": {}, "error": str(exc)}

    try:
        from memory.working_memory import get_working_memory

        summary["postgres"] = count_live_wm_entries(
            cast(WorkingMemoryStore, get_working_memory())
        )
    except Exception as exc:
        summary["postgres"] = {"total": 0, "by_section": {}, "error": str(exc)}

    return summary


def log_wm_recovery_on_reconnect(*, had_local_fallback: bool) -> WmStoreSummary:
    """Log WM state when the research host becomes available again.

    Called from :meth:`~core.runtime.operating_context.OperatingContext.set_researcher_available`.
    Local entries written during Independent Mode are **not** auto-merged into Postgres.

    Args:
        had_local_fallback: True when the trader had been on local JSON WM.

    Returns:
        The same structure as :func:`summarize_wm_stores`, plus ``had_local_fallback``.
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
