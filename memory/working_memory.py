"""
Working Memory — the trader's short-term structured monologue.

Holds *interpretations* (theses, verdicts, watch-fors, lessons), not
*observations* (prices, quotes — those are always fetched fresh).  The
agent writes here through tools; the curator drops expired entries at
the top of each cycle; the renderer produces the WORKING MEMORY block
injected into every prompt.

Five sections, each with a cap and a default expiry:

    | Section          | Cap | Default expiry  |
    |------------------|-----|-----------------|
    | open_theses      |  8  | EOD             |
    | recent_verdicts  | 12  | 30 min          |
    | watching_for     | 10  | 60 min          |
    | regime_notes     |  5  | EOD             |
    | lessons_today    |  8  | EOD             |

Eviction at cap: oldest-expires-first.  Persistence: the
``working_memory`` SQL table (created by ``memory.init_db``).  On
startup we restore today's still-valid entries; older rows are
considered stale and purged.

Design discipline (mirrors docs/PLAN_COGNITIVE_ARCHITECTURE.md §2):
- Tools never silently write to working memory; the agent writes
  through explicit tool calls.
- We do not cache market data here.
- No vector store, no embeddings — every entry renders verbatim.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ── Section configuration ───────────────────────────────────────

SECTIONS: tuple[str, ...] = (
    "open_theses",
    "recent_verdicts",
    "watching_for",
    "regime_notes",
    "lessons_today",
)

# Cap per section.  When exceeded, oldest-expires-first is evicted.
SECTION_CAPS: dict[str, int] = {
    "open_theses":     8,
    "recent_verdicts": 12,
    "watching_for":    10,
    "regime_notes":    5,
    "lessons_today":   8,
}

# Default expiry per section, in minutes.  Sentinel ``"EOD"`` means
# "end of local calendar day" (resolved at write time).  The agent can
# always override via ``expires_in_minutes``.
SECTION_DEFAULT_EXPIRY: dict[str, str | int] = {
    "open_theses":     "EOD",
    "recent_verdicts": 30,    # PASS verdicts; agent overrides for BUYs (2h)
    "watching_for":    60,
    "regime_notes":    "EOD",
    "lessons_today":   "EOD",
}


# ── Entry dataclass ─────────────────────────────────────────────

@dataclass
class WorkingMemoryEntry:
    id: int
    section: str
    entry_text: str
    created_ts: float
    expires_ts: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now_ts: float) -> bool:
        return self.expires_ts <= now_ts


# ── Helpers ─────────────────────────────────────────────────────

def _eod_ts(now_ts: float) -> float:
    """Return Unix ts of next local midnight (end-of-day boundary).

    EOD-default entries persist through the trading session and expire
    overnight when the day's interpretive context is no longer relevant.
    """
    now_dt = datetime.fromtimestamp(now_ts).astimezone()
    end_dt = now_dt.replace(hour=23, minute=59, second=59, microsecond=999_999)
    return end_dt.timestamp()


def _resolve_expiry_ts(
    section: str,
    expires_in_minutes: int | float | None,
    now_ts: float,
) -> float:
    """Resolve an expiry timestamp from explicit minutes, or section default."""
    if expires_in_minutes is not None and expires_in_minutes >= 0:
        return now_ts + float(expires_in_minutes) * 60.0
    default = SECTION_DEFAULT_EXPIRY.get(section, 30)
    if default == "EOD":
        return _eod_ts(now_ts)
    return now_ts + float(int(default)) * 60.0


def _validate_section(section: str) -> None:
    if section not in SECTIONS:
        raise ValueError(
            f"unknown working-memory section {section!r}; "
            f"valid: {sorted(SECTIONS)}"
        )


# ── WorkingMemory class ─────────────────────────────────────────

class WorkingMemory:
    """In-process working memory mirrored to SQLite.

    Thread-safe — every public method takes the instance lock.  The
    SQL table is the source of truth on restart; in-memory dict is the
    fast read path during a session.
    """

    def __init__(self, db_conn):
        self._db = db_conn
        self._lock = threading.RLock()
        # section -> list[WorkingMemoryEntry], oldest-first
        self._entries: dict[str, list[WorkingMemoryEntry]] = {
            s: [] for s in SECTIONS
        }

    # ── Public API ──────────────────────────────────────────────

    def add(
        self,
        section: str,
        entry: str,
        *,
        expires_in_minutes: int | float | None = None,
        metadata: dict[str, Any] | None = None,
        now_ts: float | None = None,
    ) -> int:
        """Append an entry; evict oldest-expires-first if at cap.

        Returns the assigned entry id.  ``entry`` is rendered verbatim
        in the prompt; keep it under ~200 tokens for ``open_theses``,
        ~80 for ``recent_verdicts``, etc. (caps in
        docs/PLAN_COGNITIVE_ARCHITECTURE.md §2).
        """
        _validate_section(section)
        text = (entry or "").strip()
        if not text:
            raise ValueError("working-memory entry must be non-empty")
        meta = dict(metadata or {})
        ts = float(now_ts if now_ts is not None else time.time())
        expires = _resolve_expiry_ts(section, expires_in_minutes, ts)

        with self._lock:
            try:
                cur = self._db.execute(
                    "INSERT INTO working_memory "
                    "(section, entry_text, created_ts, expires_ts, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (section, text, ts, expires, json.dumps(meta) if meta else None),
                )
                self._db.commit()
                entry_id = int(cur.lastrowid)
            except Exception as e:
                logger.debug("working_memory.add persistence failed: %s", e)
                # Generate a synthetic id so the in-memory store still works
                entry_id = -int(ts * 1000)

            wm_entry = WorkingMemoryEntry(
                id=entry_id,
                section=section,
                entry_text=text,
                created_ts=ts,
                expires_ts=expires,
                metadata=meta,
            )
            bucket = self._entries[section]
            bucket.append(wm_entry)
            self._enforce_cap(section)
            return entry_id

    def clear(
        self,
        section: str,
        entry_id: int | None = None,
    ) -> int:
        """Remove one entry (by id) or every entry in a section.

        Returns the count actually removed.
        """
        _validate_section(section)
        with self._lock:
            bucket = self._entries[section]
            if entry_id is None:
                removed = len(bucket)
                self._entries[section] = []
                try:
                    self._db.execute(
                        "DELETE FROM working_memory WHERE section = ?",
                        (section,),
                    )
                    self._db.commit()
                except Exception as e:
                    logger.debug("working_memory.clear(section) persistence failed: %s", e)
                return removed
            # Remove by id
            new_bucket = [e for e in bucket if e.id != entry_id]
            removed = len(bucket) - len(new_bucket)
            self._entries[section] = new_bucket
            if removed:
                try:
                    self._db.execute(
                        "DELETE FROM working_memory WHERE id = ?",
                        (entry_id,),
                    )
                    self._db.commit()
                except Exception as e:
                    logger.debug("working_memory.clear(id) persistence failed: %s", e)
            return removed

    def curate(self, now_ts: float | None = None) -> int:
        """Drop expired entries from memory + DB.  Returns drop count."""
        ts = float(now_ts if now_ts is not None else time.time())
        dropped_total = 0
        with self._lock:
            for section, bucket in self._entries.items():
                kept = [e for e in bucket if not e.is_expired(ts)]
                dropped = len(bucket) - len(kept)
                if dropped:
                    self._entries[section] = kept
                    dropped_total += dropped
            if dropped_total:
                try:
                    self._db.execute(
                        "DELETE FROM working_memory WHERE expires_ts <= ?", (ts,)
                    )
                    self._db.commit()
                except Exception as e:
                    logger.debug("working_memory.curate persistence failed: %s", e)
        return dropped_total

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON-friendly snapshot of all live entries."""
        out: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for section in SECTIONS:
                out[section] = [
                    {
                        "id": e.id,
                        "entry": e.entry_text,
                        "created_ts": e.created_ts,
                        "expires_ts": e.expires_ts,
                        "metadata": dict(e.metadata),
                    }
                    for e in self._entries[section]
                ]
        return out

    def render(self, *, now_ts: float | None = None) -> str:
        """Render the WORKING MEMORY prompt block.

        Empty sections are omitted entirely so the block stays compact
        early in the day.  If every section is empty we emit a single
        terse marker so the agent can see "memory is fresh."
        """
        ts = float(now_ts if now_ts is not None else time.time())
        lines: list[str] = ["WORKING MEMORY"]
        any_content = False
        with self._lock:
            for section in SECTIONS:
                live = [e for e in self._entries[section] if not e.is_expired(ts)]
                if not live:
                    continue
                any_content = True
                lines.append(f"  [{section}]")
                for e in live:
                    age_min = max(0, int((ts - e.created_ts) // 60))
                    lines.append(f"    - ({age_min}m ago) {e.entry_text}")
        if not any_content:
            lines.append("  (empty — fresh session or all entries expired)")
        return "\n".join(lines)

    # ── Persistence: restore on startup ─────────────────────────

    def restore_today(self, now_ts: float | None = None) -> int:
        """Load today's still-valid rows from SQL into the in-memory store.

        Older or already-expired rows are purged from the DB so the table
        doesn't accumulate dead history.  Returns the number of entries
        restored.
        """
        ts = float(now_ts if now_ts is not None else time.time())
        # Today-floor in local time — entries created before today are
        # considered stale even if their expiry hasn't passed yet.  This
        # matches the EOD-default semantic ("today's context only").
        today_floor = (
            datetime.fromtimestamp(ts).astimezone().replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
        )

        with self._lock:
            try:
                self._db.execute(
                    "DELETE FROM working_memory WHERE created_ts < ? OR expires_ts <= ?",
                    (today_floor, ts),
                )
                self._db.commit()
            except Exception as e:
                logger.debug("working_memory.restore_today purge failed: %s", e)

            try:
                cur = self._db.execute(
                    "SELECT id, section, entry_text, created_ts, expires_ts, metadata_json "
                    "FROM working_memory "
                    "WHERE created_ts >= ? AND expires_ts > ? "
                    "ORDER BY section, created_ts",
                    (today_floor, ts),
                )
                rows = cur.fetchall()
            except Exception as e:
                logger.debug("working_memory.restore_today read failed: %s", e)
                return 0

            for s in SECTIONS:
                self._entries[s] = []

            restored = 0
            for row in rows:
                section = str(row[1])
                if section not in SECTIONS:
                    continue
                try:
                    meta = json.loads(row[5]) if row[5] else {}
                except Exception:
                    meta = {}
                self._entries[section].append(
                    WorkingMemoryEntry(
                        id=int(row[0]),
                        section=section,
                        entry_text=str(row[2]),
                        created_ts=float(row[3]),
                        expires_ts=float(row[4]),
                        metadata=meta,
                    )
                )
                restored += 1
            # Defensive: enforce caps after bulk restore
            for s in SECTIONS:
                self._enforce_cap(s)
            return restored

    # ── Internals ───────────────────────────────────────────────

    def _enforce_cap(self, section: str) -> None:
        """Keep section length <= cap by evicting oldest-expires-first."""
        cap = SECTION_CAPS.get(section, 0)
        bucket = self._entries[section]
        if len(bucket) <= cap:
            return
        # Sort by expires_ts ascending; drop the front until we hit cap
        bucket.sort(key=lambda e: e.expires_ts)
        to_drop = bucket[: len(bucket) - cap]
        kept = bucket[len(bucket) - cap :]
        # Re-sort kept by created_ts ascending so render order stays
        # chronological (oldest at top within section).
        kept.sort(key=lambda e: e.created_ts)
        self._entries[section] = kept
        if to_drop:
            try:
                self._db.executemany(
                    "DELETE FROM working_memory WHERE id = ?",
                    [(e.id,) for e in to_drop],
                )
                self._db.commit()
            except Exception as e:
                logger.debug("working_memory eviction persistence failed: %s", e)


# ── Singleton accessor ──────────────────────────────────────────

_singleton: WorkingMemory | None = None
_singleton_lock = threading.Lock()


def get_working_memory() -> WorkingMemory:
    """Return the process-wide WorkingMemory bound to memory.get_db().

    Lazy-creates on first call and restores today's entries from disk.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:  # double-checked
            return _singleton
        from memory import get_db  # avoid circular import at module load
        wm = WorkingMemory(get_db())
        try:
            wm.restore_today()
        except Exception as e:
            logger.debug("WorkingMemory.restore_today failed at init: %s", e)
        _singleton = wm
        return _singleton


def reset_working_memory_for_tests() -> None:
    """Drop the singleton (test isolation only)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
