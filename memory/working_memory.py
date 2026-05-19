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

Design discipline:
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
from datetime import datetime, timezone
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


# ── Quality metadata defaults (for per-entry tracking + section trust) ──

DEFAULT_SECTION_SCORES: dict[str, dict[str, Any]] = {
    "lessons_today": {"score": 0.92, "last_updated": None, "sample_size": 0},
    "open_theses":   {"score": 0.70, "last_updated": None, "sample_size": 0},
    "watching_for":  {"score": 0.78, "last_updated": None, "sample_size": 0},
    "regime_notes":  {"score": 0.75, "last_updated": None, "sample_size": 0},
    "recent_verdicts": {"score": 0.85, "last_updated": None, "sample_size": 0},
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
        # Top-level section trust aggregates (longer-lived than daily entries)
        self._section_scores: dict[str, dict[str, Any]] = {}
        self.ensure_section_scores()  # populate defaults immediately

    def ensure_section_scores(self) -> dict[str, Any]:
        """Instance method: ensure _section_scores exist with defaults on this WM.

        Idempotent.  Returns the live _section_scores dict (mutatable by caller
        or via the module-level update_section_score).
        """
        if not self._section_scores:
            self._section_scores = {k: v.copy() for k, v in DEFAULT_SECTION_SCORES.items()}
            now = datetime.now(timezone.utc).isoformat()
            for section in self._section_scores:
                self._section_scores[section]["last_updated"] = now
        return self._section_scores

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
        ~80 for ``recent_verdicts``, etc. (see ``SECTION_CAPS``).
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
                    "VALUES (?, ?, ?, ?, ?) "
                    "RETURNING id",
                    (section, text, ts, expires, json.dumps(meta) if meta else None),
                )
                self._db.commit()
                row = cur.fetchone()
                entry_id = int(row["id"]) if row else -int(ts * 1000)
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
        """Return a JSON-friendly snapshot of all live entries.

        The returned dict now also contains the top-level "_section_scores"
        key (for quality tracking integration).  Callers that only access
        known section keys remain unaffected.
        """
        out: dict[str, Any] = {}
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
            out["_section_scores"] = {
                k: dict(v) for k, v in self._section_scores.items()
            }
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
            # Ensure quality scores survive (they are not daily-scoped)
            self.ensure_section_scores()
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


# ── Quality scoring helpers (per-entry metadata + section-level trust) ──
#
# These enable outcome-feedback paths to track reliability of WM content.
# Per-entry fields live *inside* each entry's existing "metadata" dict for
# 100% backward compatibility with legacy entries (pre-quality fields).
# Old entries receive sensible defaults on first reinforce/mark.
#
# The _section_scores live as a top-level key alongside the 5 section arrays
# in snapshot() output and in the local fallback JSON.  Section scores are
# long-lived (survive daily curate/restore); entry metadata is per-utterance.
#
# Typical usage from feedback:
#   entry = ... from snapshot ...
#   reinforce_wm_entry(entry, success=True)
#   update_section_score(scores, entry_section, success=True)
#


def ensure_section_scores(wm: dict[str, Any]) -> dict[str, Any]:
    """Guarantees the _section_scores structure exists with defaults.

    Operates on a dict representation of Working Memory (e.g. snapshot()
    result or the internal structure of LocalWorkingMemoryStore).  Safe to
    call repeatedly.  Initializes last_updated to now on first creation.

    Returns the (possibly newly created) _section_scores sub-dict.
    """
    if "_section_scores" not in wm or not isinstance(wm.get("_section_scores"), dict):
        wm["_section_scores"] = {k: v.copy() for k, v in DEFAULT_SECTION_SCORES.items()}
        now = datetime.now(timezone.utc).isoformat()
        for section in wm["_section_scores"]:
            wm["_section_scores"][section]["last_updated"] = now
    return wm["_section_scores"]


def create_wm_entry(
    text: str,
    confidence: float = 0.85,
    source: str = "agent",
    tags: Optional[list[str]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create a quality-aware metadata payload for a new WM entry.

    Returns a dict intended to be passed via the existing `metadata=`
    parameter to WorkingMemory.add() / update_working_memory tool.

    All quality fields are placed inside the caller's metadata dict so that
    the on-disk / snapshot shape (id, entry_text, created_ts, expires_ts, metadata)
    is unchanged.  Legacy entries without these keys are fully supported.

    The `text` arg is accepted for caller convenience / documentation but is
    *not* included in the return (text belongs in the `entry` param of add()).

    Example:
        meta = create_wm_entry("NVDA thesis...", confidence=0.9, tags=["earnings"])
        wm.add("open_theses", "NVDA thesis...", metadata=meta)
    """
    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "created_at": now,
        "last_used": now,
        "last_reinforced": now,
        "success_count": 0,
        "failure_count": 0,
        "confidence": max(0.5, min(1.0, float(confidence))),
        "source": source or "agent",
        "tags": list(tags) if tags else [],
        "version": 1,
    }
    payload.update(extra)  # allow symbol, condition, etc. to coexist
    return payload


def _ensure_quality_meta(meta: Any) -> dict[str, Any]:
    """Internal: coerce/return a dict for the quality fields, with defaults."""
    if not isinstance(meta, dict):
        meta = {}
    now = datetime.now(timezone.utc).isoformat()
    meta.setdefault("created_at", now)
    meta.setdefault("last_used", now)
    meta.setdefault("last_reinforced", now)
    meta.setdefault("success_count", 0)
    meta.setdefault("failure_count", 0)
    meta.setdefault("confidence", 0.85)
    meta.setdefault("source", "unknown")
    meta.setdefault("tags", [])
    meta.setdefault("version", 1)
    return meta


def reinforce_wm_entry(entry: dict[str, Any], success: bool) -> dict[str, Any]:
    """Record outcome for an existing WM entry (from .snapshot() or local store).

    Updates success/failure counts and last_reinforced inside entry["metadata"].
    Returns a (shallow) copy of the entry with the mutated metadata so callers
    can persist if the store requires explicit write-back.

    Old entries without quality keys receive defaults on first call.
    Safe to call multiple times; does not touch the rendered entry_text.
    """
    entry = dict(entry)  # top-level copy
    meta = entry.get("metadata")
    meta = _ensure_quality_meta(meta)
    entry["metadata"] = meta

    now = datetime.now(timezone.utc).isoformat()
    if success:
        meta["success_count"] = int(meta.get("success_count", 0)) + 1
    else:
        meta["failure_count"] = int(meta.get("failure_count", 0)) + 1
    meta["last_reinforced"] = now
    return entry


def mark_wm_entry_used(entry: dict[str, Any]) -> dict[str, Any]:
    """Mark that this WM entry was read/used in a decision or reasoning step.

    Updates only last_used inside the metadata.  Returns a copy for the same
    reasons as reinforce_wm_entry().
    """
    entry = dict(entry)
    meta = entry.get("metadata")
    meta = _ensure_quality_meta(meta)
    entry["metadata"] = meta
    meta["last_used"] = datetime.now(timezone.utc).isoformat()
    return entry


def update_section_score(
    section_scores: dict[str, Any],
    section: str,
    success: bool,
    decay_days: int = 14,
) -> None:
    """Exponentially update a section's aggregate trust score with time decay.

    Mutates `section_scores` (the value of _section_scores[section]) in place.
    This is the primitive called by outcome feedback loops after a decision
    that consumed content from that WM section produced a measurable result.

    Formula: EMA (alpha=0.12) blended with success (1.0/0.0), then gentle
    multiplicative decay based on days since last update (floor 0.55 after 14d).
    Score clamped [0.30, 1.00].  sample_size incremented on every call.
    """
    if not isinstance(section_scores, dict):
        return
    if section not in section_scores or not isinstance(section_scores[section], dict):
        section_scores[section] = {"score": 0.70, "last_updated": None, "sample_size": 0}

    data = section_scores[section]
    now = datetime.now(timezone.utc)
    alpha = 0.12
    current = float(data.get("score", 0.70))
    target = 1.0 if success else 0.0
    new_score = (1.0 - alpha) * current + alpha * target

    # Time decay
    last_str = data.get("last_updated")
    if last_str:
        try:
            last = datetime.fromisoformat(last_str)
            days = max(0, (now - last).days)
            decay = max(0.55, 1.0 - (days / float(decay_days)) * 0.45)
            new_score *= decay
        except Exception:
            pass  # ignore bad timestamps

    data["score"] = round(max(0.30, min(1.0, new_score)), 4)
    data["last_updated"] = now.isoformat()
    data["sample_size"] = int(data.get("sample_size", 0)) + 1
