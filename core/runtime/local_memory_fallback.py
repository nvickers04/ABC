"""Local JSON fallback for working memory (Independent Mode).

When Postgres is unreachable, :func:`~core.runtime.working_memory_access.get_active_working_memory`
returns :class:`LocalWorkingMemoryStore`, which persists the five WM sections to
``data/local_working_memory.json`` so theses and lessons survive trader restarts.

The public API mirrors :mod:`memory.working_memory` methods used by tools and the agent.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from memory.working_memory import (
    DEFAULT_SECTION_SCORES,
    SECTION_CAPS,
    SECTIONS,
    _resolve_expiry_ts,
    _validate_section,
)

LOCAL_MEMORY_FILE = Path("data/local_working_memory.json")
LOCAL_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)


class LocalWorkingMemoryStore:
    """Thread-safe JSON-backed store for the five working memory sections.

    Implements :class:`~core.runtime.working_memory_access.WorkingMemoryStore`.
    Entries are capped per section using the same limits as Postgres WM.

    Attributes:
        filepath: On-disk JSON path (default ``data/local_working_memory.json``).
    """

    def __init__(self, filepath: Path = LOCAL_MEMORY_FILE) -> None:
        """Load existing JSON state or start empty with default section scores.

        Args:
            filepath: Persistence path; parent directories are created at module import.
        """
        self.filepath = filepath
        self._lock = threading.RLock()
        self._entries: dict[str, list[dict[str, Any]]] = {s: [] for s in SECTIONS}
        self._section_scores: dict[str, dict[str, Any]] = {}
        self._load()

    def ensure_section_scores(self) -> dict[str, dict[str, Any]]:
        """Ensure section score metadata exists with defaults.

        Returns:
            Mutable ``_section_scores`` dict keyed by section name.
        """
        if not self._section_scores or not isinstance(self._section_scores, dict):
            self._section_scores = {k: v.copy() for k, v in DEFAULT_SECTION_SCORES.items()}
            now = time.time()
            for section in self._section_scores:
                self._section_scores[section]["last_updated"] = now
        return self._section_scores

    def _load(self) -> None:
        """Load sections and scores from disk; corrupt files are ignored."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for section in SECTIONS:
                    self._entries[section] = data.get(section, [])
                self._section_scores = data.get("_section_scores", {}) or {}
            except Exception:
                pass
        self.ensure_section_scores()

    def _save(self) -> None:
        """Persist current entries and section scores to disk (best-effort)."""
        try:
            payload: dict[str, Any] = {s: self._entries.get(s, []) for s in SECTIONS}
            payload["_section_scores"] = self._section_scores or {}
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception:
            pass

    def _enforce_cap(self, section: str) -> None:
        """Drop oldest-expiring entries when a section exceeds its cap."""
        cap = SECTION_CAPS.get(section, 20)
        bucket = self._entries[section]
        while len(bucket) > cap:
            bucket.sort(key=lambda e: e.get("expires_ts", 0))
            bucket.pop(0)

    def add(
        self,
        section: str,
        entry: str,
        *,
        expires_in_minutes: int | float | None = None,
        metadata: dict[str, Any] | None = None,
        now_ts: float | None = None,
    ) -> int:
        """Append an entry; evict oldest-expires-first when at cap.

        Args:
            section: One of the canonical WM section names.
            entry: Non-empty text body.
            expires_in_minutes: Optional TTL override.
            metadata: Optional JSON-serializable metadata.
            now_ts: Optional epoch seconds for deterministic tests.

        Returns:
            New entry id (millisecond timestamp).

        Raises:
            ValueError: If ``entry`` is empty after strip.
        """
        _validate_section(section)
        text = (entry or "").strip()
        if not text:
            raise ValueError("working-memory entry must be non-empty")

        ts = float(now_ts if now_ts is not None else time.time())
        expires = _resolve_expiry_ts(section, expires_in_minutes, ts)
        entry_id = int(ts * 1000)

        record: dict[str, Any] = {
            "id": entry_id,
            "entry_text": text,
            "created_ts": ts,
            "expires_ts": expires,
            "metadata": dict(metadata or {}),
        }

        with self._lock:
            self._entries[section].append(record)
            self._enforce_cap(section)
            self._save()
        return entry_id

    def clear(self, section: str, entry_id: int | None = None) -> int:
        """Remove one entry by id, or clear the whole section.

        Args:
            section: WM section name.
            entry_id: When set, remove only that id; otherwise clear the section.

        Returns:
            Count of entries removed.
        """
        _validate_section(section)
        with self._lock:
            bucket = self._entries[section]
            if entry_id is None:
                removed = len(bucket)
                self._entries[section] = []
            else:
                new_bucket = [e for e in bucket if e.get("id") != entry_id]
                removed = len(bucket) - len(new_bucket)
                self._entries[section] = new_bucket
            self._save()
            return removed

    def get_all(self, section: str) -> list[dict[str, Any]]:
        """Return a copy of all entries in a section (including expired).

        Args:
            section: WM section name.

        Returns:
            List of entry dicts.
        """
        with self._lock:
            return list(self._entries.get(section, []))

    def curate(self) -> int:
        """Remove expired entries from all sections.

        Returns:
            Total count removed across sections.
        """
        now = time.time()
        removed = 0
        with self._lock:
            for section in SECTIONS:
                before = len(self._entries[section])
                self._entries[section] = [
                    e for e in self._entries[section] if e.get("expires_ts", 0) > now
                ]
                removed += before - len(self._entries[section])
            self._save()
        return removed

    def render(
        self,
        *,
        now_ts: float | None = None,
        max_entries_per_section: int | None = None,
    ) -> str:
        """Render WM text for prompts (abbreviated vs Postgres renderer).

        Args:
            now_ts: Optional epoch seconds; defaults to ``time.time()``.
            max_entries_per_section: Keep only the newest N entries per section.

        Returns:
            Markdown-style block or a placeholder when empty.
        """
        now = float(now_ts if now_ts is not None else time.time())
        cap = max_entries_per_section if max_entries_per_section is not None else 5
        lines = ["═══ WORKING MEMORY (local fallback) ═══"]
        any_content = False
        for section in SECTIONS:
            entries = [
                e for e in self.get_all(section)
                if e.get("expires_ts", 0) > now
            ]
            if not entries:
                continue
            any_content = True
            lines.append(f"**{section.upper()}**")
            show = entries[-cap:] if cap > 0 else entries
            for e in show:
                text = str(e.get("entry_text", ""))[:300]
                lines.append(f"- {text}")
            if len(entries) > len(show):
                lines.append(f"- … +{len(entries) - len(show)} older")
            lines.append("")
        if not any_content:
            return "(Local working memory empty)"
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable state including section scores.

        Returns:
            Dict keyed by section names plus ``_section_scores``.
        """
        with self._lock:
            out: dict[str, Any] = {
                s: list(self._entries.get(s, [])) for s in SECTIONS
            }
            out["_section_scores"] = {
                k: dict(v) for k, v in (self._section_scores or {}).items()
            }
            return out


_local_store: LocalWorkingMemoryStore | None = None


def get_local_working_memory(filepath: Path | None = None) -> LocalWorkingMemoryStore:
    """Return the process-wide local WM singleton.

    Args:
        filepath: When set, replace the singleton with a store bound to that path (tests).

    Returns:
        Shared :class:`LocalWorkingMemoryStore` instance.
    """
    global _local_store
    if filepath is not None:
        _local_store = LocalWorkingMemoryStore(filepath=filepath)
        return _local_store
    if _local_store is None:
        _local_store = LocalWorkingMemoryStore()
    return _local_store


def reset_local_working_memory_for_tests(filepath: Path | None = None) -> None:
    """Clear the singleton; optionally bind the next store to an isolated file.

    Args:
        filepath: If provided, immediately create a store at this path.
    """
    global _local_store
    _local_store = None
    if filepath is not None:
        get_local_working_memory(filepath=filepath)
