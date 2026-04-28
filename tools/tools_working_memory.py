"""Working-memory tool handlers — the agent's only write path into the
short-term monologue store.

Two tools:
  - update_working_memory(section, entry, expires_in_minutes?)
  - clear_working_memory_entry(section, entry_id?)

There is intentionally NO read tool.  The rendered WORKING MEMORY block
is auto-injected into every cycle prompt (see core/agent.py), so a read
tool would only tempt the agent to burn a turn fetching information it
already has.  See docs/PLAN_COGNITIVE_ARCHITECTURE.md §3 for the
rationale.
"""

from __future__ import annotations

import logging
from typing import Any

from memory.working_memory import (
    SECTIONS,
    SECTION_CAPS,
    SECTION_DEFAULT_EXPIRY,
    get_working_memory,
)

logger = logging.getLogger(__name__)


def _valid_sections_msg() -> str:
    """Render section help with caps + defaults so error messages teach."""
    parts = []
    for s in SECTIONS:
        cap = SECTION_CAPS[s]
        default = SECTION_DEFAULT_EXPIRY[s]
        parts.append(f"{s} (cap={cap}, default={default})")
    return ", ".join(parts)


async def handle_update_working_memory(executor, params: dict) -> Any:
    """Add an entry to one of the agent's working-memory sections.

    Params:
      section            (str, required) — one of: open_theses,
                         recent_verdicts, watching_for, regime_notes,
                         lessons_today
      entry              (str, required) — the natural-language note
      expires_in_minutes (int|float, optional) — override the section
                         default; omit for the default
      metadata           (dict, optional) — structured trailer the
                         curator may parse later (e.g. attention triggers)
    """
    section = (params.get("section") or "").strip()
    if not section:
        return {"error": "section required",
                "valid_sections": _valid_sections_msg()}
    if section not in SECTIONS:
        return {"error": f"unknown section {section!r}",
                "valid_sections": _valid_sections_msg()}

    entry = params.get("entry")
    if not isinstance(entry, str) or not entry.strip():
        return {"error": "entry required (non-empty string)"}

    expires = params.get("expires_in_minutes")
    if expires is not None:
        try:
            expires = float(expires)
            if expires < 0:
                return {"error": "expires_in_minutes must be >= 0"}
        except (TypeError, ValueError):
            return {"error": f"expires_in_minutes must be numeric, got {expires!r}"}

    metadata = params.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return {"error": "metadata must be an object/dict"}

    try:
        wm = get_working_memory()
    except Exception as e:
        logger.warning("working_memory unavailable: %s", e)
        return {"error": "working_memory unavailable"}

    try:
        eid = wm.add(
            section,
            entry,
            expires_in_minutes=expires,
            metadata=metadata,
        )
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error("update_working_memory failed: %s", e)
        return {"error": f"update failed: {e}"}

    snap = wm.snapshot().get(section, [])
    return {
        "success": True,
        "section": section,
        "entry_id": eid,
        "section_size": len(snap),
        "section_cap": SECTION_CAPS[section],
    }


async def handle_clear_working_memory_entry(executor, params: dict) -> Any:
    """Remove one entry by id, or clear an entire section.

    Params:
      section  (str, required)
      entry_id (int, optional) — omit to clear the whole section
    """
    section = (params.get("section") or "").strip()
    if not section:
        return {"error": "section required",
                "valid_sections": _valid_sections_msg()}
    if section not in SECTIONS:
        return {"error": f"unknown section {section!r}",
                "valid_sections": _valid_sections_msg()}

    entry_id = params.get("entry_id")
    if entry_id is not None:
        try:
            entry_id = int(entry_id)
        except (TypeError, ValueError):
            return {"error": f"entry_id must be int, got {entry_id!r}"}

    try:
        wm = get_working_memory()
    except Exception as e:
        logger.warning("working_memory unavailable: %s", e)
        return {"error": "working_memory unavailable"}

    try:
        removed = wm.clear(section, entry_id=entry_id)
    except Exception as e:
        logger.error("clear_working_memory_entry failed: %s", e)
        return {"error": f"clear failed: {e}"}

    return {
        "success": True,
        "section": section,
        "entry_id": entry_id,
        "removed": removed,
    }


HANDLERS = {
    "update_working_memory": handle_update_working_memory,
    "clear_working_memory_entry": handle_clear_working_memory_entry,
}
