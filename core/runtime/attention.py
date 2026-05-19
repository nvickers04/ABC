"""
Attention layer — structured triggers the agent registers and the
evaluator that fires them.

Two write paths feed ``attention_triggers``:

1. **Structured (preferred)**: when the agent calls
   ``update_working_memory(section="watching_for", metadata={...})``
   the metadata may contain ``symbol``, ``condition``, ``threshold``
   and an optional ``confirm_with`` list.  The curator sweep
   (``sync_from_working_memory``) registers a trigger for any active
   watching_for entry not yet seen.

2. **Text fallback**: if no structured metadata is present, we attempt
   a regex parse of the entry text — supporting the ``[trigger: SYM
   > 530 + vol]`` trailer first, then a coarse "above 530" / "below
   530" pattern.  Failed parses leave the entry as a narrative-only
   note (no trigger registered).

The evaluator is called from the scorer round with the round's quotes
and composite scores; it updates ``last_value`` for crossing
detection, and on a fire it stamps ``fired_ts`` / ``fire_value`` /
``fire_note`` and posts to the ``wake_bus`` so the agent loop wakes.

The renderer produces the ATTENTION block at the very top of the
cycle prompt, above WORKING MEMORY.  Recently-fired (within
``RENDER_WINDOW_S``) and still-pending triggers are shown.

Cap: ``ACTIVE_CAP = 10`` active triggers; insertion past the cap
evicts the oldest active row.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Iterable, Optional

from core.log_context import get_logger
from core.memory_config import get_memory_config

logger = get_logger(__name__)


_mem_cfg = get_memory_config()
ACTIVE_CAP: int = _mem_cfg.attention_active_cap
RENDER_WINDOW_S: float = _mem_cfg.attention_render_window_s

# Conditions we evaluate.  ``crosses_above`` / ``crosses_below`` need a
# prior ``last_value`` to detect the crossing edge; ``above`` / ``below``
# fire on any observation that meets the threshold.
CONDITIONS: tuple[str, ...] = (
    "above",
    "below",
    "crosses_above",
    "crosses_below",
    "composite_above",
    "composite_below",
)

# Aliases that map to the canonical condition names.
_CONDITION_ALIASES: dict[str, str] = {
    ">": "above",
    "gt": "above",
    "above": "above",
    "break_above": "crosses_above",
    "breaks_above": "crosses_above",
    "crosses": "crosses_above",
    "<": "below",
    "lt": "below",
    "below": "below",
    "break_below": "crosses_below",
    "breaks_below": "crosses_below",
    "crosses_above": "crosses_above",
    "crosses_below": "crosses_below",
    "composite_above": "composite_above",
    "composite_below": "composite_below",
}


# ── Helpers ──────────────────────────────────────────────────────


def _normalize_condition(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    return _CONDITION_ALIASES.get(key)


def _quote_value(quote: Any) -> Optional[float]:
    """Pull a usable price (mid > last) from a Quote object or dict."""
    if quote is None:
        return None
    # Object with .mid property (data.data_provider.Quote)
    mid = getattr(quote, "mid", None)
    if isinstance(mid, (int, float)):
        return float(mid)
    last = getattr(quote, "last", None)
    if isinstance(last, (int, float)):
        return float(last)
    if isinstance(quote, dict):
        for k in ("mid", "last", "price"):
            v = quote.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        bid = quote.get("bid")
        ask = quote.get("ask")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
            return (float(bid) + float(ask)) / 2
    return None


# ── Parsing ──────────────────────────────────────────────────────


def parse_metadata(metadata: Any) -> Optional[dict]:
    """Validate a structured trigger spec from ``working_memory.metadata``.

    Returns a normalized dict with keys ``symbol``, ``condition``,
    ``threshold`` (Optional[float]), ``confirm_with`` (list[str]).
    Returns ``None`` if the metadata can't be turned into a trigger.
    """
    if not isinstance(metadata, dict):
        return None
    symbol = metadata.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    condition = _normalize_condition(metadata.get("condition"))
    if condition is None:
        return None
    threshold_raw = metadata.get("threshold")
    threshold: Optional[float] = None
    if isinstance(threshold_raw, (int, float)):
        threshold = float(threshold_raw)
    elif isinstance(threshold_raw, str):
        try:
            threshold = float(threshold_raw)
        except ValueError:
            threshold = None
    # composite_* and above/below/crosses_* all require a numeric
    # threshold (we do not support open-ended triggers in this PR).
    if threshold is None:
        return None
    confirm_raw = metadata.get("confirm_with")
    if isinstance(confirm_raw, (list, tuple)):
        confirm_with = [str(x).strip() for x in confirm_raw if str(x).strip()]
    elif isinstance(confirm_raw, str) and confirm_raw.strip():
        confirm_with = [confirm_raw.strip()]
    else:
        confirm_with = []
    return {
        "symbol": symbol.strip().upper(),
        "condition": condition,
        "threshold": threshold,
        "confirm_with": confirm_with,
    }


# Trailer:  [trigger: SYM > 530 + vol, composite_positive]
_TRAILER_RE = re.compile(
    r"\[trigger:\s*([A-Z][A-Z0-9.\-]{0,9})\s*([<>])\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*\+\s*([^\]]*))?\s*\]",
    re.IGNORECASE,
)

# Coarse fallback: first ticker-shaped word + "above|below|break above|break below" + number.
_FALLBACK_RE = re.compile(
    r"\b([A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b.*?"
    r"(break\s+above|break\s+below|above|below)\s+"
    r"\$?([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def parse_text(text: Any) -> Optional[dict]:
    """Best-effort regex parse of a ``watching_for`` entry into a trigger spec.

    Returns the same dict shape as :func:`parse_metadata` or ``None``
    on failure.  Trailer form wins when present.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    m = _TRAILER_RE.search(text)
    if m:
        sym = m.group(1).upper()
        op = m.group(2)
        try:
            thr = float(m.group(3))
        except ValueError:
            return None
        confirm_raw = m.group(4) or ""
        confirm_with = [
            tok.strip() for tok in re.split(r"[,/+]", confirm_raw) if tok.strip()
        ]
        condition = "above" if op == ">" else "below"
        return {
            "symbol": sym,
            "condition": condition,
            "threshold": thr,
            "confirm_with": confirm_with,
        }
    m = _FALLBACK_RE.search(text)
    if m:
        sym = m.group(1).upper()
        # Skip very short common words that look like tickers ("A", "I").
        if sym in {"A", "I", "AT", "IN", "ON", "OR", "IT", "IS", "BE", "TO"}:
            return None
        verb = m.group(2).lower()
        try:
            thr = float(m.group(3))
        except ValueError:
            return None
        if "above" in verb:
            condition = "crosses_above" if verb.startswith("break") else "above"
        else:
            condition = "crosses_below" if verb.startswith("break") else "below"
        return {
            "symbol": sym,
            "condition": condition,
            "threshold": thr,
            "confirm_with": [],
        }
    return None


# ── Persistence ──────────────────────────────────────────────────


def _evict_to_cap(conn) -> int:
    """Mark oldest 'active' rows as 'evicted' until ``ACTIVE_CAP`` remains.

    Returns the number of rows evicted.
    """
    cur = conn.execute(
        "SELECT COUNT(*) FROM attention_triggers WHERE state='active'"
    )
    n = int(cur.fetchone()[0])
    if n <= ACTIVE_CAP:
        return 0
    excess = n - ACTIVE_CAP
    cur = conn.execute(
        "SELECT id FROM attention_triggers WHERE state='active' "
        "ORDER BY created_ts ASC LIMIT ?",
        (excess,),
    )
    ids = [row[0] for row in cur.fetchall()]
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"UPDATE attention_triggers SET state='evicted' WHERE id IN ({placeholders})",
        ids,
    )
    conn.commit()
    return len(ids)


def register_trigger(
    conn,
    *,
    symbol: str,
    condition: str,
    threshold: Optional[float],
    confirm_with: Optional[Iterable[str]] = None,
    source_entry_id: Optional[int] = None,
    source_text: Optional[str] = None,
    now: Optional[float] = None,
) -> Optional[int]:
    """Insert one active trigger; evicts oldest if the active cap is hit.

    Returns the new row id on success, ``None`` on validation failure.
    """
    cond = _normalize_condition(condition)
    if cond is None:
        return None
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    sym = symbol.strip().upper()
    thr = float(threshold) if isinstance(threshold, (int, float)) else None
    if thr is None:
        return None
    cw = [str(x) for x in (confirm_with or []) if str(x).strip()]
    ts = float(now) if now is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO attention_triggers
            (symbol, condition, threshold, confirm_with_json,
             source_entry_id, source_text, created_ts, state)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
        RETURNING id
        """,
        (
            sym,
            cond,
            thr,
            json.dumps(cw),
            int(source_entry_id) if source_entry_id is not None else None,
            source_text if isinstance(source_text, str) else None,
            ts,
        ),
    )
    conn.commit()
    row = cur.fetchone()
    new_id = int(row["id"]) if row else None
    _evict_to_cap(conn)
    return new_id


def sync_from_working_memory(conn, *, now: Optional[float] = None) -> int:
    """Walk active ``watching_for`` rows; register triggers for any not yet
    represented in ``attention_triggers`` (matched by ``source_entry_id``).

    Returns the number of triggers newly registered.
    """
    ts = float(now) if now is not None else time.time()
    try:
        cur = conn.execute(
            """
            SELECT id, entry_text, metadata_json
            FROM working_memory
            WHERE section = 'watching_for' AND expires_ts > ?
            ORDER BY created_ts ASC
            """,
            (ts,),
        )
        rows = cur.fetchall()
    except Exception as e:
        logger.debug("attention.sync: working_memory read failed: %s", e)
        return 0
    if not rows:
        return 0
    # Pre-fetch known source ids to avoid duplicate registration.
    cur = conn.execute(
        "SELECT source_entry_id FROM attention_triggers "
        "WHERE source_entry_id IS NOT NULL"
    )
    known: set[int] = {int(r[0]) for r in cur.fetchall()}
    registered = 0
    for entry_id, entry_text, metadata_json in rows:
        if int(entry_id) in known:
            continue
        meta = None
        if metadata_json:
            try:
                meta = json.loads(metadata_json)
            except Exception:
                meta = None
        spec = parse_metadata(meta) if meta else None
        if spec is None:
            spec = parse_text(entry_text)
        if spec is None:
            continue
        rid = register_trigger(
            conn,
            symbol=spec["symbol"],
            condition=spec["condition"],
            threshold=spec["threshold"],
            confirm_with=spec["confirm_with"],
            source_entry_id=int(entry_id),
            source_text=entry_text,
            now=ts,
        )
        if rid is not None:
            registered += 1
    return registered


# ── Evaluation ───────────────────────────────────────────────────


def _check_fire(condition: str, threshold: float, value: float,
                last_value: Optional[float]) -> bool:
    if condition == "above" or condition == "composite_above":
        return value >= threshold
    if condition == "below" or condition == "composite_below":
        return value <= threshold
    if condition == "crosses_above":
        return last_value is not None and last_value < threshold and value >= threshold
    if condition == "crosses_below":
        return last_value is not None and last_value > threshold and value <= threshold
    return False


def evaluate(
    conn,
    quotes: Optional[dict] = None,
    composites: Optional[dict] = None,
    *,
    now: Optional[float] = None,
    wake: bool = True,
) -> list[dict]:
    """Evaluate all active triggers against the given round inputs.

    ``quotes`` maps symbol → ``Quote`` (or dict with ``last``/``mid``).
    ``composites`` maps symbol → composite score (float).
    Returns a list of fired-trigger dicts (one per fire).  Always
    updates ``last_value`` for crossing detection, even when not fired.
    """
    ts = float(now) if now is not None else time.time()
    quotes = quotes or {}
    composites = composites or {}
    try:
        cur = conn.execute(
            "SELECT id, symbol, condition, threshold, confirm_with_json, "
            "source_text, last_value FROM attention_triggers WHERE state='active'"
        )
        rows = cur.fetchall()
    except Exception as e:
        logger.debug("attention.evaluate: read failed: %s", e)
        return []
    fired: list[dict] = []
    for (rid, sym, cond, thr, confirm_json, source_text, last_value) in rows:
        if cond.startswith("composite_"):
            value = composites.get(sym)
        else:
            value = _quote_value(quotes.get(sym))
        if not isinstance(value, (int, float)):
            continue
        value_f = float(value)
        prior = float(last_value) if isinstance(last_value, (int, float)) else None
        try:
            did_fire = _check_fire(cond, float(thr), value_f, prior)
        except Exception:
            did_fire = False
        if did_fire:
            note = _format_fire_note(cond, float(thr), value_f)
            try:
                conn.execute(
                    "UPDATE attention_triggers SET state='fired', "
                    "fired_ts=?, fire_value=?, fire_note=?, last_value=? "
                    "WHERE id=?",
                    (ts, value_f, note, value_f, rid),
                )
            except Exception as e:
                logger.debug("attention.evaluate: fire update failed: %s", e)
                continue
            try:
                confirm_with = json.loads(confirm_json) if confirm_json else []
            except Exception:
                confirm_with = []
            fired.append({
                "id": rid,
                "symbol": sym,
                "condition": cond,
                "threshold": float(thr),
                "value": value_f,
                "fired_ts": ts,
                "note": note,
                "confirm_with": confirm_with,
                "source_text": source_text,
            })
        else:
            try:
                conn.execute(
                    "UPDATE attention_triggers SET last_value=? WHERE id=?",
                    (value_f, rid),
                )
            except Exception:
                pass
    try:
        conn.commit()
    except Exception:
        pass
    if fired and wake:
        try:
            from core.runtime.research_host_runtime import is_research_host_process

            if not is_research_host_process():
                from core.wake_events import wake_bus

                wake_bus.signal(f"attention_{fired[0]['symbol']}")
        except Exception as e:
            logger.debug("attention.evaluate: wake_bus signal failed: %s", e)
    return fired


def _format_fire_note(condition: str, threshold: float, value: float) -> str:
    if condition.startswith("composite_"):
        verb = "above" if "above" in condition else "below"
        return f"composite {verb} {threshold:.2f} (now {value:.2f})"
    if condition in ("above", "crosses_above"):
        verb = "crossed" if condition == "crosses_above" else "above"
        return f"{verb} {threshold:.2f} (now {value:.2f})"
    if condition in ("below", "crosses_below"):
        verb = "crossed below" if condition == "crosses_below" else "below"
        return f"{verb} {threshold:.2f} (now {value:.2f})"
    return f"{condition} {threshold:.2f} (now {value:.2f})"


# ── Render ───────────────────────────────────────────────────────


def render_attention_block(
    conn,
    *,
    now: Optional[float] = None,
    window_s: float | None = None,
    max_rows: int | None = None,
    max_source_chars: int | None = None,
) -> str:
    """Render the ATTENTION block for the cycle prompt.

    Shows all currently-active triggers (so the agent remembers what
    it's watching) and any triggers fired within the last
    ``window_s`` seconds.  Returns ``""`` if there's nothing to show.
    """
    cfg = get_memory_config()
    if window_s is None:
        window_s = cfg.attention_render_window_s
    if max_rows is None:
        max_rows = cfg.attention_render_default_max_rows
    if max_source_chars is None:
        max_source_chars = cfg.attention_render_default_max_source_chars
    ts = float(now) if now is not None else time.time()
    try:
        cur = conn.execute(
            "SELECT symbol, condition, threshold, confirm_with_json, "
            "source_text, fired_ts, fire_note, fire_value "
            "FROM attention_triggers "
            "WHERE state='active' OR (state='fired' AND fired_ts >= ?) "
            "ORDER BY (state='fired') DESC, fired_ts DESC, created_ts DESC",
            (ts - window_s,),
        )
        rows = cur.fetchall()
    except Exception as e:
        logger.debug("attention.render: read failed: %s", e)
        return ""
    if not rows:
        return ""
    if max_rows > 0:
        rows = rows[:max_rows]
    lines = ["⚡ ATTENTION"]
    for (sym, cond, thr, confirm_json, source_text,
         fired_ts, fire_note, fire_value) in rows:
        try:
            confirm_with = json.loads(confirm_json) if confirm_json else []
        except Exception:
            confirm_with = []
        if fired_ts is not None:
            fired_dt = datetime.fromtimestamp(float(fired_ts))
            stamp = fired_dt.strftime("%H:%M")
            head = f"- {sym} {fire_note} at {stamp}"
        else:
            head = f"- {sym} watching: {cond} {float(thr):.2f}"
        if source_text:
            st = source_text.strip()
            if max_source_chars > 0 and len(st) > max_source_chars:
                st = st[: max_source_chars - 1] + "…"
            head += f' (note: "{st}")'
        lines.append(head)
        if confirm_with:
            lines.append(f"  Confirm with: {', '.join(confirm_with)}")
    return "\n".join(lines)


# ── Test helpers ─────────────────────────────────────────────────


def reset_for_tests(conn) -> None:
    """Wipe the ``attention_triggers`` table.  Test-only."""
    try:
        conn.execute("DELETE FROM attention_triggers")
        conn.commit()
    except Exception as e:
        logger.debug("attention.reset_for_tests failed: %s", e)
