"""Best-effort JSON parsing helpers for LLM output.

Pure functions with no agent state — import from this module directly.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _try_repair_json(raw: str) -> list[dict[str, Any]]:
    """Best-effort repair for almost-valid model JSON output.

    Handles:
    - Code fence wrappers
    - Smart quotes / trailing commas
    - Truncated JSON (missing closing braces/brackets from token limit)
    """
    text = (raw or "").strip()
    if not text:
        return []

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    elif first != -1 and (last == -1 or last <= first):
        # Truncated JSON — close all open braces/brackets
        text = text[first:]
        text = _close_truncated_json(text)

    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        obj = json.loads(text)
        return [obj] if isinstance(obj, dict) else [i for i in obj if isinstance(i, dict)]
    except Exception:
        pass

    return []


def _close_truncated_json(text: str) -> str:
    """Close truncated JSON by balancing braces/brackets and fixing dangling strings."""
    # Strip trailing partial tokens (incomplete key/value after last comma or colon)
    # Remove trailing partial string value (e.g., `"key": "some truncated text`)
    text = re.sub(r',\s*"[^"]*$', '', text)  # trailing key without value
    text = re.sub(r':\s*"[^"]*$', ': ""', text)  # truncated string value -> empty
    text = re.sub(r':\s*$', ': null', text)  # hanging colon
    text = re.sub(r',\s*$', '', text)  # trailing comma

    # Count open/close braces and brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    # Close in reverse order (brackets first since they're usually nested inside objects)
    text += ']' * max(0, open_brackets)
    text += '}' * max(0, open_braces)
    return text


def _parse_json_objects(raw: str) -> list[dict]:
    """Extract all JSON objects from raw LLM output."""
    if "```json" in raw:
        json_str = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        json_str = raw.split("```")[1].split("```")[0]
    else:
        json_str = raw

    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    json_str = json_str.strip()
    while idx < len(json_str):
        brace = json_str.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(json_str, brace)
            objects.append(obj)
            idx = brace + end
        except json.JSONDecodeError:
            idx = brace + 1

    if not objects:
        objects = _try_repair_json(raw)

    return objects


__all__ = [
    "_try_repair_json",
    "_close_truncated_json",
    "_parse_json_objects",
]
