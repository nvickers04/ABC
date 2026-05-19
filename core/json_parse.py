"""Best-effort JSON parsing helpers for LLM output.

Pure functions with no agent state — import from this module directly.
Behavior toggles come from :mod:`core.loop_config`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.loop_config import get_loop_config


def _try_repair_json(raw: str) -> list[dict[str, Any]]:
    """Best-effort repair for almost-valid model JSON output."""
    lc = get_loop_config()
    if not lc.json_repair_on_decode_failure:
        return []

    text = (raw or "").strip()
    if not text:
        return []

    if lc.json_strip_code_fences:
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    elif first != -1 and (last == -1 or last <= first) and lc.json_close_truncated_braces:
        text = text[first:]
        text = _close_truncated_json(text)

    if lc.json_normalize_smart_quotes:
        text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    if lc.json_strip_trailing_commas:
        text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        obj = json.loads(text)
        return [obj] if isinstance(obj, dict) else [i for i in obj if isinstance(i, dict)]
    except Exception:
        pass

    return []


def _close_truncated_json(text: str) -> str:
    """Close truncated JSON by balancing braces/brackets and fixing dangling strings."""
    text = re.sub(r',\s*"[^"]*$', '', text)
    text = re.sub(r':\s*"[^"]*$', ': ""', text)
    text = re.sub(r':\s*$', ': null', text)
    text = re.sub(r',\s*$', '', text)

    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    text += ']' * max(0, open_brackets)
    text += '}' * max(0, open_braces)
    return text


def _parse_json_objects(raw: str) -> list[dict]:
    """Extract all JSON objects from raw LLM output."""
    lc = get_loop_config()
    if lc.json_strip_code_fences and "```json" in raw:
        json_str = raw.split("```json")[1].split("```")[0]
    elif lc.json_strip_code_fences and "```" in raw:
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

    if not objects and lc.json_repair_on_decode_failure:
        objects = _try_repair_json(raw)

    return objects


__all__ = [
    "_try_repair_json",
    "_close_truncated_json",
    "_parse_json_objects",
]
