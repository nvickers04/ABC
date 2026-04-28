"""PR14 parity tests — verify the JSON / topic helpers extracted from
``core.agent`` keep their public behavior and remain importable from
both their new home and the legacy module path.
"""

from __future__ import annotations

import core.agent as agent_mod
from core import json_parse, research_topics


class TestReExports:
    def test_json_helpers_reexported(self):
        for name in ("_try_repair_json", "_close_truncated_json", "_parse_json_objects"):
            assert getattr(agent_mod, name) is getattr(json_parse, name), name

    def test_topic_helpers_reexported(self):
        for name in ("_categorize_query", "_get_ticker_symbols", "_TICKER_SYMBOLS"):
            assert getattr(agent_mod, name) is getattr(research_topics, name), name


class TestJsonParseBehavior:
    def test_repairs_smart_quotes_and_trailing_comma(self):
        raw = '{\u201ckey\u201d: \u201cval\u201d, }'
        out = json_parse._try_repair_json(raw)
        assert out == [{"key": "val"}]

    def test_extracts_object_inside_code_fence(self):
        raw = "Sure, here:\n```json\n{\"a\": 1}\n```\nThanks."
        out = json_parse._parse_json_objects(raw)
        assert out == [{"a": 1}]

    def test_extracts_multiple_objects(self):
        raw = '{"a":1}{"b":2}'
        out = json_parse._parse_json_objects(raw)
        assert out == [{"a": 1}, {"b": 2}]

    def test_close_truncated_json_balances(self):
        # Truncated mid-array: missing both `]` and `}`.
        raw = '{"items": [1, 2, 3'
        closed = json_parse._close_truncated_json(raw)
        # Must be valid JSON now.
        import json as _json
        parsed = _json.loads(closed)
        assert parsed == {"items": [1, 2, 3]}

    def test_empty_input_returns_empty(self):
        assert json_parse._try_repair_json("") == []
        assert json_parse._try_repair_json("   ") == []


class TestCategorizeQuery:
    def test_sector_kw_classified_sector(self):
        cat, ttl = research_topics._categorize_query("sector rotation play")
        assert cat == "sector"
        assert ttl == 14400

    def test_default_macro(self):
        cat, ttl = research_topics._categorize_query("Fed rate decision today")
        assert cat == "macro"
        assert ttl == 0
