"""Parity tests for JSON / topic helpers in core.json_parse and core.research_topics."""

from __future__ import annotations

from core import json_parse, research_topics


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
        raw = '{"items": [1, 2, 3'
        closed = json_parse._close_truncated_json(raw)
        import json as _json

        parsed = _json.loads(closed)
        assert parsed == {"items": [1, 2, 3]}

    def test_empty_input_returns_empty(self):
        assert json_parse._try_repair_json("") == []
        assert json_parse._try_repair_json("   ") == []


class TestResearchTopicsBehavior:
    def test_categorize_query_returns_category_and_ttl(self):
        cat, ttl = research_topics._categorize_query("AAPL earnings date")
        assert isinstance(cat, str) and cat
        assert isinstance(ttl, int) and ttl > 0
