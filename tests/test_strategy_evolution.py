"""Strategy evolution (review-only meta-optimizer)."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_collect_performance_digest_empty():
    from core.strategy_evolution import collect_performance_digest

    d = collect_performance_digest(days=3)
    assert d["window_days"] == 3
    assert "cycle_stats" in d


def test_snapshot_strategy_corpus():
    from core.strategy_evolution import PROMPT_TARGET_IDS, snapshot_strategy_corpus

    c = snapshot_strategy_corpus(max_tools=10)
    assert c.active_profile
    assert len(c.tool_schemas) <= 10
    for tid in PROMPT_TARGET_IDS[:3]:
        assert tid in c.prompt_regions


def test_build_review_diff():
    from core.strategy_evolution import StrategyCorpus, build_review_diff

    corpus = StrategyCorpus(
        generated_at="2026-01-01T00:00:00+00:00",
        active_profile="balanced",
        trading_mode="paper",
        prompt_regions={"mode_guidance_paper": "OLD guidance"},
        tool_schemas={"plan_order": "{} -> old intent"},
    )
    suggestions = {
        "prompt_suggestions": [
            {
                "target_id": "mode_guidance_paper",
                "proposed_text": "NEW guidance",
                "rationale": "clearer",
            }
        ],
        "tool_suggestions": [
            {
                "tool_name": "plan_order",
                "proposed_schema_description": "{} -> new intent",
                "rationale": "precision",
            }
        ],
    }
    diff = build_review_diff(corpus, suggestions)
    assert "REVIEW ONLY" in diff
    assert "mode_guidance_paper" in diff
    assert "plan_order" in diff
    assert "-OLD" in diff or "OLD guidance" in diff


def test_parse_json_response():
    from core.strategy_evolution import _parse_json_response

    raw = 'Here is JSON:\n```json\n{"executive_summary": "ok", "prompt_suggestions": []}\n```'
    parsed = _parse_json_response(raw)
    assert parsed["executive_summary"] == "ok"


@pytest.mark.asyncio
async def test_invoke_meta_optimizer_mocked(monkeypatch):
    from core.strategy_evolution import StrategyCorpus, invoke_meta_optimizer

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 20

    class _Resp:
        content = json.dumps(
            {
                "executive_summary": "test",
                "performance_diagnosis": [],
                "prompt_suggestions": [],
                "tool_suggestions": [],
                "do_not_change": ["safety"],
            }
        )
        usage = _Usage()

    class _Chat:
        async def sample(self):
            return _Resp()

    class _Client:
        def __init__(self, *a, **k):
            pass

        @property
        def chat(self):
            return self

        def create(self, **kwargs):
            return _Chat()

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr("xai_sdk.AsyncClient", _Client)
    monkeypatch.setattr(
        "data.cost_tracker.get_cost_tracker",
        lambda: type("T", (), {"log_llm_usage": lambda *a, **k: 0.01})(),
    )

    perf = {"window_days": 7, "cycle_stats": {"cycles": 1}}
    corpus = StrategyCorpus(
        generated_at="t",
        active_profile="balanced",
        trading_mode="paper",
        prompt_regions={"mode_guidance_paper": "x"},
        tool_schemas={},
    )
    out = await invoke_meta_optimizer(perf, corpus)
    assert out["suggestions"]["executive_summary"] == "test"
