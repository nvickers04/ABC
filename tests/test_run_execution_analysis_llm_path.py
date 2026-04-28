"""PR30 - End-to-end happy path: ``_run_execution_analysis`` with an
injected fake Grok producing a valid graduated-param proposal.

This is the test that the analysis flagged as missing: exercise the
full LLM-coupled branch deterministically, all the way to a row in
``graduated_params``.

Coverage:
  * Calibration runs (>=5 snapshots).
  * LLM is called via the FakeLLM.
  * cost_tracker.log_llm_call is invoked with the response's usage.
  * Proposal with valid 4-part param_key is accepted.
  * Mann-Whitney comparison succeeds because we seed two different
    buckets with distinct slippage distributions.
  * graduated_params gains the new row.
  * Empty / malformed proposals are rejected silently.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

# scipy is required for _test_proposal; skip cleanly if missing.
pytest.importorskip("scipy")

from fakes import FakeLLM, FakeResponse, FakeUsage


# ── Helpers ───────────────────────────────────────────────────


def _insert_snapshot(db, *, order_type="market", time_bucket="open",
                    atr_bucket="high", slippage_bps=10.0,
                    status="filled",
                    ts="2026-04-15T10:00:00", symbol="AAPL"):
    db.execute(
        """INSERT INTO execution_snapshots (
              ts, symbol, side, quantity, order_type, intent,
              fill_price, slippage_bps, time_bucket, atr_bucket, status
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, symbol, "BUY", 100, order_type, "entry",
         100.0, slippage_bps, time_bucket, atr_bucket, status),
    )
    db.commit()


def _seed_two_buckets_with_clear_difference(db):
    """Seed enough snapshots in two buckets to drive Mann-Whitney.

    Target bucket (market.entry.open.high) has bad slippage (~50 bps);
    other bucket of same order_type (market.entry.midday.all) has good
    slippage (~5 bps). With 12 samples each and clear separation, the
    p-value will be < 0.20.
    """
    for v in [48.0, 49.0, 50.0, 51.0, 52.0,
              47.0, 49.5, 50.5, 51.5, 48.5, 50.2, 49.8]:
        _insert_snapshot(db, time_bucket="open", atr_bucket="high",
                         slippage_bps=v)
    for v in [4.0, 5.0, 6.0, 5.5, 4.5,
              5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0]:
        _insert_snapshot(db, time_bucket="midday", atr_bucket="low",
                         slippage_bps=v)


def _make_agent(grok):
    """Build a stub agent. We bind the real ``_test_proposal`` method so
    Mann-Whitney validation runs end-to-end."""
    from core.agent import TradingAgent
    cost_calls = []

    def _log(**kw):
        cost_calls.append(kw)

    cost_tracker = SimpleNamespace(log_llm_call=_log)
    agent = SimpleNamespace(
        grok=grok,
        cost_tracker=cost_tracker,
        _cost_calls=cost_calls,
    )
    # Re-bind the unbound method so ``self._test_proposal(...)`` works.
    agent._test_proposal = lambda prop, groups: TradingAgent._test_proposal(agent, prop, groups)
    return agent


async def _run(agent):
    from core.agent import TradingAgent
    await TradingAgent._run_execution_analysis(agent)


# ── Tests ─────────────────────────────────────────────────────


class TestRunExecutionAnalysisLLMPath:
    def test_valid_proposal_graduates(self, db):
        """Fake LLM returns one well-formed proposal -> row appears in
        graduated_params and cost_tracker is called."""
        from memory import get_graduated_params
        _seed_two_buckets_with_clear_difference(db)

        proposal_json = json.dumps([{
            "param_key": "market.entry.open.high",
            "param_value": "adaptive",
            "previous_value": "market",
            "evidence_summary": "open/high slippage ~50bps vs midday ~5bps",
            "estimated_improvement_bps": 30,
        }])
        fake = FakeLLM(responses=[
            [FakeResponse(content=proposal_json,
                          usage=FakeUsage(100, 50))],
        ])
        agent = _make_agent(fake)

        asyncio.run(_run(agent))

        # cost_tracker recorded the call.
        assert len(agent._cost_calls) == 1
        assert agent._cost_calls[0]["purpose"] == "execution_analysis"
        assert agent._cost_calls[0]["tokens_in"] == 100

        # Graduated param row exists.
        params = get_graduated_params(active_only=True)
        keys = {p["param_key"] for p in params}
        assert "market.entry.open.high" in keys
        new = next(p for p in params if p["param_key"] == "market.entry.open.high")
        assert new["param_value"] == "adaptive"
        # p_value should be set (Mann-Whitney ran).
        assert new["p_value"] is not None
        assert new["p_value"] < 0.20

    def test_empty_proposals_no_graduation(self, db):
        from memory import get_graduated_params
        _seed_two_buckets_with_clear_difference(db)

        fake = FakeLLM(responses=[
            [FakeResponse(content="[]", usage=FakeUsage(50, 5))],
        ])
        agent = _make_agent(fake)
        asyncio.run(_run(agent))

        # Cost still tracked even on empty array.
        assert len(agent._cost_calls) == 1
        assert get_graduated_params(active_only=True) == []

    def test_malformed_json_no_graduation(self, db):
        from memory import get_graduated_params
        _seed_two_buckets_with_clear_difference(db)

        fake = FakeLLM(responses=[
            [FakeResponse(content="not json at all <think>...</think>",
                          usage=FakeUsage(50, 5))],
        ])
        agent = _make_agent(fake)
        asyncio.run(_run(agent))

        # The except-Exception catch around the whole try block does
        # NOT include the cost_tracker call's JSON parsing -- log_llm_call
        # runs first. So cost is tracked.
        assert len(agent._cost_calls) == 1
        assert get_graduated_params(active_only=True) == []

    def test_invalid_param_key_rejected(self, db):
        """Three-part key fails validate_param_key -> not graduated."""
        from memory import get_graduated_params
        _seed_two_buckets_with_clear_difference(db)

        proposal_json = json.dumps([{
            "param_key": "market.entry.open",  # only 3 parts
            "param_value": "adaptive",
        }])
        fake = FakeLLM(responses=[
            [FakeResponse(content=proposal_json, usage=FakeUsage(50, 10))],
        ])
        agent = _make_agent(fake)
        asyncio.run(_run(agent))
        assert get_graduated_params(active_only=True) == []

    def test_only_first_two_proposals_processed(self, db):
        """The function processes at most ``proposals[:2]`` -- a third
        valid proposal is ignored even if otherwise acceptable."""
        from memory import get_graduated_params
        _seed_two_buckets_with_clear_difference(db)

        proposal_json = json.dumps([
            {"param_key": "market.entry.open.high",
             "param_value": "adaptive"},
            {"param_key": "market.entry.midday.low",
             "param_value": "limit"},
            # Third proposal -- should be silently dropped.
            {"param_key": "market.entry.morning.medium",
             "param_value": "twap"},
        ])
        fake = FakeLLM(responses=[
            [FakeResponse(content=proposal_json, usage=FakeUsage(80, 30))],
        ])
        agent = _make_agent(fake)
        asyncio.run(_run(agent))

        keys = {p["param_key"] for p in get_graduated_params(active_only=True)}
        # No row for the dropped third proposal.
        assert "market.entry.morning.medium" not in keys
