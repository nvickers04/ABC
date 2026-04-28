"""PR29 - Verify the LLM injection seam.

Pins:
  * ``GrokLLM`` instances satisfy ``LLMClientProtocol`` structurally.
  * The ``FakeLLM`` test double also satisfies the protocol.
  * ``TradingAgent.__init__`` accepts an injected client and assigns it
    to ``self.grok`` (no real API key needed).
  * The fake's ``client.chat.create(...).sample()`` round-trip works
    end-to-end so future tests can exercise the agent loop.
"""

from __future__ import annotations

import asyncio

import pytest

from core.runtime.interfaces import LLMClientProtocol
from fakes import FakeLLM, FakeResponse, FakeUsage


class TestLLMInjectionSeam:
    def test_fake_satisfies_protocol(self):
        fake = FakeLLM()
        # ``runtime_checkable`` Protocols enable isinstance() at runtime
        # for attribute presence (not signatures), which is fine here.
        assert isinstance(fake, LLMClientProtocol)
        assert hasattr(fake, "model")
        assert hasattr(fake.client, "chat")
        assert hasattr(fake.client.chat, "create")

    def test_grok_llm_satisfies_protocol(self, monkeypatch):
        # Avoid touching real network: GrokLLM only needs an API key
        # placeholder to construct.
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        # ``GrokLLM`` instantiates a grpc.aio client which calls
        # ``asyncio.get_event_loop()``; on Python 3.13 that raises if no
        # loop is current (e.g. after a sibling test ran ``asyncio.run``).
        # Bind a fresh loop to keep this test self-contained.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from core.grok_llm import GrokLLM
            llm = GrokLLM()
            assert isinstance(llm, LLMClientProtocol)
            assert isinstance(llm.model, str) and llm.model
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_fake_round_trip(self):
        fake = FakeLLM(
            model="fake-grok-1",
            responses=[[FakeResponse(content="hello",
                                     usage=FakeUsage(4, 2))]],
        )
        chat = fake.client.chat.create(
            model=fake.model, messages=[],
            temperature=0.0, max_tokens=10, seed=1,
        )
        resp = asyncio.run(chat.sample())
        assert resp.content == "hello"
        assert resp.usage.prompt_tokens == 4
        assert resp.usage.completion_tokens == 2
        # The factory recorded what was asked of it.
        assert fake.create_calls[0]["model"] == "fake-grok-1"
        assert fake.create_calls[0]["temperature"] == 0.0

    def test_trading_agent_accepts_injected_grok(self, monkeypatch):
        """TradingAgent should adopt the injected client without
        constructing a real GrokLLM."""
        # Block the fallback so we know it's unused.
        def _boom(*a, **k):
            raise AssertionError("get_grok_llm fallback should not run "
                                 "when grok is injected")
        monkeypatch.setattr("core.agent.get_grok_llm", _boom)

        # Minimal stubs for the other constructor args.
        from core.agent import TradingAgent

        class _G:
            cash_value = 1.0
            net_liquidation = 1.0

            async def get_account_summary(self):
                return {}

            async def get_positions(self):
                return []

            async def get_open_orders(self):
                return []

            async def flatten_all(self):
                return {}

        gateway = _G()
        tools = object()  # ToolExecutor is structurally typed in __init__

        fake = FakeLLM()
        agent = TradingAgent(gateway, tools, grok=fake)
        assert agent.grok is fake
        assert agent.grok.model == "fake-grok"

    def test_fake_queue_supports_multiple_chats(self):
        fake = FakeLLM(responses=[
            [FakeResponse(content="reply-1", usage=FakeUsage(1, 1))],
            [FakeResponse(content="reply-2", usage=FakeUsage(2, 2))],
        ])
        c1 = fake.client.chat.create(model="m", messages=[])
        c2 = fake.client.chat.create(model="m", messages=[])
        r1 = asyncio.run(c1.sample())
        r2 = asyncio.run(c2.sample())
        assert r1.content == "reply-1"
        assert r2.content == "reply-2"
