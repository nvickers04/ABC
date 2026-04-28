"""Reusable fake LLM clients for tests (PR29).

These satisfy the ``LLMClientProtocol`` from ``core.runtime.interfaces``
and let tests drive ``TradingAgent`` without a real Grok API key.

Usage:

    fake = FakeLLM(responses=[FakeResponse(content='{"answer": 42}',
                                            usage=FakeUsage(10, 5))])
    agent = TradingAgent(gateway, tools, grok=fake)

Note: import as ``from fakes import FakeLLM`` — pytest's rootdir
mechanism puts ``tests/`` on ``sys.path`` so a flat import works
without making ``tests/`` a package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class FakeResponse:
    content: str = ""
    usage: FakeUsage = field(default_factory=FakeUsage)


class _FakeChat:
    """A single chat handle. Pops one response per ``sample()`` call;
    raises ``IndexError`` if the canned list is exhausted (tests should
    pre-load enough responses)."""

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.sampled: list[FakeResponse] = []

    async def sample(self) -> FakeResponse:
        r = self._responses.pop(0)
        self.sampled.append(r)
        return r


class _FakeChatFactory:
    def __init__(self, owner: "FakeLLM"):
        self._owner = owner

    def create(self, **kwargs: Any) -> _FakeChat:
        self._owner.create_calls.append(kwargs)
        chat = _FakeChat(self._owner.next_responses())
        self._owner.chats.append(chat)
        return chat


class _FakeClient:
    def __init__(self, owner: "FakeLLM"):
        self.chat = _FakeChatFactory(owner)


class FakeLLM:
    """Mimics ``GrokLLM`` for tests.

    Pre-load it with one or more **lists** of canned responses, where each
    list is consumed by one ``client.chat.create(...).sample()`` sequence.
    """

    def __init__(self, model: str = "fake-grok",
                 responses: list[list[FakeResponse]] | None = None):
        self.model = model
        self.client = _FakeClient(self)
        self._response_queues: list[list[FakeResponse]] = list(responses or [])
        self.create_calls: list[dict] = []
        self.chats: list[_FakeChat] = []

    def queue(self, responses: list[FakeResponse]) -> None:
        self._response_queues.append(list(responses))

    def next_responses(self) -> list[FakeResponse]:
        if self._response_queues:
            return self._response_queues.pop(0)
        return []
