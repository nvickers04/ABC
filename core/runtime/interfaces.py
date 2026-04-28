"""Runtime adapter contracts.

These ``Protocol`` classes describe the *minimal* surface that the runtime
layer (``core.runtime.*``) needs from external dependencies. They exist so
that the runtime can be unit-tested with lightweight stubs and so that
implementation modules (``data.broker_gateway``, ``data.market_hours``,
``data.cost_tracker``) can evolve without forcing matching changes inside
the agent loop.

These protocols intentionally mirror the *current* duck-typed call sites in
``core.agent`` — they are not an aspirational redesign. Adding new methods
must be paired with a characterization test.
"""

from __future__ import annotations

from typing import Any, Awaitable, Protocol, runtime_checkable


# ── Broker / Account ─────────────────────────────────────────────

AccountSummary = dict[str, Any]
"""Free-form account summary dict.

Keys consumed by the runtime today (all optional, all numeric unless noted):

* ``totalcashvalue``   — cash component of the account.
* ``netliquidation``   — total liquidation value (used for loss / drawdown).
* ``dailypnl``         — broker-reported P&L (string or number).
* ``unrealizedpnl``    — open-position unrealized P&L.
* ``realizedpnl``      — realized P&L for the session.
"""


@runtime_checkable
class BrokerGatewayProtocol(Protocol):
    """Subset of the broker gateway used by runtime modules."""

    cash_value: float
    net_liquidation: float

    def get_account_summary(self) -> Awaitable[AccountSummary]: ...
    def get_positions(self) -> Awaitable[list[dict[str, Any]]]: ...
    def get_open_orders(self) -> Awaitable[list[dict[str, Any]]]: ...
    def flatten_all(self) -> Awaitable[dict[str, Any]]: ...


# ── Market Hours ─────────────────────────────────────────────────


@runtime_checkable
class MarketHoursProtocol(Protocol):
    """Subset of ``data.market_hours`` used by the state builder."""

    def get_session_info(self) -> dict[str, Any]: ...


# ── Cost Tracker ─────────────────────────────────────────────────


@runtime_checkable
class CostTrackerProtocol(Protocol):
    """Subset of ``data.cost_tracker`` used by the safety controller."""

    def get_budget_summary(self) -> Any: ...


# ── Wake Bus ─────────────────────────────────────────────────────


@runtime_checkable
class WakeBusProtocol(Protocol):
    """Subset of ``core.wake_events.WakeBus`` used by the scheduler.

    The scheduler awaits :meth:`wait` after each cycle to sleep until
    either the timeout expires (returns ``"timeout"``) or a producer
    calls :meth:`signal` (returns the supplied reason). ``signal`` is
    not used by the scheduler itself but is part of the documented
    contract so test stubs can drive the bus.
    """

    def wait(self, timeout: float) -> Awaitable[str]: ...
    def signal(self, reason: str) -> None: ...


# -- LLM Client -------------------------------------------------


@runtime_checkable
class LLMUsageProtocol(Protocol):
    """Token usage on a single LLM response."""

    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class LLMResponseProtocol(Protocol):
    """Single reply object returned by `chat.sample()`."""

    content: str
    usage: LLMUsageProtocol


@runtime_checkable
class LLMChatProtocol(Protocol):
    """The mutable chat handle the agent loop drives turn-by-turn."""

    def sample(self) -> Awaitable[LLMResponseProtocol]: ...


@runtime_checkable
class LLMChatFactoryProtocol(Protocol):
    """`llm.client.chat` namespace used to build new chat handles."""

    def create(
        self,
        *,
        model: str,
        messages: list[Any],
        temperature: float = ...,
        max_tokens: int = ...,
        seed: int | None = ...,
    ) -> LLMChatProtocol: ...


@runtime_checkable
class LLMClientNamespaceProtocol(Protocol):
    """`llm.client` -- the only attribute the agent reads off the SDK."""

    chat: LLMChatFactoryProtocol


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Minimal duck-typed contract that `self.grok` must satisfy.

    Any object with a `.client.chat.create(...)` factory and a `.model`
    attribute is acceptable. `GrokLLM` is the concrete production
    implementation; tests inject a fake (see
    `tests/fakes/llm.py`).
    """

    model: str
    client: LLMClientNamespaceProtocol
