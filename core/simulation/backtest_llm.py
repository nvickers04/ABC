"""Offline LLM shim driving the real ReAct loop without xAI API calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from core.loop_config import get_loop_config
from core.simulation.llm_cost_estimate import DEFAULT_SIM_LLM_MODEL, LlmUsageEstimate
from research.config import COMPOSITE_TRADE_THRESHOLD


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _Response:
    content: str = ""
    usage: _Usage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _Usage()


class _Chat:
    def __init__(self, responses: list[_Response], owner: "BacktestLLM | None" = None):
        self._responses = list(responses)
        self._owner = owner
        self.messages: list[Any] = []

    def append(self, message: Any) -> None:
        """Mirror xAI chat history API used by :meth:`TradingAgent.run_cycle`."""
        self.messages.append(message)

    async def sample(self) -> _Response:
        if not self._responses:
            cd = self._owner._done_cooldown_seconds() if self._owner else 30
            return _Response(
                content=json.dumps(
                    {"action": "done", "summary": "eof", "cooldown": cd}
                )
            )
        return self._responses.pop(0)


class _ChatFactory:
    def __init__(self, owner: "BacktestLLM"):
        self._owner = owner

    def create(self, **kwargs: Any) -> _Chat:
        self._owner.create_calls.append(kwargs)
        return _Chat(self._owner.next_responses(), owner=self._owner)


class _Client:
    def __init__(self, owner: "BacktestLLM"):
        self.chat = _ChatFactory(owner)


@dataclass
class CyclePlanContext:
    session_date: str
    cycle_index: int
    top_symbol: str | None = None
    top_score: float = 0.0
    traded_today: bool = False


class BacktestLLM:
    """Mimics ``GrokLLM``; queues JSON tool actions per cycle."""

    def __init__(self, model: str = "backtest-sim") -> None:
        self.model = model
        self.client = _Client(self)
        self._response_queues: list[list[_Response]] = []
        self.create_calls: list[dict] = []
        self._ctx = CyclePlanContext(session_date="", cycle_index=0)
        self.usage = LlmUsageEstimate(
            model=DEFAULT_SIM_LLM_MODEL if model == "backtest-sim" else model,
        )

    @property
    def estimated_cost_usd(self) -> float:
        return self.usage.estimated_cost_usd

    @property
    def prompt_tokens(self) -> int:
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.usage.completion_tokens

    def prepare_cycle(self, ctx: CyclePlanContext) -> None:
        self._ctx = ctx
        turns = self._build_turns()
        self._response_queues.append(turns)
        for turn in turns:
            u = turn.usage or _Usage()
            self.usage.add(u.prompt_tokens, u.completion_tokens, samples=1)

    def _trade_score_threshold(self) -> float:
        """Min |composite| to attempt a sim trade (aligned with runner default composite)."""
        raw = os.getenv("ABC_SIM_TRADE_SCORE_THRESHOLD")
        if raw is not None:
            try:
                return max(COMPOSITE_TRADE_THRESHOLD, float(raw))
            except ValueError:
                pass
        try:
            default = float(os.getenv("ABC_SIM_DEFAULT_COMPOSITE_SCORE", "0.72"))
        except ValueError:
            default = 0.72
        return max(COMPOSITE_TRADE_THRESHOLD, default)

    def _done_cooldown_seconds(self) -> int:
        lc = get_loop_config()
        return lc.clamp_cooldown(lc.react_cooldown_min_seconds)

    def _default_stop_distance_pct(self) -> float:
        raw = os.getenv("ABC_SIM_STOP_DISTANCE_PCT")
        if raw is not None:
            try:
                return max(0.5, float(raw))
            except ValueError:
                pass
        return 2.0

    def _build_turns(self) -> list[_Response]:
        turns: list[_Response] = []
        turns.append(
            _Response(
                content=json.dumps({"action": "quality_status"}),
                usage=_Usage(800, 120),
            )
        )
        threshold = self._trade_score_threshold()
        stop_pct = self._default_stop_distance_pct()
        cooldown = self._done_cooldown_seconds()
        if (
            self._ctx.top_symbol
            and self._ctx.top_score >= threshold
            and not self._ctx.traded_today
            and self._ctx.cycle_index == 0
        ):
            sym = self._ctx.top_symbol
            turns.append(
                _Response(
                    content=json.dumps(
                        {
                            "action": "calculate_size",
                            "symbol": sym,
                            "side": "BUY",
                            "stop_distance_pct": stop_pct,
                        }
                    ),
                    usage=_Usage(600, 80),
                )
            )
            turns.append(
                _Response(
                    content=json.dumps(
                        {
                            "action": "plan_order",
                            "symbol": sym,
                            "side": "BUY",
                            "quantity": 1,
                            "execute": True,
                            "intent": "entry",
                            "stop_distance_pct": stop_pct,
                        }
                    ),
                    usage=_Usage(700, 100),
                )
            )
        turns.append(
            _Response(
                content=json.dumps(
                    {
                        "action": "done",
                        "summary": f"sim cycle {self._ctx.cycle_index} {self._ctx.session_date}",
                        "cooldown": cooldown,
                    }
                ),
                usage=_Usage(400, 60),
            )
        )
        return turns

    def next_responses(self) -> list[_Response]:
        if self._response_queues:
            return self._response_queues.pop(0)
        return [
            _Response(
                content=json.dumps(
                    {
                        "action": "done",
                        "summary": "empty queue",
                        "cooldown": self._done_cooldown_seconds(),
                    }
                ),
                usage=_Usage(10, 5),
            )
        ]
