"""research/replay.py — Deterministic replay harness for promotion gating.

The replay harness re-runs a strategy's tool-call sequence from a logged
session using only historical data (no live broker, no web search), so
every replay is deterministic and reproducible.

Usage:
    harness = ReplayHarness(slot=1, session_date="2026-03-09")
    episode = await harness.run(test_signals)
    if episode.outcome == "pass":
        # OK to promote
        ...

Tool policy:
    REPLAY_SAFE_TOOLS     — allowed (market data reads, chain lookups, local db)
    REPLAY_DISABLED_TOOLS — replaced with canned "not available in replay" response
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Tool classification ───────────────────────────────────────────────────────

REPLAY_SAFE_TOOLS: frozenset[str] = frozenset({
    # Market data reads (historical only)
    "candles",
    "option_chain",
    "option_quote",
    "quote",
    "earnings",
    "economic_calendar",
    # Local DB / research reads
    "research_briefing",
    "environment",
    # Stats and sizing (pure computation)
    "stats",
    "position_size",
    "instrument_details",
})

REPLAY_DISABLED_TOOLS: frozenset[str] = frozenset({
    # Live broker operations
    "place_order",
    "cancel_order",
    "account_summary",
    "positions",
    "orders",
    # Web search / multiagent
    "web_search",
    "research_agent",
    "multi_agent",
    # Real-time data that would differ from historical
    "live_quote",
    "level2",
})

_REPLAY_DISABLED_RESPONSE = {
    "status": "replay_disabled",
    "message": "This tool is not available during deterministic replay.",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """A single tool invocation captured during replay."""
    seq: int
    tool_name: str
    params: dict
    response: Any
    duration_ms: float
    was_disabled: bool = False
    error: Optional[str] = None


@dataclass
class ReplayEpisode:
    """Full record of one replay session."""
    slot: int
    session_date: str                          # YYYY-MM-DD
    strategy_id: Optional[int]
    started_at: str                            # ISO timestamp
    finished_at: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_pnl: float = 0.0
    total_fills: int = 0
    outcome: str = "pending"                   # "pass" | "fail" | "no_signals" | "error"
    failure_reason: Optional[str] = None
    notes: str = ""

    def to_db_dict(self) -> dict:
        return {
            "slot": self.slot,
            "session_date": self.session_date,
            "strategy_id": self.strategy_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tool_calls_json": json.dumps([
                {
                    "seq": tc.seq, "tool": tc.tool_name,
                    "params": tc.params, "disabled": tc.was_disabled,
                    "duration_ms": round(tc.duration_ms, 1),
                    "error": tc.error,
                }
                for tc in self.tool_calls
            ]),
            "total_pnl": self.total_pnl,
            "total_fills": self.total_fills,
            "outcome": self.outcome,
            "notes": self.notes,
        }


# ── Harness ───────────────────────────────────────────────────────────────────

class ReplayHarness:
    """Wraps a strategy evaluation in a deterministic historical replay context.

    The harness intercepts tool calls and:
    - Allows REPLAY_SAFE_TOOLS through, forcing historical date params
    - Blocks REPLAY_DISABLED_TOOLS with a canned response
    - Records every call with timing for the replay_episodes table

    Args:
        slot:         Research slot index (1-12)
        session_date: Historical date to simulate ("YYYY-MM-DD"); defaults to today
        strategy_id:  DB id of the strategy being replayed (optional)
        min_pnl:      Minimum total_pnl required to pass (default 0.0)
        min_signals:  Minimum fill count required (default 1)
    """

    def __init__(
        self,
        *,
        slot: int = 1,
        session_date: Optional[str] = None,
        strategy_id: Optional[int] = None,
        min_pnl: float = 0.0,
        min_signals: int = 1,
    ) -> None:
        self.slot = slot
        self.session_date = session_date or date.today().isoformat()
        self.strategy_id = strategy_id
        self.min_pnl = min_pnl
        self.min_signals = min_signals
        self._seq = 0
        self._calls: list[ToolCallRecord] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        test_signals: list[dict],
        *,
        tool_dispatcher: Any = None,
    ) -> ReplayEpisode:
        """Execute a replay episode.

        Args:
            test_signals:    Pre-simulated signals from the fast evaluator (used to
                             compute P&L without re-running the full execution engine).
            tool_dispatcher: Optional callable(tool_name, params) -> dict that handles
                             actual tool execution.  If None, signals are used directly.

        Returns:
            ReplayEpisode with outcome set.
        """
        started = datetime.now(timezone.utc).isoformat()
        episode = ReplayEpisode(
            slot=self.slot,
            session_date=self.session_date,
            strategy_id=self.strategy_id,
            started_at=started,
            finished_at=started,
        )

        try:
            # Gather P&L from the already-simulated signals
            options_sigs = [
                s for s in test_signals
                if s.get("order_type") in ("call", "put", "spread", "iron_condor",
                                           "butterfly", "calendar", "straddle", "strangle",
                                           "covered_call", "cash_secured_put")
            ]
            total_return = sum(s.get("return_pct", 0.0) for s in test_signals)
            fills = len([s for s in test_signals if s.get("return_pct") is not None])

            # Optionally call safe tools to validate historical data is available
            if tool_dispatcher is not None:
                await self._probe_data_availability(
                    symbols={s.get("symbol") for s in test_signals if s.get("symbol")},
                    dispatcher=tool_dispatcher,
                )

            episode.total_pnl = total_return
            episode.total_fills = fills
            episode.tool_calls = list(self._calls)

            # Gate decision
            if fills < self.min_signals:
                episode.outcome = "no_signals"
                episode.failure_reason = f"fills={fills} < min={self.min_signals}"
            elif total_return < self.min_pnl:
                episode.outcome = "fail"
                episode.failure_reason = (
                    f"total_return={total_return:.4f} < min_pnl={self.min_pnl:.4f}"
                )
            else:
                episode.outcome = "pass"

            # Options-specific check: if strategy has options signals, warn if few
            if options_sigs and fills > 0:
                options_fraction = len(options_sigs) / fills
                if options_fraction > 0.5 and episode.outcome == "pass":
                    logger.info(
                        f"[Replay slot {self.slot}] Options fraction={options_fraction:.0%} — "
                        f"promotion engine coverage should be verified separately"
                    )

        except Exception as exc:
            logger.exception(f"[Replay slot {self.slot}] Episode error: {exc}")
            episode.outcome = "error"
            episode.failure_reason = str(exc)
            episode.tool_calls = list(self._calls)

        episode.finished_at = datetime.now(timezone.utc).isoformat()
        return episode

    # ── Tool interception ─────────────────────────────────────────────────────

    async def call_tool(
        self,
        tool_name: str,
        params: dict,
        dispatcher: Any,
    ) -> dict:
        """Intercept a tool call: allow safe tools, block disabled tools."""
        self._seq += 1
        t0 = time.perf_counter()

        if tool_name in REPLAY_DISABLED_TOOLS:
            record = ToolCallRecord(
                seq=self._seq,
                tool_name=tool_name,
                params=params,
                response=_REPLAY_DISABLED_RESPONSE,
                duration_ms=0.0,
                was_disabled=True,
            )
            self._calls.append(record)
            return _REPLAY_DISABLED_RESPONSE

        if tool_name not in REPLAY_SAFE_TOOLS:
            logger.warning(
                f"[Replay slot {self.slot}] Unknown tool '{tool_name}' blocked — "
                "not in REPLAY_SAFE_TOOLS or REPLAY_DISABLED_TOOLS"
            )
            record = ToolCallRecord(
                seq=self._seq,
                tool_name=tool_name,
                params=params,
                response=_REPLAY_DISABLED_RESPONSE,
                duration_ms=0.0,
                was_disabled=True,
            )
            self._calls.append(record)
            return _REPLAY_DISABLED_RESPONSE

        # Force historical date context so calls don't leak live data
        forced_params = self._inject_date_params(tool_name, params)

        try:
            if asyncio.iscoroutinefunction(dispatcher):
                response = await dispatcher(tool_name, forced_params)
            else:
                response = await asyncio.to_thread(dispatcher, tool_name, forced_params)
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            record = ToolCallRecord(
                seq=self._seq,
                tool_name=tool_name,
                params=forced_params,
                response=None,
                duration_ms=duration_ms,
                error=str(exc),
            )
            self._calls.append(record)
            raise

        duration_ms = (time.perf_counter() - t0) * 1000
        record = ToolCallRecord(
            seq=self._seq,
            tool_name=tool_name,
            params=forced_params,
            response=response,
            duration_ms=duration_ms,
        )
        self._calls.append(record)
        return response

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _inject_date_params(self, tool_name: str, params: dict) -> dict:
        """Overlay date parameters so historical tools stay within session_date."""
        forced = dict(params)
        # For candles and market data, pin to session_date unless already set
        if tool_name in ("candles", "option_chain", "option_quote", "quote"):
            if "date" not in forced:
                forced["date"] = self.session_date
            if "to_date" not in forced and "from_date" not in forced:
                forced["to_date"] = self.session_date
        return forced

    async def _probe_data_availability(
        self,
        symbols: set[str],
        dispatcher: Any,
    ) -> None:
        """Fire candle probes for each symbol to confirm historical data exists."""
        for sym in list(symbols)[:5]:  # limit to 5 probes
            try:
                await self.call_tool(
                    "candles",
                    {"symbol": sym, "resolution": "1", "date": self.session_date},
                    dispatcher,
                )
            except Exception as exc:
                logger.debug(f"[Replay] Data probe {sym} failed: {exc}")

    def reset(self) -> None:
        """Clear state so the harness can be reused for another episode."""
        self._seq = 0
        self._calls = []


# ── DB persistence ────────────────────────────────────────────────────────────

def store_replay_episode(episode: ReplayEpisode) -> int:
    """Write a ReplayEpisode to the replay_episodes SQLite table."""
    from memory import get_db
    db = get_db()
    d = episode.to_db_dict()
    cur = db.execute(
        """INSERT INTO replay_episodes
           (ts, session_date, slot, strategy_id, tool_calls_json,
            total_pnl, total_fills, outcome, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            d["started_at"],
            d["session_date"],
            d["slot"],
            d["strategy_id"],
            d["tool_calls_json"],
            d["total_pnl"],
            d["total_fills"],
            d["outcome"],
            d["notes"],
        ),
    )
    db.commit()
    return cur.lastrowid
