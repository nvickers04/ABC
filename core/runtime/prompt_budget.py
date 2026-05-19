"""Prompt size control and lightweight token estimates for the agent loop.

Uses a simple chars÷4 heuristic (no tiktoken dependency). QualityMatrix
posture lines are always preserved; narrative blocks are trimmed first.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from core.log_context import get_logger

logger = get_logger(__name__)

_CHARS_PER_TOKEN = 4.0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


# Soft caps (override via env)
CYCLE_WM_MAX_CHARS = _env_int("CYCLE_WM_MAX_CHARS", 2400)
CYCLE_ATTENTION_MAX_CHARS = _env_int("CYCLE_ATTENTION_MAX_CHARS", 1200)
CYCLE_INTUITION_TOP_N = _env_int("CYCLE_INTUITION_TOP_N", 3)
CYCLE_SNAPSHOT_MAX = _env_int("CYCLE_SNAPSHOT_MAX", 3)
CYCLE_SNAPSHOT_CHARS = _env_int("CYCLE_SNAPSHOT_CHARS", 100)
CYCLE_LAST_SUMMARY_CHARS = _env_int("CYCLE_LAST_SUMMARY_CHARS", 160)
CYCLE_GUIDANCE_MAX_CHARS = _env_int("CYCLE_GUIDANCE_MAX_CHARS", 400)


def estimate_tokens(text: str) -> int:
    """Rough token count from character length (~4 chars/token)."""
    if not text:
        return 0
    return max(1, int((len(text) + _CHARS_PER_TOKEN - 1) / _CHARS_PER_TOKEN))


def truncate_text(text: str, max_chars: int, *, marker: str = "\n…[trimmed]\n") -> str:
    """Truncate with a visible marker; no-op when ``max_chars <= 0``."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    room = max(80, max_chars - len(marker))
    return text[:room] + marker


@dataclass
class CyclePromptMetrics:
    """Per-cycle prompt component sizes (chars + estimated tokens)."""

    quality_chars: int = 0
    attention_chars: int = 0
    intuition_chars: int = 0
    wm_chars: int = 0
    state_chars: int = 0
    continuity_chars: int = 0
    inject_chars: int = 0
    guidance_chars: int = 0
    user_total_chars: int = 0
    system_chars: int = 0
    est_user_tokens: int = 0
    est_system_tokens: int = 0
    est_first_turn_tokens: int = 0
    react_turn: int = 0
    actual_prompt_tokens: int | None = None
    actual_completion_tokens: int | None = None
    actual_reasoning_tokens: int | None = None

    def record_user_total(self) -> None:
        self.user_total_chars = (
            self.quality_chars
            + self.attention_chars
            + self.intuition_chars
            + self.wm_chars
            + self.state_chars
            + self.continuity_chars
            + self.inject_chars
            + self.guidance_chars
        )
        self.est_user_tokens = estimate_tokens(" " * self.user_total_chars)
        self.est_first_turn_tokens = self.est_system_tokens + self.est_user_tokens

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "est_user_tok": self.est_user_tokens,
            "est_sys_tok": self.est_system_tokens,
            "est_turn_tok": self.est_first_turn_tokens,
            "qm": self.quality_chars,
            "attn": self.attention_chars,
            "intu": self.intuition_chars,
            "wm": self.wm_chars,
            "state": self.state_chars,
            "cont": self.continuity_chars,
            "react_turn": self.react_turn,
            "act_prompt": self.actual_prompt_tokens,
            "act_completion": self.actual_completion_tokens,
        }


def build_continuity_block(
    *,
    last_cycle_summary: str = "",
    last_wait_reason: str = "",
    last_wake_reason: str = "",
    market_snapshots: list[str] | None = None,
    max_snapshots: int = CYCLE_SNAPSHOT_MAX,
    max_snapshot_chars: int = CYCLE_SNAPSHOT_CHARS,
    max_summary_chars: int = CYCLE_LAST_SUMMARY_CHARS,
) -> str:
    """Compact rolling continuity (last cycle + wake + recent snapshots)."""
    parts: list[str] = []
    if last_cycle_summary:
        parts.append(f"LAST: {truncate_text(last_cycle_summary, max_summary_chars, marker='…')}")
    if last_wait_reason or last_wake_reason:
        wr = last_wait_reason or "—"
        wk = last_wake_reason or "timer"
        parts.append(f"WAIT:{truncate_text(wr, 80, marker='…')} | WAKE:{truncate_text(wk, 60, marker='…')}")
    snaps = market_snapshots or []
    if snaps:
        tail = snaps[-max_snapshots:]
        short = [
            truncate_text(s, max_snapshot_chars, marker="…")
            for s in tail
        ]
        parts.append("SNAPS: " + " | ".join(short))
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def log_cycle_prompt_budget(
    cycle_id: int,
    metrics: CyclePromptMetrics,
    *,
    level: int = logging.INFO,
) -> None:
    """One-line estimated prompt budget for the cycle (or ReAct turn)."""
    metrics.record_user_total()
    turn_note = f" turn={metrics.react_turn}" if metrics.react_turn else ""
    actual = ""
    if metrics.actual_prompt_tokens is not None:
        actual = (
            f" | actual_prompt={metrics.actual_prompt_tokens}"
            f" completion={metrics.actual_completion_tokens or 0}"
        )
        if metrics.actual_reasoning_tokens:
            actual += f" reasoning={metrics.actual_reasoning_tokens}"
    logger.log(
        level,
        "CYCLE %d tokens%s est: sys=~%d user=~%d first_turn=~%d | "
        "chars qm=%d attn=%d intu=%d wm=%d state=%d cont=%d%s",
        cycle_id,
        turn_note,
        metrics.est_system_tokens,
        metrics.est_user_tokens,
        metrics.est_first_turn_tokens,
        metrics.quality_chars,
        metrics.attention_chars,
        metrics.intuition_chars,
        metrics.wm_chars,
        metrics.state_chars,
        metrics.continuity_chars,
        actual,
    )


def attach_turn_usage(metrics: CyclePromptMetrics, usage: Any) -> None:
    """Copy xAI usage fields into metrics for logging."""
    if usage is None:
        return
    metrics.actual_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    metrics.actual_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    metrics.actual_reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0) or None
