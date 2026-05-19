"""LLM cost / token estimates for simulation and optimization (no live API spend)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from data.cost_tracker import MODEL_COSTS, estimate_llm_cost_usd

logger = logging.getLogger(__name__)

# Pricing model for "what-if live" estimates (BacktestLLM uses scripted turns).
DEFAULT_SIM_LLM_MODEL = os.getenv("ABC_SIM_LLM_PRICE_MODEL", "grok-4.3").strip() or "grok-4.3"

# Per-cycle upper bound from BacktestLLM max turn plan (quality + size + plan + done).
_MAX_PROMPT_PER_CYCLE = 800 + 600 + 700 + 400
_MAX_COMPLETION_PER_CYCLE = 120 + 80 + 100 + 60

# Default caps for estimated live-equivalent spend across a run / optimizer sweep.
DEFAULT_MAX_ESTIMATED_LLM_USD = float(os.getenv("ABC_SIM_MAX_ESTIMATED_LLM_USD", "25.0"))
DEFAULT_OPTIMIZER_MAX_ESTIMATED_LLM_USD = float(
    os.getenv("ABC_OPTIMIZER_MAX_ESTIMATED_LLM_USD", "200.0")
)


class SimulationBudgetError(RuntimeError):
    """Estimated LLM cost for a simulation or optimizer sweep exceeds configured cap."""


@dataclass
class LlmUsageEstimate:
    """Token and USD estimates using the same pricing table as :mod:`data.cost_tracker`."""

    model: str = DEFAULT_SIM_LLM_MODEL
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sample_calls: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, *, samples: int = 1) -> float:
        """Accumulate usage; return marginal USD for this batch."""
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))
        self.sample_calls += max(0, int(samples))
        marginal = estimate_llm_cost_usd(
            self.model,
            prompt_tokens,
            completion_tokens,
        )
        self.estimated_cost_usd = estimate_llm_cost_usd(
            self.model,
            self.prompt_tokens,
            self.completion_tokens,
        )
        return marginal

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "sample_calls": self.sample_calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


@dataclass
class BacktestRunStats:
    """Timing and token metrics for one backtest run."""

    elapsed_sec: float = 0.0
    archive_elapsed_sec: float = 0.0
    cycles_elapsed_sec: float = 0.0
    llm_usage: LlmUsageEstimate = field(default_factory=LlmUsageEstimate)

    def as_dict(self) -> dict[str, object]:
        return {
            "elapsed_sec": round(self.elapsed_sec, 3),
            "archive_elapsed_sec": round(self.archive_elapsed_sec, 3),
            "cycles_elapsed_sec": round(self.cycles_elapsed_sec, 3),
            "llm": self.llm_usage.as_dict(),
        }


def max_estimated_llm_usd_per_backtest() -> float:
    raw = os.getenv("ABC_SIM_MAX_ESTIMATED_LLM_USD", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_ESTIMATED_LLM_USD


def max_estimated_llm_usd_optimizer() -> float:
    raw = os.getenv("ABC_OPTIMIZER_MAX_ESTIMATED_LLM_USD", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_OPTIMIZER_MAX_ESTIMATED_LLM_USD


def estimate_backtest_llm_upper_bound(
    trading_days: int,
    cycles_per_day: int,
    *,
    model: str | None = None,
) -> LlmUsageEstimate:
    """Conservative upper bound if every cycle used the max BacktestLLM turn plan."""
    cycles = max(0, trading_days) * max(1, min(cycles_per_day, 4))
    usage = LlmUsageEstimate(model=model or DEFAULT_SIM_LLM_MODEL)
    usage.add(_MAX_PROMPT_PER_CYCLE * cycles, _MAX_COMPLETION_PER_CYCLE * cycles, samples=cycles)
    return usage


def estimate_optimizer_llm_upper_bound(
    candidate_count: int,
    trading_days: int,
    cycles_per_day: int,
    *,
    model: str | None = None,
) -> LlmUsageEstimate:
    """Upper bound for a full grid/GA sweep (each candidate = one backtest)."""
    per = estimate_backtest_llm_upper_bound(trading_days, cycles_per_day, model=model)
    total = LlmUsageEstimate(model=per.model)
    n = max(0, candidate_count)
    total.add(per.prompt_tokens * n, per.completion_tokens * n, samples=per.sample_calls * n)
    return total


def check_backtest_budget(
    usage: LlmUsageEstimate,
    *,
    trading_days: int,
    cycles_per_day: int,
    profile_max_daily_llm: float | None = None,
) -> None:
    """Raise if estimated live-equivalent cost exceeds caps (sim uses $0 API; this is guardrail)."""
    cap = max_estimated_llm_usd_per_backtest()
    if profile_max_daily_llm is not None and profile_max_daily_llm > 0:
        # Multi-day backtest vs single-day live cap (planning heuristic).
        day_cap = float(profile_max_daily_llm) * max(1, trading_days)
        cap = min(cap, day_cap) if cap > 0 else day_cap
    if cap > 0 and usage.estimated_cost_usd > cap:
        raise SimulationBudgetError(
            f"Estimated LLM cost ${usage.estimated_cost_usd:.4f} exceeds "
            f"backtest cap ${cap:.2f} ({trading_days} days × {cycles_per_day} cycles/day). "
            f"Set ABC_SIM_MAX_ESTIMATED_LLM_USD to raise or reduce window/cycles."
        )


def check_optimizer_budget(
    usage: LlmUsageEstimate,
    *,
    candidate_count: int,
) -> None:
    cap = max_estimated_llm_usd_optimizer()
    if cap > 0 and usage.estimated_cost_usd > cap:
        raise SimulationBudgetError(
            f"Optimizer estimated LLM cost ${usage.estimated_cost_usd:.2f} exceeds "
            f"cap ${cap:.2f} for {candidate_count} candidates. "
            f"Use --quick, fewer --days, lower --cycles-per-day, or raise "
            f"ABC_OPTIMIZER_MAX_ESTIMATED_LLM_USD."
        )


def format_run_summary(
    *,
    label: str,
    stats: BacktestRunStats,
    trading_days: int,
    cycles_run: int,
) -> str:
    u = stats.llm_usage
    return (
        f"{label}: {stats.elapsed_sec:.2f}s total "
        f"(archives {stats.archive_elapsed_sec:.2f}s, cycles {stats.cycles_elapsed_sec:.2f}s) | "
        f"{cycles_run} cycles / {trading_days} sessions | "
        f"tokens in={u.prompt_tokens} out={u.completion_tokens} "
        f"est_live=${u.estimated_cost_usd:.4f} ({u.model})"
    )
