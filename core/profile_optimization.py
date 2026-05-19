"""Profile grid search helpers for historical simulation optimization."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from core.profit_profile_cache import (
    clear_profit_profile_cache,
    load_cached_profit_profile,
    log_profit_profile_cache_summary,
)
from core.profit_profiles import PROFIT_PROFILE_ENV, ProfitProfile
from core.profit_profiles import _patch_model as patch_model
from core.simulation.sim_broker import SimFill
from core.simulation.types import BacktestResult

PatchDict = dict[str, dict[str, Any]]
RelativePatch = dict[str, dict[str, Callable[[Any], Any]]]

DEFAULT_CYCLES_PER_DAY = 1
REFERENCE_CYCLES_PER_DAY = 4
COMPOSITE_FORMULA_DOC = (
    "0.4*sharpe_norm + 0.3*profit_factor_norm + 0.3*win_rate_norm; "
    "pf/win scaled when cycles_per_day < reference (default reference=4)"
)

SAFETY_GATES_ACTIVE = [
    "QualityMatrix (risk multiplier / execution quality)",
    "SafetyController (daily loss, intraday drawdown, LLM cost)",
    "LoopConfig gap guard and posture caps",
    "RiskExecutionConfig spend and loss rails",
    "SimulatedBroker (no live IBKR orders)",
    "BacktestLLM (no live xAI spend)",
]


@dataclass(frozen=True)
class ProfileCandidate:
    """One simulation variant: base profile plus optional sub-config patches."""

    candidate_id: str
    base_profile: ProfitProfile
    patches: PatchDict = field(default_factory=dict)

    def label(self) -> str:
        return self.candidate_id


def apply_config_patches(
    cfg: Any,
    patches: PatchDict,
    *,
    profile_label: str | None = None,
    sync_singleton: bool = True,
) -> Any:
    """Apply ``risk`` / ``loop`` / ``memory`` / ``prompt`` overrides and sync runtime installs."""
    from dataclasses import replace

    from core.central_profit_config import _sync_core_config_exports, refresh_profit_config_cache
    from core.loop_config import install_loop_config
    from core.memory_config import install_memory_config
    from core.prompt_config import install_prompt_config
    from core.risk_execution_config import install_risk_execution_config
    from core.tool_registry import install_tool_registry

    from core.central_profit_config import ProfitConfig

    if isinstance(cfg, ProfitConfig):
        base = cfg.composed
        return_singleton = True
    else:
        base = cfg
        return_singleton = False

    if not patches:
        return cfg

    try:
        risk = patch_model(base.risk, patches.get("risk", {}))
        loop = patch_model(base.loop, patches.get("loop", {}))
        memory = patch_model(base.memory, patches.get("memory", {}))
        prompt = patch_model(base.prompt, patches.get("prompt", {}))
    except Exception as exc:
        from core.profit_config_state import log_active_profit_config

        log_active_profit_config(
            logging.getLogger(__name__),
            "apply_config_patches validation failed",
            cfg=cfg,
            exc=exc,
            patches=patches,
        )
        raise

    updated = replace(base, risk=risk, loop=loop, memory=memory, prompt=prompt)
    if sync_singleton:
        install_risk_execution_config(risk)
        install_memory_config(memory)
        install_loop_config(loop)
        install_prompt_config(prompt)
        install_tool_registry(base.tools)
        _sync_core_config_exports(risk)
        refresh_profit_config_cache(updated)
        if profile_label:
            from core.profit_config_state import set_active_profile_label

            set_active_profile_label(profile_label)
    if return_singleton:
        return cfg
    return updated


def _load_profile_config(base_profile: ProfitProfile, *, dotenv: bool = False) -> Any:
    return load_cached_profit_profile(str(base_profile), dotenv=dotenv)


def resolve_relative_patches(
    base_profile: ProfitProfile,
    relative: RelativePatch,
) -> PatchDict:
    """Turn multipliers/lambdas into concrete patch dicts from a loaded base profile."""
    cfg = load_cached_profit_profile(str(base_profile), dotenv=False)
    out: PatchDict = {}
    section_map = {
        "risk": cfg.risk,
        "loop": cfg.loop,
        "memory": cfg.memory,
        "prompt": cfg.prompt,
    }
    for section, fields in relative.items():
        model = section_map.get(section)
        if model is None:
            continue
        sec: dict[str, Any] = {}
        for fname, fn in fields.items():
            sec[fname] = fn(getattr(model, fname))
        if sec:
            out[section] = sec
    return out


def build_candidate_grid(
    *,
    include_perturbations: bool = True,
    dotenv: bool = False,
    use_cache: bool = True,
) -> list[ProfileCandidate]:
    """Base profiles plus small perturbations on top of each."""
    if not use_cache:
        clear_profit_profile_cache()

    def _profile_cfg(base: ProfitProfile) -> Any:
        if use_cache:
            return load_cached_profit_profile(str(base), dotenv=dotenv)
        from core.central_profit_config import get_profit_config

        os.environ[PROFIT_PROFILE_ENV] = str(base)
        return get_profit_config().reload(dotenv=dotenv)

    candidates: list[ProfileCandidate] = [
        ProfileCandidate(candidate_id=p, base_profile=p, patches={})  # type: ignore[arg-type]
        for p in ("conservative", "balanced", "aggressive")
    ]
    if not include_perturbations:
        if use_cache:
            log_profit_profile_cache_summary(prefix="build_candidate_grid")
        return candidates

    _cons_cfg = _profile_cfg("conservative")
    _bal_cfg = _profile_cfg("balanced")

    def _rel(base: ProfitProfile, cid: str, rel: RelativePatch) -> ProfileCandidate:
        return ProfileCandidate(
            candidate_id=cid,
            base_profile=base,
            patches=resolve_relative_patches(base, rel),
        )

    # Conservative: slightly tighter spend / higher entry bar
    candidates.extend(
        [
            _rel(
                "conservative",
                "conservative_llm_tight",
                {"risk": {"max_daily_llm_cost": lambda v: round(v * 0.85, 2)}},
            ),
            _rel(
                "conservative",
                "conservative_rm_gate_up",
                {
                    "loop": {
                        "min_rm_for_new_risk": lambda v: min(
                            round(v + 0.05, 2),
                            _cons_cfg.loop.limited_posture_rm_cap - 0.01,
                        )
                    }
                },
            ),
        ]
    )

    # Balanced: spend and R:R sweeps
    candidates.extend(
        [
            _rel(
                "balanced",
                "balanced_llm_tight",
                {"risk": {"max_daily_llm_cost": lambda v: round(v * 0.85, 2)}},
            ),
            _rel(
                "balanced",
                "balanced_llm_loose",
                {"risk": {"max_daily_llm_cost": lambda v: round(v * 1.15, 2)}},
            ),
            _rel(
                "balanced",
                "balanced_loss_tight",
                {"risk": {"max_daily_loss_pct": lambda v: round(v * 0.9, 2)}},
            ),
            _rel(
                "balanced",
                "balanced_min_rr_up",
                {"prompt": {"min_rr_paper": lambda v: round(v + 0.25, 2)}},
            ),
            _rel(
                "balanced",
                "balanced_rm_gate_up",
                {
                    "loop": {
                        "min_rm_for_new_risk": lambda v: min(
                            round(v + 0.05, 2),
                            _bal_cfg.loop.limited_posture_rm_cap - 0.01,
                        )
                    }
                },
            ),
        ]
    )

    # Aggressive: safety-biased variants (still on aggressive base)
    candidates.extend(
        [
            _rel(
                "aggressive",
                "aggressive_llm_cap",
                {"risk": {"max_daily_llm_cost": lambda v: round(v * 0.9, 2)}},
            ),
            _rel(
                "aggressive",
                "aggressive_loss_tight",
                {"risk": {"max_daily_loss_pct": lambda v: round(v * 0.92, 2)}},
            ),
            _rel(
                "aggressive",
                "aggressive_min_rr_up",
                {"prompt": {"min_rr_paper": lambda v: round(v + 0.2, 2)}},
            ),
        ]
    )

    if use_cache:
        log_profit_profile_cache_summary(prefix="build_candidate_grid")
    return candidates


def profit_factor_from_trades(trades: list[SimFill]) -> float:
    """Gross wins / gross losses; capped when there are no losses."""
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss <= 0:
        return min(3.0, gross_profit / 100.0) if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def profit_factor_from_result(result: BacktestResult) -> float:
    return float(result.profit_factor)


def normalize_metric(sharpe: float, profit_factor: float, win_rate_pct: float) -> dict[str, float]:
    """Map raw metrics to 0–1 for compositing."""
    return {
        "sharpe_norm": max(0.0, min(1.0, sharpe / 2.0)),
        "profit_factor_norm": max(0.0, min(1.0, profit_factor / 2.5)),
        "win_rate_norm": max(0.0, min(1.0, win_rate_pct / 100.0)),
    }


def composite_score(
    sharpe: float,
    profit_factor: float,
    win_rate_pct: float,
    *,
    cycles_per_day: float | None = None,
    reference_cycles_per_day: int = REFERENCE_CYCLES_PER_DAY,
) -> float:
    """0.4×Sharpe + 0.3×profit_factor + 0.3×win_rate (each normalized to 0–1).

    When ``cycles_per_day`` is below ``reference_cycles_per_day``, profit-factor and
    win-rate norms are scaled up (capped at 1) so rankings stay comparable across
    different cycle densities.
    """
    n = normalize_metric(sharpe, profit_factor, win_rate_pct)
    sharpe_term = 0.4 * n["sharpe_norm"]
    pf_term = 0.3 * n["profit_factor_norm"]
    win_term = 0.3 * n["win_rate_norm"]
    if cycles_per_day is None or cycles_per_day >= reference_cycles_per_day:
        return sharpe_term + pf_term + win_term
    ratio = max(0.25, float(cycles_per_day) / float(reference_cycles_per_day))
    pf_term = 0.3 * min(1.0, n["profit_factor_norm"] / ratio)
    win_term = 0.3 * min(1.0, n["win_rate_norm"] / ratio)
    return sharpe_term + pf_term + win_term


def score_backtest_result(
    result: BacktestResult,
    *,
    cycles_per_day: int | None = None,
) -> dict[str, Any]:
    pf = profit_factor_from_result(result)
    requested = cycles_per_day if cycles_per_day is not None else result.cycles_per_day
    if requested and requested > 0:
        effective = float(requested)
    elif result.trading_days > 0:
        effective = result.cycles_run / result.trading_days
    else:
        effective = float(DEFAULT_CYCLES_PER_DAY)
    raw = composite_score(result.sharpe_ratio, pf, result.win_rate)
    comp = composite_score(
        result.sharpe_ratio,
        pf,
        result.win_rate,
        cycles_per_day=effective,
        reference_cycles_per_day=REFERENCE_CYCLES_PER_DAY,
    )
    norms = normalize_metric(result.sharpe_ratio, pf, result.win_rate)
    scored: dict[str, Any] = {
        "composite_score": round(comp, 4),
        "composite_score_raw": round(raw, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "profit_factor": round(pf, 4),
        "win_rate_pct": round(result.win_rate, 2),
        "normalized": {k: round(v, 4) for k, v in norms.items()},
        "total_profit_usd": round(result.total_profit, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "trade_count": result.trade_count,
        "cycles_run": result.cycles_run,
        "cycles_per_day": int(requested) if requested else None,
        "cycles_per_day_effective": round(effective, 4),
        "reference_cycles_per_day": REFERENCE_CYCLES_PER_DAY,
        "llm_cost_usd": round(result.llm_cost_usd, 4),
    }
    if result.run_stats is not None:
        scored["runtime"] = result.run_stats.as_dict()
        scored["llm_prompt_tokens"] = result.run_stats.llm_usage.prompt_tokens
        scored["llm_completion_tokens"] = result.run_stats.llm_usage.completion_tokens
        scored["estimated_live_llm_cost_usd"] = round(
            result.run_stats.llm_usage.estimated_cost_usd, 6
        )
    return scored


def diff_against_baseline(
    candidate: ProfileCandidate,
    *,
    baseline_profile: ProfitProfile = "balanced",
) -> dict[str, Any]:
    """Recommended config deltas vs baseline profile (no extra patches)."""
    if candidate.base_profile == baseline_profile and not candidate.patches:
        return {"note": "baseline profile; no changes recommended"}

    baseline_cfg = _load_profile_config(baseline_profile)
    cand_cfg = apply_config_patches(_load_profile_config(candidate.base_profile), candidate.patches)

    changes: dict[str, dict[str, dict[str, Any]]] = {}
    if candidate.base_profile != baseline_profile:
        changes["profit_profile"] = {
            "from": baseline_profile,
            "to": candidate.base_profile,
        }

    for section in ("risk", "loop", "memory", "prompt"):
        base_model = getattr(baseline_cfg, section)
        cand_model = getattr(cand_cfg, section)
        sec: dict[str, Any] = {}
        for fname in type(base_model).model_fields:
            bval = getattr(base_model, fname)
            cval = getattr(cand_model, fname)
            if bval != cval:
                sec[fname] = {"from": bval, "to": cval}
        if sec:
            changes[section] = sec

    return changes


def lookback_dates(days: int, *, end_date: str | None = None) -> tuple[str, str]:
    """Inclusive ``YYYY-MM-DD`` window spanning at least ``days`` calendar days."""
    from datetime import date, timedelta

    end = date.fromisoformat(end_date) if end_date else date.today()
    start = end - timedelta(days=max(1, days) - 1)
    return start.isoformat(), end.isoformat()


__all__ = [
    "DEFAULT_CYCLES_PER_DAY",
    "REFERENCE_CYCLES_PER_DAY",
    "ProfileCandidate",
    "SAFETY_GATES_ACTIVE",
    "apply_config_patches",
    "build_candidate_grid",
    "clear_profit_profile_cache",
    "COMPOSITE_FORMULA_DOC",
    "composite_score",
    "diff_against_baseline",
    "lookback_dates",
    "profit_factor_from_trades",
    "resolve_relative_patches",
    "score_backtest_result",
]
