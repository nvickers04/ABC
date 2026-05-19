"""Shared backtest evaluation for grid and genetic profile optimizers."""

from __future__ import annotations

import time
from typing import Any

from core.central_profit_config import ReplayDataProvider, load_cached_profit_profile, simulate_backtest
from core.profile_optimization import ProfileCandidate, apply_config_patches, score_backtest_result


def build_composed_for_candidate(
    cand: ProfileCandidate,
    *,
    dotenv: bool = False,
) -> Any:
    """Load profile + patches without mutating the process :class:`ProfitConfig` singleton."""
    composed = load_cached_profit_profile(str(cand.base_profile), dotenv=dotenv)
    if cand.patches:
        composed = apply_config_patches(
            composed,
            cand.patches,
            profile_label=cand.candidate_id,
            sync_singleton=False,
        )
    return composed


def evaluate_optimizer_candidate(
    index: int,
    total: int,
    cand: ProfileCandidate,
    *,
    args: Any,
    start_date: str,
    end_date: str,
    shared_replay: ReplayDataProvider,
    reload_dotenv: bool,
) -> dict[str, Any]:
    """Run one optimizer candidate (sequential or ThreadPoolExecutor worker)."""
    cand_t0 = time.perf_counter()
    header = f"[{index}/{total}] {cand.candidate_id} (base={cand.base_profile}) ..."
    try:
        composed = build_composed_for_candidate(cand, dotenv=reload_dotenv)
        result = simulate_backtest(
            cand.base_profile,
            start_date,
            end_date,
            initial_cash=float(args.initial_cash),
            cycles_per_day=int(args.cycles_per_day),
            candidate_id=cand.candidate_id,
            reload_dotenv=False,
            replay_data=shared_replay,
            composed=composed,
        )
        metrics = score_backtest_result(result, cycles_per_day=args.cycles_per_day)
        est_cost = 0.0
        timing = ""
        if result.run_stats is not None:
            est_cost = float(result.run_stats.llm_usage.estimated_cost_usd)
            rs = result.run_stats
            timing = (
                f" | {rs.elapsed_sec:.1f}s "
                f"tok={rs.llm_usage.prompt_tokens}+{rs.llm_usage.completion_tokens} "
                f"est=${rs.llm_usage.estimated_cost_usd:.4f}"
            )
        detail = (
            f"    composite={metrics['composite_score']:.3f} "
            f"sharpe={metrics['sharpe_ratio']:.2f} "
            f"pf={metrics['profit_factor']:.2f} "
            f"win%={metrics['win_rate_pct']:.1f} "
            f"pnl=${metrics['total_profit_usd']:+,.2f}"
            f"{timing} ({time.perf_counter() - cand_t0:.1f}s wall)"
        )
        return {
            "header": header,
            "detail": detail,
            "ranking": {
                "candidate_id": cand.candidate_id,
                "base_profile": cand.base_profile,
                "patches": cand.patches,
                "metrics": metrics,
                "simulation_notes": list(result.notes),
            },
            "error": None,
            "est_cost": est_cost,
        }
    except Exception as exc:
        return {
            "header": header,
            "detail": f"    FAILED: {exc}",
            "ranking": None,
            "error": {
                "candidate_id": cand.candidate_id,
                "error": str(exc),
                "exc": exc,
            },
            "est_cost": 0.0,
        }


__all__ = [
    "build_composed_for_candidate",
    "evaluate_optimizer_candidate",
]
