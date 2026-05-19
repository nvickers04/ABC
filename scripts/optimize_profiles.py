#!/usr/bin/env python3
"""Grid-search or genetic optimization of ProfitConfig profiles via historical simulation.

Each candidate uses :func:`core.optimizer_backtest.evaluate_optimizer_candidate` with a
:class:`~core.central_profit_config.ComposedProfitConfig` (thread-local when ``--parallel``
is set) and shared :class:`~core.central_profit_config.ReplayDataProvider` archives.

Grid mode (default) runs built-in profiles plus perturbations (``--quick`` for three only).
Genetic mode (``--genetic``) evolves risk/loop/memory/prompt levers across generations.

Examples::

    python scripts/optimize_profiles.py --days 30
    python scripts/optimize_profiles.py --days 14 --quick --parallel 2
    python scripts/optimize_profiles.py --days 30 --cycles-per-day 1
    python scripts/optimize_profiles.py --genetic --generations 25 --save-profile evolved_v1

See ``docs/simulation-and-optimization.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize profitability profiles via historical backtest",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Calendar lookback from today (default: 30)",
    )
    parser.add_argument(
        "--end",
        metavar="YYYY-MM-DD",
        default=None,
        help="Inclusive end date (default: today)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="JSON output path (default: data/profile_optimization.json or data/genetic_optimization.json)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Starting equity per simulation (default: 100000)",
    )
    parser.add_argument(
        "--cycles-per-day",
        type=int,
        default=1,
        metavar="N",
        help="Agent cycles per session day (default: 1, max 4)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grid only: skip perturbations (three built-in profiles)",
    )
    parser.add_argument(
        "--baseline",
        default="balanced",
        choices=("conservative", "balanced", "aggressive"),
        help="Profile to diff recommended changes against (default: balanced)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Include top N ranked candidates in JSON (default: 5)",
    )
    # Genetic algorithm
    parser.add_argument(
        "--genetic",
        action="store_true",
        help="Run genetic search over key levers (conservative/balanced/aggressive seeds)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=25,
        metavar="N",
        help="GA generations (default: 25; use 20–30)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=12,
        help="GA population size (default: 12)",
    )
    parser.add_argument(
        "--elite",
        type=int,
        default=2,
        help="Elite individuals preserved each generation (default: 2)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.25,
        help="Per-gene mutation probability (default: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible GA",
    )
    parser.add_argument(
        "--save-profile",
        metavar="NAME",
        default=None,
        help="Save best genome as evolved profile (data/evolved_profiles.json)",
    )
    parser.add_argument(
        "--emit-snippet",
        metavar="PATH",
        default=None,
        help="Write profit_profiles.py paste snippet for the saved profile",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        nargs="?",
        const=2,
        default=None,
        metavar="N",
        help=(
            "Run candidate backtests in parallel (2–4 workers; bare --parallel uses 2). "
            "Omit for sequential (default)."
        ),
    )
    args = parser.parse_args(argv)
    if args.cycles_per_day < 1 or args.cycles_per_day > 4:
        parser.error("--cycles-per-day must be between 1 and 4 (inclusive)")
    if args.parallel is not None and (args.parallel < 1 or args.parallel > 4):
        parser.error("--parallel must be between 1 and 4 (inclusive)")
    return args


def _default_output(genetic: bool) -> str:
    return "data/genetic_optimization.json" if genetic else "data/profile_optimization.json"


def _trading_day_count(start_date: str, end_date: str) -> int:
    from core.simulation.runner import _trading_days

    return len(_trading_days(start_date, end_date))


def _parallel_workers(args: argparse.Namespace) -> int:
    """Worker count when ``--parallel`` was passed; otherwise 1 (sequential)."""
    parallel = getattr(args, "parallel", None)
    if parallel is None:
        return 1
    return max(1, min(4, int(parallel)))


def _parallel_explicitly_set(args: argparse.Namespace) -> bool:
    return getattr(args, "parallel", None) is not None


def run_grid_optimization(args: argparse.Namespace) -> dict:
    import asyncio
    import logging
    import time

    from core.central_profit_config import (
        clear_profit_profile_cache,
        clear_shared_replay_data,
        get_shared_replay_data,
    )
    from core.profit_config_state import log_active_profit_config, validate_backtest_date_range
    from core.simulation.llm_cost_estimate import (
        check_optimizer_budget,
        estimate_optimizer_llm_upper_bound,
    )
    from core.optimizer_backtest import evaluate_optimizer_candidate
    from core.profile_optimization import (
        COMPOSITE_FORMULA_DOC,
        REFERENCE_CYCLES_PER_DAY,
        SAFETY_GATES_ACTIVE,
        ProfileCandidate,
        build_candidate_grid,
        diff_against_baseline,
        lookback_dates,
    )
    from core.profit_profiles import normalize_profit_profile

    log = logging.getLogger(__name__)
    start_date, end_date = lookback_dates(args.days, end_date=args.end)
    start_date, end_date = validate_backtest_date_range(start_date, end_date)
    baseline = normalize_profit_profile(args.baseline)
    clear_profit_profile_cache()
    candidates = build_candidate_grid(include_perturbations=not args.quick, use_cache=True)

    rankings: list[dict] = []
    errors: list[dict] = []
    trading_days = _trading_day_count(start_date, end_date)
    budget = estimate_optimizer_llm_upper_bound(
        len(candidates),
        trading_days,
        args.cycles_per_day,
    )
    check_optimizer_budget(budget, candidate_count=len(candidates))

    print(f"Profile optimization: {start_date} -> {end_date} ({len(candidates)} candidates)")
    log.info(
        "grid optimization cycles_per_day=%d reference_cycles_per_day=%d",
        args.cycles_per_day,
        REFERENCE_CYCLES_PER_DAY,
    )
    print(
        f"cycles_per_day={args.cycles_per_day} "
        f"(composite reference={REFERENCE_CYCLES_PER_DAY})"
    )
    print(
        f"Est. live LLM upper bound (all candidates): ${budget.estimated_cost_usd:.2f} "
        f"({budget.prompt_tokens:,} in / {budget.completion_tokens:,} out tokens)"
    )
    print("Safety gates: " + "; ".join(SAFETY_GATES_ACTIVE[:3]) + "; ...")
    print()

    clear_shared_replay_data()
    shared_replay = get_shared_replay_data(start_date, end_date)
    prefetch = asyncio.run(shared_replay.load())
    print(
        f"Loaded shared replay for {prefetch['symbol_count']} symbols in "
        f"{prefetch['elapsed_sec']:.2f}s"
    )
    if prefetch.get("missing_symbols"):
        print(f"  Warning: empty bars for {', '.join(prefetch['missing_symbols'])}")
    print()

    workers = _parallel_workers(args)
    if _parallel_explicitly_set(args):
        log.info("grid optimization parallel workers=%d", workers)
        if workers > 2:
            log.warning(
                "parallel=%d may saturate CPU; 2 workers is recommended for backtests",
                workers,
            )
        print(f"Parallel backtests: {workers} workers")
    else:
        print("Sequential backtests (pass --parallel N to enable parallelism)")

    opt_started = time.perf_counter()
    total_est_live_cost = 0.0
    total = len(candidates)

    def _consume(outcome: dict) -> None:
        nonlocal total_est_live_cost
        print(outcome["header"], flush=True)
        print(outcome["detail"], flush=True)
        total_est_live_cost += outcome["est_cost"]
        if outcome["ranking"] is not None:
            rankings.append(outcome["ranking"])
        elif outcome["error"] is not None:
            err = outcome["error"]
            errors.append({"candidate_id": err["candidate_id"], "error": err["error"]})

    if workers <= 1:
        reload_env = True
        for i, cand in enumerate(candidates, 1):
            outcome = evaluate_optimizer_candidate(
                i,
                total,
                cand,
                args=args,
                start_date=start_date,
                end_date=end_date,
                shared_replay=shared_replay,
                reload_dotenv=reload_env,
            )
            reload_env = False
            if outcome["error"] is not None:
                err = outcome["error"]
                log_active_profit_config(
                    log,
                    "optimizer candidate failed",
                    exc=err.get("exc"),
                    candidate_id=err["candidate_id"],
                    base_profile=cand.base_profile,
                )
            _consume(outcome)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    evaluate_optimizer_candidate,
                    i,
                    total,
                    cand,
                    args=args,
                    start_date=start_date,
                    end_date=end_date,
                    shared_replay=shared_replay,
                    reload_dotenv=False,
                ): cand
                for i, cand in enumerate(candidates, 1)
            }
            for fut in as_completed(futures):
                outcome = fut.result()
                if outcome["error"] is not None:
                    err = outcome["error"]
                    cand = futures[fut]
                    log_active_profit_config(
                        log,
                        "optimizer candidate failed",
                        exc=err.get("exc"),
                        candidate_id=err["candidate_id"],
                        base_profile=cand.base_profile,
                    )
                _consume(outcome)

    if not rankings:
        raise SystemExit("All candidates failed; see errors in output JSON.")

    rankings.sort(
        key=lambda r: (
            r["metrics"]["composite_score"],
            r["metrics"]["sharpe_ratio"],
            r["metrics"]["total_profit_usd"],
        ),
        reverse=True,
    )
    best_entry = rankings[0]
    best_cand = ProfileCandidate(
        candidate_id=best_entry["candidate_id"],
        base_profile=best_entry["base_profile"],  # type: ignore[arg-type]
        patches=best_entry.get("patches") or {},
    )
    recommended = diff_against_baseline(best_cand, baseline_profile=baseline)  # type: ignore[arg-type]

    opt_elapsed = time.perf_counter() - opt_started
    log.info(
        "grid optimization finished in %.1fs | candidates=%s est_live_llm_total=$%.4f",
        opt_elapsed,
        len(rankings),
        total_est_live_cost,
    )

    return {
        "mode": "grid",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.days,
        "start_date": start_date,
        "end_date": end_date,
        "baseline_profile": baseline,
        "cycles_per_day": args.cycles_per_day,
        "reference_cycles_per_day": REFERENCE_CYCLES_PER_DAY,
        "composite_formula": COMPOSITE_FORMULA_DOC,
        "safety_gates_active": SAFETY_GATES_ACTIVE,
        "runtime": {
            "optimizer_elapsed_sec": round(opt_elapsed, 3),
            "parallel_workers": workers,
            "archive_prefetch": prefetch,
            "estimated_live_llm_upper_bound_usd": round(budget.estimated_cost_usd, 4),
            "estimated_live_llm_actual_usd": round(total_est_live_cost, 6),
        },
        "best": {
            "candidate_id": best_entry["candidate_id"],
            "base_profile": best_entry["base_profile"],
            "patches": best_entry.get("patches") or {},
            "metrics": best_entry["metrics"],
        },
        "recommended_config_changes": recommended,
        "rankings": rankings[: max(1, args.top)],
        "errors": errors,
    }


def run_genetic_optimization(args: argparse.Namespace) -> dict:
    import asyncio
    import logging
    import time

    from core.central_profit_config import clear_shared_replay_data, get_shared_replay_data
    from core.optimizer_backtest import evaluate_optimizer_candidate
    from core.profit_config_state import log_active_profit_config, validate_backtest_date_range
    from core.simulation.llm_cost_estimate import (
        check_optimizer_budget,
        estimate_optimizer_llm_upper_bound,
    )
    from core.profile_genetic import (
        GeneticSearchConfig,
        Genome,
        format_profit_profiles_snippet,
        genome_to_patches,
        run_genetic_search,
    )
    from core.profile_optimization import (
        COMPOSITE_FORMULA_DOC,
        REFERENCE_CYCLES_PER_DAY,
        SAFETY_GATES_ACTIVE,
        ProfileCandidate,
        lookback_dates,
    )
    from core.profit_profiles import (
        EvolvedProfileEntry,
        format_evolved_profile_python_block,
        save_evolved_profile,
    )

    log = logging.getLogger(__name__)
    start_date, end_date = lookback_dates(args.days, end_date=args.end)
    start_date, end_date = validate_backtest_date_range(start_date, end_date)
    generations = max(1, min(50, args.generations))
    pop = max(4, args.population)
    max_evals = generations * pop
    trading_days = _trading_day_count(start_date, end_date)
    budget = estimate_optimizer_llm_upper_bound(
        max_evals,
        trading_days,
        args.cycles_per_day,
    )
    check_optimizer_budget(budget, candidate_count=max_evals)

    print(f"Genetic profile search: {start_date} -> {end_date}")
    log.info(
        "genetic optimization cycles_per_day=%d reference_cycles_per_day=%d",
        args.cycles_per_day,
        REFERENCE_CYCLES_PER_DAY,
    )
    print(
        f"cycles_per_day={args.cycles_per_day} "
        f"(composite reference={REFERENCE_CYCLES_PER_DAY})"
    )
    print(
        f"Est. live LLM upper bound (~{max_evals} evals): ${budget.estimated_cost_usd:.2f} "
        f"({budget.prompt_tokens:,} in / {budget.completion_tokens:,} out tokens)"
    )
    print(f"  generations={generations} population={args.population} elite={args.elite}")
    print("  genes: RISK_PER_TRADE, confidence floors, WM/attention/summary caps")
    print("Safety gates: " + "; ".join(SAFETY_GATES_ACTIVE[:3]) + "; ...")
    print()

    clear_shared_replay_data()
    shared_replay = get_shared_replay_data(start_date, end_date)
    prefetch = asyncio.run(shared_replay.load())
    print(
        f"Loaded shared replay for {prefetch['symbol_count']} symbols in "
        f"{prefetch['elapsed_sec']:.2f}s\n"
    )

    ga_started = time.perf_counter()
    reload_env = True
    total_est_live_cost = 0.0

    def evaluate(genome: Genome) -> dict:
        nonlocal reload_env, total_est_live_cost
        cand_id = genome.candidate_id()
        cand = ProfileCandidate(
            candidate_id=cand_id,
            base_profile=genome.base_profile,
            patches=genome_to_patches(genome),
        )
        outcome = evaluate_optimizer_candidate(
            0,
            0,
            cand,
            args=args,
            start_date=start_date,
            end_date=end_date,
            shared_replay=shared_replay,
            reload_dotenv=reload_env,
        )
        reload_env = False
        print(
            f"  eval {cand_id} (gen={genome.generation} base={genome.base_profile}) ...",
            flush=True,
        )
        print(outcome["detail"], flush=True)
        if outcome["error"] is not None:
            err = outcome["error"]
            log_active_profit_config(
                log,
                "genetic optimizer candidate failed",
                exc=err.get("exc"),
                candidate_id=err["candidate_id"],
                base_profile=genome.base_profile,
            )
            raise err["exc"]
        total_est_live_cost += outcome["est_cost"]
        return outcome["ranking"]["metrics"]

    ga_cfg = GeneticSearchConfig(
        generations=generations,
        population_size=pop,
        elite_count=max(1, min(args.elite, args.population - 1)),
        mutation_rate=max(0.05, min(0.6, args.mutation_rate)),
        seed=args.seed,
    )
    search = run_genetic_search(evaluate, ga_cfg)
    best = search.best
    patches = genome_to_patches(best)

    ga_elapsed = time.perf_counter() - ga_started
    log.info(
        "genetic optimization finished in %.1fs | evals=%s est_live_llm_total=$%.4f",
        ga_elapsed,
        search.evaluated_count,
        total_est_live_cost,
    )

    payload: dict = {
        "mode": "genetic",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.days,
        "start_date": start_date,
        "end_date": end_date,
        "cycles_per_day": args.cycles_per_day,
        "reference_cycles_per_day": REFERENCE_CYCLES_PER_DAY,
        "composite_formula": COMPOSITE_FORMULA_DOC,
        "safety_gates_active": SAFETY_GATES_ACTIVE,
        "runtime": {
            "optimizer_elapsed_sec": round(ga_elapsed, 3),
            "parallel_workers": 1,
            "archive_prefetch": prefetch,
            "estimated_live_llm_upper_bound_usd": round(budget.estimated_cost_usd, 4),
            "estimated_live_llm_actual_usd": round(total_est_live_cost, 6),
            "evaluated_count": search.evaluated_count,
        },
        "genetic_config": {
            "generations": ga_cfg.generations,
            "population_size": ga_cfg.population_size,
            "elite_count": ga_cfg.elite_count,
            "mutation_rate": ga_cfg.mutation_rate,
            "seed": ga_cfg.seed,
            "evaluated_count": search.evaluated_count,
        },
        "generation_history": search.history,
        "best": {
            "genome_id": best.genome_id,
            "base_profile": best.base_profile,
            "generation": best.generation,
            "genes": best.genes,
            "patches": patches,
            "metrics": best.metrics,
            "candidate_id": best.candidate_id(),
        },
        "top_genomes": [
            {
                "genome_id": g.genome_id,
                "base_profile": g.base_profile,
                "fitness": g.fitness,
                "genes": g.genes,
            }
            for g in search.population_final[: max(1, args.top)]
        ],
    }

    profile_name = args.save_profile
    if profile_name:
        note = (
            f"Genetic search best over {start_date}..{end_date} "
            f"(composite={best.metrics.get('composite_score', 0):.3f})."
        )
        entry = EvolvedProfileEntry(
            base_profile=best.base_profile,
            patches=patches,
            note=note,
            genes=dict(best.genes),
            metrics=dict(best.metrics),
        )
        reg_path = save_evolved_profile(profile_name, entry)
        payload["saved_profile"] = {
            "name": profile_name.strip().lower(),
            "registry_path": str(reg_path),
            "usage": f"Set PROFIT_PROFILE={profile_name.strip().lower()}",
        }
        snippet = format_evolved_profile_python_block(profile_name.strip().lower(), entry)
        payload["profit_profiles_snippet"] = snippet
        if args.emit_snippet:
            snip_path = Path(args.emit_snippet)
            if not snip_path.is_absolute():
                snip_path = _REPO / snip_path
            snip_path.parent.mkdir(parents=True, exist_ok=True)
            snip_path.write_text(
                format_profit_profiles_snippet(
                    profile_name.strip().lower(),
                    best,
                    note=note,
                    metrics=best.metrics,
                )
                + "\n\n"
                + snippet,
                encoding="utf-8",
            )
            payload["saved_profile"]["snippet_path"] = str(snip_path)
        print(f"\nSaved evolved profile {profile_name!r} -> {reg_path}")
        print(f"  Use: set PROFIT_PROFILE={profile_name.strip().lower()}")

    return payload


def main(argv: list[str] | None = None) -> int:
    import os

    args = _parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO / ".env", override=True)
    except ImportError:
        pass

    if args.genetic:
        payload = run_genetic_optimization(args)
    else:
        payload = run_grid_optimization(args)

    out_path = args.output or _default_output(args.genetic)
    out = Path(out_path)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    best = payload["best"]
    score = best.get("metrics", {}).get("composite_score", best.get("fitness"))
    label = best.get("candidate_id") or best.get("genome_id")
    print()
    print(f"Best: {label} (composite={score})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
