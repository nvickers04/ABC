#!/usr/bin/env python3
"""End-to-end validation: ProfitConfig + simulate + logger + optimizer."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

START = "2024-06-03"
END = "2024-06-07"
OPT_JSON = _REPO / "data" / "e2e_validation_opt.json"


def main() -> int:
    os.environ["PROFIT_PROFILE"] = "balanced"
    os.environ["ABC_SIMULATION"] = "1"
    os.environ.pop("DATABASE_URL", None)

    from core.central_profit_config import get_profit_config, simulate_backtest
    from core.profit_cycle_logger import load_daily_entries
    from core.profit_profiles import PROFIT_PROFILE_ENV
    from core.profile_optimization import score_backtest_result

    print("=" * 60)
    print("ProfitConfig E2E validation")
    print("=" * 60)

    # 1) Load balanced ProfitConfig
    os.environ[PROFIT_PROFILE_ENV] = "balanced"
    cfg = get_profit_config().reload(dotenv=False)
    print(f"\n[1] ProfitConfig (balanced)")
    print(f"    trading_mode={cfg.trading_mode}")
    print(f"    max_daily_llm_cost={cfg.risk.max_daily_llm_cost}")
    print(f"    min_rm_for_new_risk={cfg.loop.min_rm_for_new_risk}")

    # 2) Short simulation
    print(f"\n[2] simulate_backtest {START} .. {END}")
    result = simulate_backtest("balanced", START, END, cycles_per_day=1)
    balanced_metrics = score_backtest_result(result)
    print(f"    cycles={result.cycles_run} trades={result.trade_count} pnl=${result.total_profit:+.2f}")
    print(f"    composite={balanced_metrics['composite_score']:.4f} sharpe={result.sharpe_ratio:.2f}")
    if result.run_stats:
        u = result.run_stats.llm_usage
        print(f"    est_live_llm=${u.estimated_cost_usd:.4f} tokens={u.prompt_tokens}+{u.completion_tokens}")

    # 3) Logger snapshot from today's JSON (sim cycles log to wall-clock date)
    log_date = date.today().isoformat()
    entries = load_daily_entries(log_date)
    sim_entries = [e for e in entries if "sim cycle" in (e.get("cycle_summary") or "")]
    print(f"\n[3] Profit cycle logger ({log_date})")
    print(f"    total_entries={len(entries)} sim_marked={len(sim_entries)}")
    ok_logger = False
    if sim_entries:
        last = sim_entries[-1]
        snap = last.get("config") or {}
        prof = last.get("profit_profile")
        ok_logger = prof == "balanced" and snap.get("profit_profile") == "balanced"
        print(f"    last_sim_profile={prof!r} config.max_daily_llm_cost={snap.get('risk', {}).get('max_daily_llm_cost')}")
    if not ok_logger:
        print("    WARN: expected balanced sim cycle log with config snapshot")

    # 4) Optimizer (quick grid) — run inline if JSON missing/stale
    print(f"\n[4] Optimizer grid (--quick) {START} .. {END}")
    if not OPT_JSON.is_file():
        import scripts.optimize_profiles as opt  # noqa: PLC0415

        class Args:
            days = 5
            end = END
            output = str(OPT_JSON)
            initial_cash = 100_000.0
            cycles_per_day = 1
            quick = True
            baseline = "balanced"
            top = 5
            parallel = None
            genetic = False
            generations = 3
            population = 6
            elite = 1
            mutation_rate = 0.25
            seed = None
            save_profile = None
            emit_snippet = None

        from core.central_profit_config import clear_shared_replay_data

        clear_shared_replay_data()
        opt.run_grid_optimization(Args())  # type: ignore[arg-type]

    payload = json.loads(OPT_JSON.read_text(encoding="utf-8"))
    best = payload["best"]
    rankings = payload.get("rankings") or []
    baseline_row = next((r for r in rankings if r["candidate_id"] == "balanced"), None)
    best_score = float(best["metrics"]["composite_score"])
    baseline_score = float(baseline_row["metrics"]["composite_score"]) if baseline_row else 0.0
    print(f"    best={best['candidate_id']!r} composite={best_score:.4f}")
    print(f"    balanced composite={baseline_score:.4f}")
    print(f"    recommended: {payload.get('recommended_config_changes', {}).get('profit_profile', 'n/a')}")

    score_ok = best_score >= baseline_score
    trades_ok = result.trade_count > 0
    print(f"\n[5] Score check: best >= balanced ? {score_ok}  (best={best_score:.4f} balanced={baseline_score:.4f})")
    print(f"    Sim trades (balanced)={result.trade_count}  recommended_profile={best['candidate_id']!r}")

    print("\n" + "=" * 60)
    if ok_logger and result.cycles_run > 0 and score_ok and trades_ok:
        print("PASS: central ProfitConfig pipeline OK")
        return 0
    print("FAIL: see warnings above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
