"""Run TradingAgent.run_cycle() over a historical date range."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import exchange_calendars as ecals

from core.agent import TradingAgent
from core.central_profit_config import get_profit_config
from core.profile_optimization import DEFAULT_CYCLES_PER_DAY
from core.profit_profiles import PROFIT_PROFILE_ENV
from core.profit_config_state import (
    InvalidProfitProfileError,
    SimulationDataError,
    log_active_profit_config,
    resolve_profile_label,
    safe_normalize_profit_profile,
    set_active_profile_label,
    validate_backtest_date_range,
)
from core.simulation.backtest_llm import BacktestLLM, CyclePlanContext
from core.central_profit_config import (
    ReplayDataProvider as SharedReplayDataProvider,
    get_shared_replay_data,
)
from core.simulation.replay_data import ReplaySessionDataProvider
from core.simulation.sim_broker import SimulatedBroker
from core.simulation.sim_clock import et_to_utc, frozen_utc
from core.simulation.sim_market_hours import SimulatedMarketHoursProvider
from core.simulation.llm_cost_estimate import (
    BacktestRunStats,
    check_backtest_budget,
    estimate_backtest_llm_upper_bound,
    format_run_summary,
)
from core.simulation.stats import build_backtest_result
from core.simulation.types import BacktestResult
from data.cost_tracker import get_cost_tracker
from tools.tools_executor import ToolExecutor

logger = logging.getLogger(__name__)

# ET times for intraday cycles (regular session)
_CYCLE_ET_SLOTS = ((9, 35), (11, 0), (13, 0), (15, 30))


def _trading_days(start_date: str, end_date: str) -> list[str]:
    cal = ecals.get_calendar("XNYS")
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    days: list[str] = []
    d = start
    while d <= end:
        if cal.is_session(d):
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def _top_composite_for_date(session_date: str) -> tuple[str | None, float]:
    try:
        from memory import get_db

        db = get_db()
        row = db.execute(
            """
            SELECT symbol, composite_score
            FROM composite_scores
            WHERE date(ts) = date(?)
            ORDER BY composite_score DESC
            LIMIT 1
            """,
            (session_date,),
        ).fetchone()
        if row:
            return str(row[0]), float(row[1] or 0)
    except Exception as e:
        logger.debug("composite_lookup_skipped: %s", e)
    if os.getenv("ABC_SIMULATION") == "1":
        # Deterministic driver so BacktestLLM can exercise trade tools without Postgres scorer.
        try:
            score = float(os.getenv("ABC_SIM_DEFAULT_COMPOSITE_SCORE", "0.72"))
        except ValueError:
            score = 0.72
        return "SPY", score
    return None, 0.0


async def run_backtest_async(
    profile_name: str,
    start_date: str,
    end_date: str,
    *,
    initial_cash: float = 100_000.0,
    cycles_per_day: int = DEFAULT_CYCLES_PER_DAY,
    config_patches: dict[str, dict[str, object]] | None = None,
    candidate_id: str | None = None,
    reload_dotenv: bool = True,
    replay_data: SharedReplayDataProvider | None = None,
    composed: Any | None = None,
) -> BacktestResult:
    """Load profile, replay data, run agent cycles with gates enabled."""
    from data.data_provider import install_data_provider
    from data.market_hours import install_market_hours_provider

    profit_cfg = None
    label = candidate_id or profile_name
    providers_installed = False
    run_started = time.perf_counter()
    run_stats = BacktestRunStats()
    pushed_composed = False

    try:
        start_date, end_date = validate_backtest_date_range(start_date, end_date)
        profile = safe_normalize_profit_profile(profile_name)
        label = resolve_profile_label(candidate_id=candidate_id, base_profile=str(profile))
        set_active_profile_label(label)

        os.environ[PROFIT_PROFILE_ENV] = str(profile)
        os.environ["ABC_SIMULATION"] = "1"
        os.environ.setdefault("TRADING_MODE", "paper")
        os.environ.setdefault("IBKR_ACCOUNT_TYPE", "paper")

        if composed is not None:
            from core.profit_config_context import push_composed_config

            push_composed_config(composed)
            pushed_composed = True
            profit_cfg = composed
        else:
            profit_cfg = get_profit_config().reload(dotenv=reload_dotenv)
            if config_patches:
                from core.profile_optimization import apply_config_patches

                profit_cfg = apply_config_patches(
                    profit_cfg,
                    config_patches,
                    profile_label=label,
                )

        notes = [
            f"Candidate: {label} (base profile {profile}, {profit_cfg.risk.max_daily_llm_cost:.2f} USD/day LLM cap).",
            "ReAct loop: core.agent.TradingAgent.run_cycle (BacktestLLM, no live xAI).",
            "QualityMatrix + SafetyController: active.",
        ]

        days = _trading_days(start_date, end_date)
        slots_count = max(1, min(cycles_per_day, len(_CYCLE_ET_SLOTS)))
        logger.info(
            "backtest cycles_per_day=%d (ET slots=%d) profile=%s window=%s..%s sessions=%d",
            cycles_per_day,
            slots_count,
            label,
            start_date,
            end_date,
            len(days),
        )
        if not days:
            return build_backtest_result(
                profile=label,
                start_date=start_date,
                end_date=end_date,
                trading_days=0,
                cycles_run=0,
                cycles_per_day=cycles_per_day,
                initial_equity=initial_cash,
                equity_curve=[],
                closed_trades=[],
                trade_log=[],
                llm_cost_usd=0.0,
                notes=notes + ["No trading sessions in range."],
            )

        upper = estimate_backtest_llm_upper_bound(len(days), cycles_per_day)
        check_backtest_budget(
            upper,
            trading_days=len(days),
            cycles_per_day=cycles_per_day,
            profile_max_daily_llm=profit_cfg.risk.max_daily_llm_cost,
        )

        shared = replay_data or get_shared_replay_data(start_date, end_date)
        was_loaded = shared.is_loaded
        arch_t0 = time.perf_counter()
        if not was_loaded:
            try:
                await shared.load()
            except Exception as exc:
                log_active_profit_config(
                    logger,
                    "archive fetch failed",
                    cfg=profit_cfg,
                    exc=exc,
                    start_date=start_date,
                    end_date=end_date,
                )
                raise SimulationDataError(
                    f"Failed to load historical replay data: {exc}"
                ) from exc
            run_stats.archive_elapsed_sec = float(
                shared.load_stats.get("elapsed_sec", time.perf_counter() - arch_t0)
            )
        else:
            run_stats.archive_elapsed_sec = 0.0

        replay: ReplaySessionDataProvider = shared.spawn_session()
        missing_symbols = list(shared.missing_symbols)

        if missing_symbols:
            warn = (
                f"Missing or empty bar data for: {', '.join(missing_symbols)} "
                "(replay may use stale/flat prices)."
            )
            logger.warning(warn)
            notes.append(warn)

        broker = SimulatedBroker(
            initial_cash=initial_cash,
            replay=replay,
            profit_profile=profile,
        )
        market_hours = SimulatedMarketHoursProvider()
        llm = BacktestLLM()

        install_data_provider(replay)
        install_market_hours_provider(market_hours)
        providers_installed = True

        cost_tracker = get_cost_tracker()
        llm_cost_start = float(getattr(cost_tracker, "get_today_cost", lambda: 0.0)() or 0.0)

        equity_curve: list[tuple[str, float]] = []
        cycles_run = 0
        slots = _CYCLE_ET_SLOTS[: max(1, min(cycles_per_day, len(_CYCLE_ET_SLOTS)))]
        cycles_t0 = time.perf_counter()

        for session_date in days:
            broker.set_session_date(session_date)
            traded_today = False
            top_sym, top_score = _top_composite_for_date(session_date)

            for cycle_idx, (hh, mm) in enumerate(slots):
                now_utc = et_to_utc(session_date, hh, mm)
                replay.set_sim_context(session_date, now_utc)
                market_hours.set_now(now_utc)
                broker.set_sim_time(now_utc)

                ctx = CyclePlanContext(
                    session_date=session_date,
                    cycle_index=cycle_idx,
                    top_symbol=top_sym,
                    top_score=top_score,
                    traded_today=traded_today,
                )
                llm.prepare_cycle(ctx)

                with frozen_utc(now_utc):
                    tools = ToolExecutor(
                        broker,
                        replay,
                        market_hours_provider=market_hours,
                        cost_tracker=cost_tracker,
                    )
                    agent = TradingAgent(broker, tools, grok=llm)
                    tools._state_builder = agent._build_state_context
                    tools._agent = agent
                    if cycle_idx == 0:
                        agent._capture_start_of_day_cash()

                    try:
                        await agent.run_cycle()
                        cycles_run += 1
                    except Exception as exc:
                        log_active_profit_config(
                            logger,
                            "sim_cycle_failed",
                            cfg=profit_cfg,
                            level=logging.WARNING,
                            exc=exc,
                            date=session_date,
                            cycle_index=cycle_idx,
                        )
                        notes.append(f"Cycle failed {session_date} #{cycle_idx}: {exc}")

                if broker._positions:
                    traded_today = True

            await broker.flatten_all()
            broker._update_nlv()
            equity_curve.append((session_date, broker.net_liquidation))

        run_stats.cycles_elapsed_sec = time.perf_counter() - cycles_t0
        run_stats.llm_usage = llm.usage
        run_stats.elapsed_sec = time.perf_counter() - run_started

        llm_cost = llm.estimated_cost_usd
        try:
            llm_cost = max(llm_cost, float(cost_tracker.get_today_cost()) - llm_cost_start)
        except Exception:
            pass

        check_backtest_budget(
            run_stats.llm_usage,
            trading_days=len(days),
            cycles_per_day=cycles_per_day,
            profile_max_daily_llm=profit_cfg.risk.max_daily_llm_cost,
        )

        result = build_backtest_result(
            profile=label,
            start_date=start_date,
            end_date=end_date,
            trading_days=len(days),
            cycles_run=cycles_run,
            cycles_per_day=cycles_per_day,
            initial_equity=initial_cash,
            equity_curve=equity_curve,
            closed_trades=broker.closed_trades,
            trade_log=list(broker.trade_log),
            llm_cost_usd=llm_cost,
            notes=notes,
            run_stats=run_stats,
        )
        logger.info(
            format_run_summary(
                label=label,
                stats=run_stats,
                trading_days=len(days),
                cycles_run=cycles_run,
            )
        )
        return result
    except (InvalidProfitProfileError, SimulationDataError, ValueError):
        raise
    except Exception as exc:
        log_active_profit_config(
            logger,
            "run_backtest_async failed",
            cfg=profit_cfg,
            exc=exc,
            profile_name=profile_name,
            candidate_id=candidate_id,
            start_date=start_date,
            end_date=end_date,
        )
        raise
    finally:
        if pushed_composed:
            from core.profit_config_context import pop_composed_config

            pop_composed_config()
        if providers_installed:
            install_data_provider(None)
            install_market_hours_provider(None)
