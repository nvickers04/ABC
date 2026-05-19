"""
Tiered scoring orchestrator — wide cheap scan → narrow expensive scan → trade recs.

Replaces the old slot-based research loop with a three-tier signal pipeline:
  Tier 1: 150 symbols × 40 cheap signals (~3s)
  Tier 2: 20 symbols × 10 volatility signals (~14s)
  Tier 3: 8 symbols → template selection (<1s)
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import time
from functools import partial
from typing import Any

from core.async_utils import safe_sleep
from core.runtime.mda_budget import (
    log_mda_round_status,
    mda_total_sleep_multiplier,
    persist_mda_snapshot_to_db,
    should_skip_subdaily_candle_fetches,
)
from research.config import (
    OPTION_CHAIN_DTE_RANGE,
    OPTION_CHAIN_STRIKE_LIMIT,
    RESEARCH_UNIVERSE,
)
from core import config as _core_config
from signals.base import SIGNAL_REGISTRY
from signals.combiner import combine_signals, compute_composite_scores
from signals.templates import init_default_boundaries, select_template, write_recommendations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Control state for scorer lifecycle (trader in-process thread or research host loop).
_scorer_thread: "threading.Thread | None" = None  # type: ignore[name-defined]
_scorer_stop_event: "threading.Event | None" = None  # type: ignore[name-defined]
_scorer_pause_event: "threading.Event | None" = None  # type: ignore[name-defined]

# When True, ``run_research`` uses the session-aware cadence helper
# (daemon mode) instead of the fixed ROUND_DELAY_SECS.  Set by
# ``run_research(use_cadence=True)`` / ``run_research_threaded``.
_USE_CADENCE: bool = False


def is_scorer_running() -> bool:
    """True if a scorer thread exists and is alive."""
    return bool(_scorer_thread and _scorer_thread.is_alive())


def is_scorer_paused() -> bool:
    """True if the scorer is running but currently paused between rounds."""
    return bool(_scorer_pause_event and _scorer_pause_event.is_set())


def pause_scorer() -> bool:
    """Ask the scorer to stop starting new rounds. Returns prior state."""
    if _scorer_pause_event is None:
        return False
    was = _scorer_pause_event.is_set()
    _scorer_pause_event.set()
    return not was


def resume_scorer() -> bool:
    """Allow the scorer to start new rounds again. Returns prior state."""
    if _scorer_pause_event is None:
        return False
    was = _scorer_pause_event.is_set()
    _scorer_pause_event.clear()
    return was


def stop_scorer() -> bool:
    """Signal the scorer thread to exit after its current round."""
    global _scorer_thread, _scorer_stop_event, _scorer_pause_event
    if _scorer_stop_event is None:
        return False
    _scorer_stop_event.set()
    if _scorer_pause_event is not None:
        _scorer_pause_event.clear()
    return True


def run_research_threaded(*, verbose: bool = False, use_cadence: bool = False) -> None:
    """Start `run_research` on a dedicated daemon thread with its own
    asyncio event loop. Returns immediately.

    When ``use_cadence`` is True the loop sleeps according to the
    session-aware cadence (regular/extended/overnight) instead of the
    fixed ``ROUND_DELAY_SECS``.  Used by ``python -m research``.

    Idempotent: if the scorer is already running this is a no-op.
    """
    import threading

    global _scorer_thread, _scorer_stop_event, _scorer_pause_event

    if is_scorer_running():
        return

    _scorer_stop_event = threading.Event()
    _scorer_pause_event = threading.Event()

    def _thread_target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_research(verbose=verbose, use_cadence=use_cadence))
        except Exception:
            logger.exception("Scorer thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                pass

    _scorer_thread = threading.Thread(target=_thread_target, name="scorer", daemon=True)
    _scorer_thread.start()


async def run_research(*, verbose: bool = False, use_cadence: bool = False) -> None:
    """
    Main research scoring loop — replaces old slot-based research agent.

    Each round:
      1. Tier 1: Bulk data fetch + 40 cheap signals for 150 symbols
      2. Tier 2: Option chain fetch + 10 volatility signals for top 20
      3. Tier 3: Template selection for top 8
      4. Compute forward returns, run combiner, persist results

    When ``use_cadence`` is True (research-host mode) the loop sleeps
    according to ``core.runtime.cadence.cadence_seconds()`` between
    rounds instead of ``ROUND_DELAY_SECS``.
    """
    global _USE_CADENCE, _scorer_stop_event, _scorer_pause_event
    _USE_CADENCE = bool(use_cadence)
    if use_cadence and _scorer_stop_event is None:
        import threading

        _scorer_stop_event = threading.Event()
        _scorer_pause_event = threading.Event()

    from data.data_provider import DataProvider
    from memory import get_db

    try:
        from core.runtime.research_host_runtime import is_research_host_process

        _on_research_host = is_research_host_process()
    except Exception:
        _on_research_host = False
    logger.info(
        "Signal scoring research loop started (cadence=%s research_host=%s)",
        use_cadence,
        _on_research_host,
    )
    dp = DataProvider()
    conn = get_db()
    init_default_boundaries(conn)

    # Import all signal modules to populate SIGNAL_REGISTRY
    _import_all_signals()

    round_num = 0
    round_mda_sleep_mult = 1.0
    while True:
        try:
            from core.runtime.research_host_runtime import shutdown_requested

            if shutdown_requested():
                logger.info("Scoring loop exiting: research host shutdown requested")
                break
        except Exception:
            pass
        # Honour stop/pause controls between rounds.
        if _scorer_stop_event is not None and _scorer_stop_event.is_set():
            logger.info("Scoring loop stopped by request after %d rounds", round_num)
            break
        if _scorer_pause_event is not None and _scorer_pause_event.is_set():
            await asyncio.sleep(5)
            continue
        round_num += 1
        round_mda_sleep_mult = 1.0
        if _on_research_host:
            try:
                from core.runtime.heartbeat import (
                    ResearchHostStatus,
                    publish_research_host_heartbeat,
                )

                publish_research_host_heartbeat(
                    status=ResearchHostStatus.SCORING,
                    round_num=round_num,
                )
            except Exception:
                pass
        try:
            t0 = time.time()
            # Run the round directly on the main loop. The per-loop
            # httpx client cache in MarketDataClient ensures we don't
            # collide with any other event loop. Running inline avoids
            # a Windows ProactorEventLoop quirk where
            # `call_soon_threadsafe` callbacks from a ThreadPoolExecutor
            # future were not waking the main loop promptly when it
            # was concurrently serving long-running xAI / ib_insync
            # coroutines.
            credits_used = await _scoring_round(dp, conn, round_num)
            elapsed = time.time() - t0
            logger.info(
                "Scoring round %d complete: %.1fs, ~%d scorer_units (rough cost model; not MDA dashboard credits)",
                round_num, elapsed, credits_used,
            )
            try:
                from core.runtime.research_host_runtime import notify_scorer_round_complete

                notify_scorer_round_complete(round_num)
            except Exception:
                pass
            try:
                usage = dp.get_mda_usage()
                round_mda_sleep_mult, _mda_burn_note = mda_total_sleep_multiplier(usage)
                persist_mda_snapshot_to_db(usage, round_mda_sleep_mult)
                log_mda_round_status(
                    round_num,
                    usage,
                    round_mda_sleep_mult,
                    should_skip_subdaily_candle_fetches(usage),
                    burn_note=_mda_burn_note,
                )
            except Exception:
                pass

            # Researcher daily activity cap (research host only; trader has separate Grok caps).
            if not _on_research_host:
                try:
                    from core.runtime.heartbeat import write_heartbeat

                    write_heartbeat()
                except Exception:
                    pass

            cap_verdict = None
            try:
                from core.runtime.research_host_runtime import (
                    is_research_host_process,
                    record_round_usage,
                    request_shutdown,
                )

                if is_research_host_process():
                    cap_verdict = record_round_usage(round_delta=75.0)
                    if _on_research_host:
                        from core.runtime.heartbeat import (
                            ResearchHostStatus,
                            publish_research_host_heartbeat,
                        )

                        status = (
                            ResearchHostStatus.CAP_STOPPED
                            if cap_verdict.should_stop
                            else ResearchHostStatus.RUNNING
                        )
                        publish_research_host_heartbeat(
                            status=status,
                            round_num=round_num,
                            usage_pct=cap_verdict.pct,
                        )
                    if cap_verdict.should_stop:
                        request_shutdown("daily_token_cap")
                        logger.info(
                            "Scoring loop stopping after round %d — daily token cap reached",
                            round_num,
                        )
                        break
            except Exception:
                pass  # never let cap tracking crash the scorer
        except asyncio.CancelledError:
            logger.info("Scoring loop cancelled")
            break
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error("Scoring round %d failed: %s", round_num, e, exc_info=True)

        # Sleep until the next round.  When ``use_cadence`` was passed
        # to ``run_research`` we use the session-aware cadence (daemon
        # mode); otherwise fall back to the configured fixed delay.
        try:
            if _USE_CADENCE:
                from core.runtime.cadence import cadence_seconds

                _base = float(cadence_seconds())
                # Reuse multiplier from end of this round (burn state updates once per round).
                _sleep_s = _base * round_mda_sleep_mult
                if round_mda_sleep_mult > 1.01:
                    logger.info(
                        "MDA pacing: sleep %.0fs (base %.0fs x %.2f combined MDA stretch)",
                        _sleep_s,
                        _base,
                        round_mda_sleep_mult,
                    )
                await safe_sleep(_sleep_s)
            else:
                from core.central_profit_config import get_research_settings

                await safe_sleep(get_research_settings().round_delay_secs)
        except asyncio.CancelledError:
            logger.info("Scoring loop cancelled")
            break


# ---------------------------------------------------------------------------
# Single scoring round
# ---------------------------------------------------------------------------

async def _scoring_round(dp, conn, round_num: int) -> int:
    """Execute one full Tier 1 → Tier 2 → Tier 3 scoring round."""
    from core.central_profit_config import get_research_settings

    rs = get_research_settings()
    credits_used = 0

    # ── Universe ────────────────────────────────────────────
    # The research host scores two tiers of symbols:
    #   * focus  — symbols the trader is actively engaged with (active
    #              attention triggers; usually held + watched names).
    #              Scored EVERY round so positions are freshest.
    #   * base   — RESEARCH_UNIVERSE static list.  Scored every Nth
    #              round per core.runtime.cadence.base_universe_every_n_rounds.
    # See cadence.py module docstring for the budget reasoning.
    from research.config import RESEARCH_UNIVERSE
    from core.runtime.cadence import base_universe_every_n_rounds
    from core.runtime.focus_universe import get_focus_symbols, merge_universes

    focus_syms = get_focus_symbols(conn)
    base_every_n = base_universe_every_n_rounds()
    # round_num is 1-indexed; this makes round 1 include base (no cold
    # start with focus-only rounds before the first base sweep).
    include_base = ((round_num - 1) % base_every_n) == 0
    universe = merge_universes(
        RESEARCH_UNIVERSE, focus_syms, include_base=include_base,
    )
    # Safety net: if attention has nothing AND we'd skip base, score base
    # anyway — an empty round wastes a wakeup and still costs the
    # bulk-quotes call.
    if not universe:
        universe = list(RESEARCH_UNIVERSE)
        include_base = True
    logger.info(
        "Round %d starting: universe=%d (focus=%d, base_included=%s, base_every_n=%d), signals=%d",
        round_num, len(universe), len(focus_syms),
        include_base, base_every_n, len(SIGNAL_REGISTRY),
    )

    # ── Bulk data fetch (Tier 1 data) ──────────────────────
    # All MDA calls use async siblings to avoid nest_asyncio deadlock on Py3.13.
    logger.info("Round %d: fetching bulk quotes...", round_num)
    t_qs = time.time()
    quotes = await dp.get_quotes_bulk_async(universe)
    logger.info("Round %d: bulk quotes in %.1fs (%d)", round_num, time.time() - t_qs, len(quotes))
    credits_used += 1

    # Per-symbol candles — run concurrently so blocking serial calls don't
    # freeze the event loop (which would also stall the agent's LLM cycle).
    logger.info("Round %d: fetching candles for %d symbols...", round_num, len(universe))
    t_c = time.time()
    candles_results = await asyncio.gather(
        *[dp.get_candles_async(sym, days_back=60) for sym in universe],
        return_exceptions=True,
    )
    candles_map: dict[str, Any] = {}
    for sym, res in zip(universe, candles_results):
        if isinstance(res, Exception):
            logger.debug("Candles fetch failed for %s: %s", sym, res)
            continue
        if res:
            candles_map[sym] = res
    credits_used += len(universe)
    logger.info("Round %d: candles fetched for %d/%d symbols in %.1fs", round_num, len(candles_map), len(universe), time.time() - t_c)

    # ── Multi-resolution fetch for forward-return measurement ──────
    # Each signal declares its own (return_resolution, return_lookback_days)
    # for IC computation.  Fetch one bundle per (resolution, days_back)
    # combination for the universe; daily already fetched above and is
    # reused.  Sub-daily resolutions add MDA cost but unlock honest
    # microstructure/intraday IC.  Failures are non-fatal — those signals
    # simply skip forward-return computation that round.
    candles_by_res: dict[str, dict[str, Any]] = {"D": candles_map}
    needed_res: dict[str, int] = {}
    for sig in SIGNAL_REGISTRY.values():
        res = getattr(sig, "return_resolution", "D")
        if res == "D":
            continue
        lb = int(getattr(sig, "return_lookback_days", 30))
        needed_res[res] = max(needed_res.get(res, 0), lb)

    _usage_ration = dp.get_mda_usage()
    _skip_subdaily = should_skip_subdaily_candle_fetches(_usage_ration)
    if _skip_subdaily and needed_res:
        logger.warning(
            "Round %d: MDA rationing — skipping %d sub-daily candle bundle(s) "
            "(1m/5m/1h); daily bars only until credits recover",
            round_num,
            len(needed_res),
        )
    if not _skip_subdaily:
        for res, lb_days in needed_res.items():
            t_r = time.time()
            bundle = await asyncio.gather(
                *[dp.get_candles_async(sym, resolution=res, days_back=lb_days) for sym in universe],
                return_exceptions=True,
            )
            cmap: dict[str, Any] = {}
            for sym, raw in zip(universe, bundle):
                if isinstance(raw, Exception):
                    logger.debug("Candles[%s] fetch failed for %s: %s", res, sym, raw)
                    continue
                if raw:
                    cmap[sym] = raw
            candles_by_res[res] = cmap
            credits_used += len(universe)
            logger.info(
                "Round %d: candles[%s] fetched for %d/%d symbols in %.1fs (lb=%dd)",
                round_num, res, len(cmap), len(universe), time.time() - t_r, lb_days,
            )

    # SPY/QQQ for market-level signals
    spy_candles, qqq_candles = await asyncio.gather(
        dp.get_candles_async("SPY", days_back=60),
        dp.get_candles_async("QQQ", days_back=60),
    )

    # Environment snapshot
    # _get_environment makes sync dp.* calls (nest_asyncio-based) — run
    # in a worker thread so the main event loop is not blocked/patched.
    env = await asyncio.to_thread(
        _get_environment, dp, universe, candles_map, round_num,
    )

    # ── Tier 1: Score cheap signals (40 signals, full universe) ─
    # Skip signals that require an IBKR connection when this process
    # has IBKR quotes disabled (e.g. the research host).  The trader
    # process runs them and writes their scores directly.
    _ibkr_on = bool(getattr(_core_config, "IBKR_QUOTES_ENABLED", False))
    tier1_signals = {
        name: sig for name, sig in SIGNAL_REGISTRY.items()
        if sig.tier == 1 and (_ibkr_on or not getattr(sig, "requires_ibkr", False))
    }

    now = time.time()

    # Per-symbol Tier 1 work (9+ sync dp.* fundamentals calls each) must
    # run off-loop; parallelize with a semaphore to avoid credit spikes.
    t1_sem = asyncio.Semaphore(16)

    def _score_tier1(sym: str) -> tuple[str, dict]:
        sym_data = _build_symbol_data(
            sym, dp, quotes, candles_map,
            spy_candles=spy_candles, qqq_candles=qqq_candles,
            env=env, tier=1,
        )
        sym_scores = {}
        for sig_name, sig in tier1_signals.items():
            sym_scores[sig_name] = sig.score(sym, sym_data)
        return sym, sym_scores

    async def _t1_task(sym: str):
        async with t1_sem:
            return await asyncio.to_thread(_score_tier1, sym)

    t1_results = await asyncio.gather(
        *[_t1_task(s) for s in universe], return_exceptions=True,
    )
    tier1_scores: dict[str, dict[str, dict]] = {}
    for item in t1_results:
        if isinstance(item, Exception):
            logger.debug("Tier 1 symbol failed: %s", item)
            continue
        sym, scores = item
        tier1_scores[sym] = scores
    logger.info("Round %d: Tier 1 scored %d/%d symbols", round_num, len(tier1_scores), len(universe))

    # Persist Tier 1 scores
    _persist_scores(conn, tier1_scores, now)

    # ── Partial composite (Tier 1 weights only) ────────────
    combo = combine_signals(conn, signal_names=list(tier1_signals.keys()))
    weights = combo["weights"]

    partial_composites: dict[str, float] = {}
    for sym in universe:
        composite = 0.0
        for sig_name, w in weights.items():
            s = tier1_scores.get(sym, {}).get(sig_name, {}).get("score", 0.0)
            composite += w * s
        partial_composites[sym] = max(-1.0, min(1.0, composite))

    # ── Tier 2: Deep scan top N by |partial_composite| ─────
    ranked = sorted(partial_composites.items(), key=lambda x: abs(x[1]), reverse=True)
    tier2_symbols = [sym for sym, _ in ranked[: rs.deep_scan_top_n]]

    tier2_signals = {
        name: sig for name, sig in SIGNAL_REGISTRY.items()
        if sig.tier == 2 and (_ibkr_on or not getattr(sig, "requires_ibkr", False))
    }
    # Diagnostic: which symbols won the deep-scan slot this round?
    # Tracked over time, this surfaces whether tier-2 rotates across
    # the full universe or keeps re-selecting the same names (which
    # would starve the others of options-derived signals).
    logger.info(
        "Round %d: Tier 2 selected %s",
        round_num, ",".join(tier2_symbols),
    )

    def _score_tier2(sym: str) -> tuple[str, dict]:
        try:
            iv_info = dp.get_iv_info(sym, dte_min=OPTION_CHAIN_DTE_RANGE[0], dte_max=OPTION_CHAIN_DTE_RANGE[1])
        except Exception:
            iv_info = None
        try:
            # CRITICAL: pass DTE range + strike limit.  Without these MDA returns
            # the ENTIRE chain (all expirations × all strikes × C+P) which costs
            # 1 credit per contract — easily 2,000-5,000 credits per symbol per
            # round.  Constants from research.config keep this bounded.
            chain = dp.get_option_chain(
                sym,
                dte_range=OPTION_CHAIN_DTE_RANGE,
                strike_limit=OPTION_CHAIN_STRIKE_LIMIT,
            )
        except Exception:
            chain = None
        sym_data = _build_symbol_data(
            sym, dp, quotes, candles_map,
            spy_candles=spy_candles, qqq_candles=qqq_candles,
            env=env, tier=2,
            iv_info=iv_info, option_chain=chain,
        )
        sym_scores = {}
        for sig_name, sig in tier2_signals.items():
            sym_scores[sig_name] = sig.score(sym, sym_data)
        return sym, sym_scores

    t2_sem = asyncio.Semaphore(4)

    async def _t2_task(sym: str):
        async with t2_sem:
            return await asyncio.to_thread(_score_tier2, sym)

    t2_results = await asyncio.gather(
        *[_t2_task(s) for s in tier2_symbols], return_exceptions=True,
    )
    tier2_scores: dict[str, dict[str, dict]] = {}
    for item in t2_results:
        if isinstance(item, Exception):
            logger.debug("Tier 2 symbol failed: %s", item)
            continue
        sym, scores = item
        tier2_scores[sym] = scores
    credits_used += 2 * len(tier2_scores)
    logger.info("Round %d: Tier 2 scored %d/%d symbols", round_num, len(tier2_scores), len(tier2_symbols))

    # Persist Tier 2 scores
    _persist_scores(conn, tier2_scores, now)

    # ── Full composite (all 50 signals for Tier 2 symbols) ─
    full_combo = combine_signals(conn)
    full_weights = full_combo["weights"]
    n_eff = full_combo["n_eff"]

    full_composites = compute_composite_scores(conn, full_weights, tier2_symbols)

    # ── Attention evaluator ────────────────────────────────
    # Sync any new watching_for entries into structured triggers, then
    # check every active trigger against this round's quotes &
    # composites.  May wake the trader loop (in-process dev only).  Fail-soft.
    try:
        from core.runtime import attention as _attention
        _attention.sync_from_working_memory(conn)
        fired = _attention.evaluate(conn, quotes=quotes, composites=full_composites)
        if fired:
            logger.info("Round %d: %d attention trigger(s) fired", round_num, len(fired))
    except Exception as e:
        logger.debug("attention layer skipped: %s", e)

    # ── Tier 3: Template selection for top N ───────────────
    top_symbols = sorted(
        full_composites.items(), key=lambda x: abs(x[1]), reverse=True,
    )[: rs.trade_rec_top_n]

    def _build_rec(sym: str, comp_score: float):
        quote = quotes.get(sym)
        iv_info = None
        try:
            iv_info = dp.get_iv_info(sym, dte_min=7, dte_max=60)
        except Exception:
            pass
        iv_rank = getattr(iv_info, "iv_rank", None) if iv_info else None
        atr_pct = None
        try:
            atr_pct = dp.get_atr_percent(sym)
        except Exception:
            pass
        vol_regime = env.get("volatility_regime", "normal") if env else "normal"
        return select_template(
            conn, sym, comp_score,
            iv_rank=iv_rank, atr_pct=atr_pct,
            vol_regime=vol_regime, quote=quote,
        )

    rec_results = await asyncio.gather(
        *[asyncio.to_thread(_build_rec, sym, cs) for sym, cs in top_symbols],
        return_exceptions=True,
    )
    recommendations = []
    for item in rec_results:
        if isinstance(item, Exception):
            logger.debug("Template selection failed: %s", item)
            continue
        if item:
            recommendations.append(item)

    write_recommendations(conn, recommendations)

    # ── Forward returns for prior-round scores ─────────────
    await asyncio.to_thread(
        partial(
            _compute_forward_returns,
            universe_symbols=universe,
            skip_subdaily=_skip_subdaily,
        ),
        conn,
        dp,
        candles_by_res,
        now,
    )

    logger.info(
        "Round %d: N_eff=%.1f, %d composites, %d recs, %d scorer_units",
        round_num, n_eff, len(full_composites), len(recommendations), credits_used,
    )

    return credits_used


# ---------------------------------------------------------------------------
# Data assembly helpers
# ---------------------------------------------------------------------------

def _import_all_signals():
    """Import all signal modules to populate SIGNAL_REGISTRY."""
    import importlib
    import pkgutil
    import signals
    for importer, modname, ispkg in pkgutil.iter_modules(signals.__path__):
        if modname.startswith("_") or modname in ("base", "combiner", "templates", "template_evolution", "briefing", "scorer"):
            continue
        try:
            importlib.import_module(f"signals.{modname}")
        except Exception as e:
            logger.warning("Failed to import signal module %s: %s", modname, e)


def _build_symbol_data(
    symbol: str,
    dp,
    quotes: dict,
    candles_map: dict,
    *,
    spy_candles=None,
    qqq_candles=None,
    env=None,
    tier: int = 1,
    iv_info=None,
    option_chain=None,
) -> dict:
    """Build the data dict that signals expect."""
    data: dict[str, Any] = {
        "data_provider": dp,
        "quote": quotes.get(symbol),
        "candles": candles_map.get(symbol),
        "spy_candles": spy_candles,
        "qqq_candles": qqq_candles,
        "environment": env,
    }

    # Fundamentals (yfinance, cached ~1h — no credits)
    try:
        basic_fund = dp.get_fundamentals(symbol)
        data["fundamentals"] = basic_fund
        data["basic_fundamentals"] = basic_fund  # Preserve for size_factor (has market_cap)
    except Exception:
        pass
    # Extended fundamentals has all the fields signals need (FCF, ROE, etc.)
    # Use it as the primary fundamentals source when available.
    try:
        ext = dp.get_extended_fundamentals(symbol)
        if ext:
            data["fundamentals"] = ext
    except Exception:
        pass
    try:
        data["earnings"] = dp.get_earnings_info(symbol)
    except Exception:
        pass
    try:
        data["earnings_history"] = dp.get_earnings_history(symbol)
    except Exception:
        pass
    try:
        data["analyst"] = dp.get_analyst_data(symbol)
    except Exception:
        pass
    try:
        data["institutional"] = dp.get_institutional_data(symbol)
    except Exception:
        pass
    try:
        data["insider"] = dp.get_insider_data(symbol)
    except Exception:
        pass
    try:
        data["peer_comparison"] = dp.get_peer_comparison(symbol)
    except Exception:
        pass
    try:
        data["news"] = dp.get_news(symbol)
    except Exception:
        pass

    if tier >= 2:
        data["iv_info"] = iv_info
        data["option_chain"] = option_chain

    if tier >= 2:
        data["iv_info"] = iv_info
        data["option_chain"] = option_chain

    # Aliases for signals that use alternate key names
    data["candles_daily"] = data.get("candles")         # gap, straddle_cost, realized_vol_cone, iv_rv_spread
    data["earnings_info"] = data.get("earnings")            # earnings (earnings_momentum)

    return data


def _get_environment(
    dp,
    universe: list[str],
    candles_map: dict,
    round_num: int | None = None,
) -> dict | None:
    """Compute environment snapshot for this round."""
    try:
        import datetime
        import pandas as pd
        from research.environment import compute_session_environment
        session_date = datetime.date.today().isoformat()

        # compute_session_environment expects {symbol: pd.DataFrame}
        df_map = {}
        for sym, candles in candles_map.items():
            if candles and len(candles) > 0:
                df_map[sym] = pd.DataFrame({
                    "open": candles.open,
                    "high": candles.high,
                    "low": candles.low,
                    "close": candles.close,
                    "volume": candles.volume[:len(candles.close)],
                })

        if not df_map:
            return None

        env = compute_session_environment(df_map, session_date)

        # ── Enrich with cross-asset correlation ─────────────────
        import numpy as np
        daily_returns = []
        for sym, candles in candles_map.items():
            if candles and len(candles) >= 5:
                closes = np.array(candles.close, dtype=float)
                rets = np.diff(closes) / closes[:-1]
                daily_returns.append(rets)
        if len(daily_returns) >= 3:
            min_len = min(len(r) for r in daily_returns)
            if min_len >= 3:
                mat = np.array([r[:min_len] for r in daily_returns])
                corr_matrix = np.corrcoef(mat)
                n_sym = corr_matrix.shape[0]
                off_diag = []
                for i in range(n_sym):
                    for j in range(i + 1, n_sym):
                        v = corr_matrix[i, j]
                        if np.isfinite(v):
                            off_diag.append(v)
                if off_diag:
                    env["cross_asset_correlation"] = float(np.mean(off_diag))

        # ── Macro event proximity ───────────────────────────────
        try:
            from data.economic_calendar import get_upcoming_events
            today = datetime.date.today()
            upcoming = get_upcoming_events(days=30, today=today)
            for event in upcoming:
                days_out = (event.date - today).days
                name_lower = event.name.lower()
                if "fomc" in name_lower and env.get("days_to_fomc") is None:
                    env["days_to_fomc"] = days_out
                elif ("nonfarm" in name_lower or "nfp" in name_lower) and env.get("days_to_nfp") is None:
                    env["days_to_nfp"] = days_out
                elif "cpi" in name_lower and env.get("days_to_cpi") is None:
                    env["days_to_cpi"] = days_out
        except Exception:
            pass

        # ── Regime duration tracking ────────────────────────────
        try:
            from memory import get_db
            conn = get_db()
            vol_regime = env.get("volatility_regime")
            trend_regime = env.get("trend_regime")

            # Count consecutive prior snapshots with same regime
            prev_rows = conn.execute(
                "SELECT volatility_regime, trend_regime FROM environment_snapshots ORDER BY id DESC LIMIT 20"
            ).fetchall()
            vol_dur = 1
            trend_dur = 1
            for row in prev_rows:
                if row["volatility_regime"] == vol_regime:
                    vol_dur += 1
                else:
                    break
            for row in prev_rows:
                if row["trend_regime"] == trend_regime:
                    trend_dur += 1
                else:
                    break
            env["vol_regime_duration"] = vol_dur
            env["trend_regime_duration"] = trend_dur

            # Persist snapshot using existing schema
            import datetime as _dt
            conn.execute(
                """INSERT INTO environment_snapshots
                   (ts, round_num, volatility_regime, trend_regime, breadth_regime,
                    momentum_regime, volume_regime, avg_atr_pct, dispersion,
                    raw_snapshot_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    round_num,
                    env.get("volatility_regime"),
                    env.get("trend_regime"),
                    env.get("breadth_regime"),
                    env.get("momentum_regime"),
                    env.get("volume_regime"),
                    env.get("avg_atr_pct"),
                    env.get("dispersion"),
                    json.dumps(env),
                ),
            )
            conn.commit()
        except Exception:
            env.setdefault("vol_regime_duration", 1)
            env.setdefault("trend_regime_duration", 1)

        return env
    except Exception as e:
        logger.debug("Environment computation failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _persist_scores(conn, scores: dict[str, dict[str, dict]], ts: float) -> None:
    """Persist signal scores to DB."""
    rows = []
    for sym, sym_scores in scores.items():
        for sig_name, result in sym_scores.items():
            rows.append((
                sig_name, sym, ts,
                result.get("score", 0.0),
                result.get("confidence", 0.0),
                json.dumps(result.get("components", {})),
            ))
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO signal_scores "
            "(signal_name, symbol, ts, score, confidence, components_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def _compute_forward_returns(
    conn,
    dp,
    candles_by_res: dict[str, dict],
    current_ts: float,
    *,
    universe_symbols: list[str] | None = None,
    skip_subdaily: bool = False,
) -> None:
    """
    Compute forward returns for prior-round scores and write R(i,s) to signal_returns.

    R(i,s) = score(i, sym, s) × forward_return(sym, entry_bar → entry_bar + h)

    Each signal declares its own (return_resolution, return_horizon) on the
    Signal subclass.  Bars are looked up in ``candles_by_res[res][sym]``.
    Rows are keyed by ``(signal, symbol, entry_bar_ts)`` — multiple intraday
    score rounds that resolve to the same bar collapse via INSERT OR REPLACE,
    keeping the most recent score for that bar.  This is the correct dedup
    because each signal has its OWN bar grid (a 1-min signal has 390 bars/day,
    a daily signal has 1).

    Backlog drains with a per-resolution budget (shortest-resolution first
    so the freshest microstructure data lands in IC fastest).  Scores too
    old to ever mature (no candles for the symbol at the signal's resolution)
    are deleted at the top of the function.

    ``universe_symbols`` and ``skip_subdaily`` are supplied by the scoring
    round: the former keeps immature rows for tickers scored this cycle
    (including focus names outside ``RESEARCH_UNIVERSE``); the latter skips
    sub-daily resolution scans when MDA rationing omitted those bundles.
    """
    # ── Prune scores that can never mature ─────────────────────────
    # Only delete rows for symbols that are not in the canonical research
    # universe, not in this round's scored universe, and not present in any
    # candle map we hold.  Using ``live_symbols`` alone was wrong: on
    # focus-only rounds we still skip fetching the full base list, but those
    # symbols' immature scores must survive until the next base-inclusive
    # round — otherwise ``signal_returns`` / IC starve.
    live_symbols: set[str] = set()
    for cmap in candles_by_res.values():
        for sym in cmap.keys():
            if isinstance(sym, str) and sym.strip():
                live_symbols.add(sym.strip().upper())

    protected: set[str] = set()
    for sym in RESEARCH_UNIVERSE:
        if isinstance(sym, str) and sym.strip():
            protected.add(sym.strip().upper())
    if universe_symbols:
        for sym in universe_symbols:
            if isinstance(sym, str) and sym.strip():
                protected.add(sym.strip().upper())
    protected.update(live_symbols)

    if protected:
        placeholders = ",".join("?" * len(protected))
        n = conn.execute(
            f"DELETE FROM signal_scores WHERE UPPER(symbol) NOT IN ({placeholders})",
            tuple(sorted(protected)),
        ).rowcount
        if n > 0:
            conn.commit()
            logger.info(
                "Pruned %d orphan signal_scores (symbol outside research+round+candles)",
                n,
            )

    # TTL purge: anything older than 30 days is past every signal's horizon
    # (longest horizon today = 10 trading days ≈ 14 calendar).  Without
    # this, ancient unmatured scores eat the per-round LIMIT forever.
    ttl_cutoff = current_ts - 30 * 86400
    n_ttl = conn.execute(
        "DELETE FROM signal_scores WHERE ts < ?", (ttl_cutoff,)
    ).rowcount
    if n_ttl > 0:
        conn.commit()
        logger.info("Pruned %d stale signal_scores (older than 30d)", n_ttl)

    # Registry-orphan prune: scores for signals removed from the registry
    # never appear in the per-resolution scan below (the SQL filters
    # signal_name IN (registered_names)), so they would sit in the table
    # until the 30-day TTL fires.  Drop them now.
    live_signals = list(SIGNAL_REGISTRY.keys())
    if live_signals:
        placeholders = ",".join("?" * len(live_signals))
        n_reg = conn.execute(
            f"DELETE FROM signal_scores WHERE signal_name NOT IN ({placeholders})",
            tuple(live_signals),
        ).rowcount
        if n_reg > 0:
            conn.commit()
            logger.info("Pruned %d orphan signal_scores (signal not in registry)", n_reg)
    else:
        # Empty registry — every row is an orphan.
        n_reg = conn.execute("DELETE FROM signal_scores").rowcount
        if n_reg > 0:
            conn.commit()
            logger.info("Pruned %d orphan signal_scores (registry empty)", n_reg)

    # ── Group signals by resolution; drain shortest first ──────────
    sigs_by_res: dict[str, list[str]] = {}
    for name, sig in SIGNAL_REGISTRY.items():
        res = getattr(sig, "return_resolution", "D")
        sigs_by_res.setdefault(res, []).append(name)

    # Per-resolution backlog budget.  Sub-daily resolutions can produce
    # thousands of bars per symbol per day; daily produces 1.  Caps keep
    # any single resolution from monopolising the round's wall time.
    BUDGET_BY_RES = {
        "1min": 2000,
        "5min": 1500,
        "15min": 1000,
        "1h":   800,
        "D":    500,
    }
    # Sort by typical bar size (fastest first) so freshest data lands first.
    res_order = ["1min", "5min", "15min", "1h", "D"]

    totals = {"wrote": 0, "pending": 0, "no_entry": 0, "no_candles": 0, "scanned": 0, "zombies": 0}
    per_res_log: list[str] = []

    for res in res_order:
        sig_names = sigs_by_res.get(res)
        if not sig_names:
            continue
        # When MDA rationing skipped sub-daily fetches, those resolutions are
        # absent from ``candles_by_res`` — do not scan thousands of rows that
        # would all hit ``no_candles`` and starve the daily backlog.
        if res != "D" and skip_subdaily and res not in candles_by_res:
            per_res_log.append(f"{res}:skipped_mda_no_subdaily_bundle")
            continue
        cmap = candles_by_res.get(res, {})
        budget = BUDGET_BY_RES.get(res, 500)

        placeholders = ",".join("?" * len(sig_names))
        cur = conn.execute(
            f"""SELECT ss.signal_name, ss.symbol, ss.ts, ss.score
                  FROM signal_scores ss
                 WHERE ss.ts < ?
                   AND ss.signal_name IN ({placeholders})
                 ORDER BY ss.ts ASC
                 LIMIT ?""",
            (current_ts, *sig_names, budget),
        )
        prior_scores = cur.fetchall()
        totals["scanned"] += len(prior_scores)

        rows = []
        matured_keys: list[tuple[str, str, float]] = []
        # Zombie rows: scores that can never mature in the current state.
        # We delete these alongside matured rows so they stop dominating
        # the LIMIT 2000 ORDER BY ts ASC scan on subsequent rounds.
        zombie_keys: list[tuple[str, str, float]] = []
        wrote = pending = no_entry = no_candles = zombies = 0

        for sig_name, sym, score_ts, score in prior_scores:
            sig = SIGNAL_REGISTRY.get(sig_name)
            if not sig:
                # Signal retired from registry — score can never resolve.
                zombie_keys.append((sig_name, sym, float(score_ts)))
                zombies += 1
                continue
            horizon = int(getattr(sig, "return_horizon", 5))

            candles = cmap.get(sym)
            if not candles:
                no_candles += 1
                continue
            closes = candles.close
            ts_list = candles.timestamps
            if len(closes) < 2 or not ts_list or len(ts_list) != len(closes):
                no_candles += 1
                continue

            # Most recent bar with ts <= score_ts.
            i_entry = bisect.bisect_right(ts_list, score_ts) - 1
            if i_entry < 0:
                # ``score_ts`` predates every candle in the current window.
                # The candle window slides forward over time, so this score
                # can never resolve — purge it.
                zombie_keys.append((sig_name, sym, float(score_ts)))
                no_entry += 1
                zombies += 1
                continue
            i_exit = i_entry + horizon
            if i_exit >= len(ts_list):
                pending += 1
                continue

            entry_price = closes[i_entry]
            exit_price = closes[i_exit]
            if not (entry_price and entry_price > 0):
                # Bad/zero entry price for the matched bar — unrecoverable.
                zombie_keys.append((sig_name, sym, float(score_ts)))
                zombies += 1
                continue

            fwd_return = (exit_price - entry_price) / entry_price
            r_value = float(score) * fwd_return
            entry_bar_ts = float(ts_list[i_entry])
            rows.append((
                sig_name, sym, entry_bar_ts,
                float(score), fwd_return, r_value, horizon,
            ))
            matured_keys.append((sig_name, sym, float(score_ts)))
            wrote += 1

        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO signal_returns "
                "(signal_name, symbol, ts, score_at_entry, forward_return, r_value, horizon_bars) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        # Delete matured + zombie rows from signal_scores so they don't
        # dominate the ORDER BY ts ASC LIMIT queue on subsequent rounds.
        # Without this, ancient matured/unmaturable scores would starve
        # fresh ones until the 30-day TTL purge fires.  The R(i,s) row in
        # signal_returns is the durable record; signal_scores is a
        # maturation queue.
        cleanup_keys = matured_keys + zombie_keys
        if cleanup_keys:
            conn.executemany(
                "DELETE FROM signal_scores "
                "WHERE signal_name = ? AND symbol = ? AND ts = ?",
                cleanup_keys,
            )
            conn.commit()

        totals["wrote"] += wrote
        totals["pending"] += pending
        totals["no_entry"] += no_entry
        totals["no_candles"] += no_candles
        totals["zombies"] += zombies
        per_res_log.append(
            f"{res}:wrote={wrote},pending={pending},no_candles={no_candles},zombies={zombies}"
        )

    logger.info(
        "Forward returns: wrote=%d pending=%d no_entry=%d no_candles=%d "
        "zombies=%d (scanned %d) [%s]",
        totals["wrote"], totals["pending"], totals["no_entry"],
        totals["no_candles"], totals["zombies"], totals["scanned"],
        " ".join(per_res_log),
    )
