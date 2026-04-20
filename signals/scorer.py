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
from typing import Any

from core.async_utils import safe_sleep
from research.config import (
    DEEP_SCAN_TOP_N,
    FORWARD_RETURN_HORIZON,
    MAX_CREDITS_PER_ROUND,
    OPTION_CHAIN_DTE_RANGE,
    OPTION_CHAIN_STRIKE_LIMIT,
    ROUND_DELAY_SECS,
    TIER1_UNIVERSE_SIZE,
    TRADE_REC_TOP_N,
)
from signals.base import SIGNAL_REGISTRY
from signals.combiner import combine_signals, compute_composite_scores
from signals.templates import init_default_boundaries, select_template, write_recommendations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Control state for agent-controlled scorer lifecycle.
_scorer_thread: "threading.Thread | None" = None  # type: ignore[name-defined]
_scorer_stop_event: "threading.Event | None" = None  # type: ignore[name-defined]
_scorer_pause_event: "threading.Event | None" = None  # type: ignore[name-defined]


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


def run_research_threaded(*, verbose: bool = False) -> None:
    """Start `run_research` on a dedicated daemon thread with its own
    asyncio event loop. Returns immediately.

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
            loop.run_until_complete(run_research(verbose=verbose))
        except Exception:
            logger.exception("Scorer thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                pass

    _scorer_thread = threading.Thread(target=_thread_target, name="scorer", daemon=True)
    _scorer_thread.start()


async def run_research(*, verbose: bool = False) -> None:
    """
    Main research scoring loop — replaces old slot-based research agent.

    Each round:
      1. Tier 1: Bulk data fetch + 40 cheap signals for 150 symbols
      2. Tier 2: Option chain fetch + 10 volatility signals for top 20
      3. Tier 3: Template selection for top 8
      4. Compute forward returns, run combiner, persist results
    """
    from data.data_provider import DataProvider
    from memory import get_db

    logger.info("Signal scoring research loop started")
    dp = DataProvider()
    conn = get_db()
    init_default_boundaries(conn)

    # Import all signal modules to populate SIGNAL_REGISTRY
    _import_all_signals()

    round_num = 0
    while True:
        # Honour stop/pause controls between rounds.
        if _scorer_stop_event is not None and _scorer_stop_event.is_set():
            logger.info("Scoring loop stopped by request after %d rounds", round_num)
            break
        if _scorer_pause_event is not None and _scorer_pause_event.is_set():
            await asyncio.sleep(5)
            continue
        round_num += 1
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
                "Scoring round %d complete: %.1fs, ~%d credits",
                round_num, elapsed, credits_used,
            )
        except asyncio.CancelledError:
            logger.info("Scoring loop cancelled")
            break
        except Exception as e:
            logger.error("Scoring round %d failed: %s", round_num, e, exc_info=True)

        try:
            await safe_sleep(ROUND_DELAY_SECS)
        except asyncio.CancelledError:
            logger.info("Scoring loop cancelled")
            break


# ---------------------------------------------------------------------------
# Single scoring round
# ---------------------------------------------------------------------------

async def _scoring_round(dp, conn, round_num: int) -> int:
    """Execute one full Tier 1 → Tier 2 → Tier 3 scoring round."""
    credits_used = 0

    # ── Universe ────────────────────────────────────────────
    from research.config import RESEARCH_UNIVERSE
    universe = list(RESEARCH_UNIVERSE)
    logger.info(
        "Round %d starting: universe=%d, signals=%d",
        round_num, len(universe), len(SIGNAL_REGISTRY),
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

    # SPY/QQQ for market-level signals
    spy_candles, qqq_candles = await asyncio.gather(
        dp.get_candles_async("SPY", days_back=60),
        dp.get_candles_async("QQQ", days_back=60),
    )

    # Environment snapshot
    # _get_environment makes sync dp.* calls (nest_asyncio-based) — run
    # in a worker thread so the main event loop is not blocked/patched.
    env = await asyncio.to_thread(_get_environment, dp, universe, candles_map)

    # ── Tier 1: Score cheap signals (40 signals, full universe) ─
    tier1_signals = {
        name: sig for name, sig in SIGNAL_REGISTRY.items()
        if sig.tier == 1
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
    tier2_symbols = [sym for sym, _ in ranked[:DEEP_SCAN_TOP_N]]

    tier2_signals = {
        name: sig for name, sig in SIGNAL_REGISTRY.items()
        if sig.tier == 2
    }

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

    # ── Tier 3: Template selection for top N ───────────────
    top_symbols = sorted(
        full_composites.items(), key=lambda x: abs(x[1]), reverse=True,
    )[:TRADE_REC_TOP_N]

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
    await asyncio.to_thread(_compute_forward_returns, conn, dp, candles_map, now)

    logger.info(
        "Round %d: N_eff=%.1f, %d composites, %d recs, %d credits",
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


def _get_environment(dp, universe: list[str], candles_map: dict) -> dict | None:
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
                   (ts, volatility_regime, trend_regime, breadth_regime,
                    momentum_regime, volume_regime, avg_atr_pct, dispersion,
                    raw_snapshot_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _dt.datetime.now(_dt.timezone.utc).isoformat(),
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


def _compute_forward_returns(conn, dp, candles_map: dict, current_ts: float) -> None:
    """
    Compute forward returns for prior-round scores and write R(i,s) to signal_returns.

    R(i,s) = score(i,sym,s) × forward_return(sym, s→s+h)

    CRITICAL: the forward return must be measured from the bar AT the score
    timestamp to `horizon` bars later.  Earlier versions used trailing price
    windows relative to the current time, which caused label leakage (recent
    price moves fed back into IC via the same prices used to compute
    momentum/mean-reversion signals → fabricated IC magnitudes of 0.5+).
    Rows whose horizon has not yet elapsed are left unrecorded and retried
    on the next round when fresh candles land.
    """
    # Prune scores for symbols no longer in the current universe.  Without
    # this, a universe change leaves orphan rows that can never be matured
    # (we won't fetch their candles again) — they then dominate the LIMIT
    # 5000 ORDER BY ts ASC budget below and starve the new universe of IC.
    if candles_map:
        live_symbols = tuple(candles_map.keys())
        if live_symbols:
            placeholders = ",".join("?" * len(live_symbols))
            n = conn.execute(
                f"DELETE FROM signal_scores WHERE symbol NOT IN ({placeholders})",
                live_symbols,
            ).rowcount
            if n > 0:
                conn.commit()
                logger.info("Pruned %d orphan signal_scores (symbol no longer in universe)", n)

    # Get prior-round scores that don't yet have forward returns computed.
    # Order by ts ASC so the OLDEST scores (most likely to have their horizon
    # already elapsed) are processed first.  LIMIT is a per-round budget, not
    # a hard ceiling on the total backlog; subsequent rounds drain the rest.
    cur = conn.execute(
        """SELECT ss.signal_name, ss.symbol, ss.ts, ss.score
           FROM signal_scores ss
           WHERE ss.ts < ?
             AND NOT EXISTS (
               SELECT 1 FROM signal_returns sr
               WHERE sr.signal_name = ss.signal_name
                 AND sr.symbol = ss.symbol
                 AND sr.ts = ss.ts
             )
           ORDER BY ss.ts ASC
           LIMIT 5000""",
        (current_ts,),
    )
    prior_scores = cur.fetchall()

    if not prior_scores:
        return

    rows = []
    pending = 0
    no_candles = 0
    no_entry = 0
    for row in prior_scores:
        sig_name, sym, score_ts, score = row
        sig = SIGNAL_REGISTRY.get(sig_name)
        if not sig:
            continue

        horizon = FORWARD_RETURN_HORIZON.get(sig.category, 5)

        candles = candles_map.get(sym)
        if not candles:
            no_candles += 1
            continue

        closes = candles.close
        ts_list = candles.timestamps
        if len(closes) < 2 or not ts_list or len(ts_list) != len(closes):
            no_candles += 1
            continue

        # Locate the bar representing price knowledge AS OF score_ts:
        # the most recent bar with timestamp <= score_ts.
        i_entry = bisect.bisect_right(ts_list, score_ts) - 1
        if i_entry < 0:
            # score_ts precedes all candle history.
            no_entry += 1
            continue

        i_exit = i_entry + horizon
        if i_exit >= len(ts_list):
            # Horizon has not elapsed.  Leave unrecorded; retry next round.
            pending += 1
            continue

        entry_price = closes[i_entry]
        exit_price = closes[i_exit]

        if entry_price and entry_price > 0:
            fwd_return = (exit_price - entry_price) / entry_price
            r_value = float(score) * fwd_return
            rows.append((
                sig_name, sym, score_ts,
                float(score), fwd_return, r_value, horizon,
            ))

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO signal_returns "
            "(signal_name, symbol, ts, score_at_entry, forward_return, r_value, horizon_bars) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    logger.info(
        "Forward returns: wrote=%d pending=%d no_entry=%d no_candles=%d (of %d prior scores)",
        len(rows), pending, no_entry, no_candles, len(prior_scores),
    )
