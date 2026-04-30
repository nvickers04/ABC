"""
Template evolution — continuous walk-forward boundary optimisation.

Runs as third asyncio.gather() task alongside agent and research scorer.
Evolves template decision boundaries by mutating and testing on OOS data.
Reuses existing simulator infrastructure — no new simulation code.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import time
from typing import Any

import numpy as np

from data.market_hours import get_market_hours_provider
from memory import get_db
from research.config import (
    EVOLUTION_COOLDOWN_MARKET_HOURS,
    EVOLUTION_COOLDOWN_OFF_HOURS,
    TEMPLATE_EVOLUTION_MIN_TRADES,
    TEMPLATE_EVOLUTION_TRAIN_PCT,
)
from research.simulator import simulate
from signals.templates import (
    TEMPLATE_DEFS,
    init_default_boundaries,
    load_boundaries,
    save_boundaries,
)

logger = logging.getLogger(__name__)

# Mutation ranges for boundary parameters
_MUTATION_DELTA = {
    "composite_min": 0.05,
    "composite_max": 0.05,
    "iv_rank_min": 5.0,
    "iv_rank_max": 5.0,
    "atr_pct_min": 0.2,
    "atr_pct_max": 0.2,
}


def _normalize_boundary_pairs(params: dict[str, float]) -> dict[str, float]:
    """Ensure *_min/*_max pairs are ordered after mutation."""
    out = dict(params)
    pairs = [
        ("composite_min", "composite_max"),
        ("iv_rank_min", "iv_rank_max"),
        ("atr_pct_min", "atr_pct_max"),
    ]
    for min_k, max_k in pairs:
        if min_k in out and max_k in out:
            lo = float(out[min_k])
            hi = float(out[max_k])
            if lo > hi:
                lo, hi = hi, lo
            out[min_k] = lo
            out[max_k] = hi
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_evolution_stop_event: bool = False
_evolution_paused: bool = False
_evolution_task: "asyncio.Task | None" = None


def is_evolution_running() -> bool:
    return bool(_evolution_task and not _evolution_task.done())


def is_evolution_paused() -> bool:
    return _evolution_paused


def pause_evolution() -> bool:
    global _evolution_paused
    was = _evolution_paused
    _evolution_paused = True
    return not was


def resume_evolution() -> bool:
    global _evolution_paused
    was = _evolution_paused
    _evolution_paused = False
    return was


def stop_evolution() -> bool:
    global _evolution_stop_event
    if _evolution_stop_event:
        return False
    _evolution_stop_event = True
    return True


def start_evolution_task() -> "asyncio.Task":
    """Schedule run_template_evolution on the current loop. Idempotent."""
    global _evolution_task, _evolution_stop_event
    if is_evolution_running():
        return _evolution_task  # type: ignore[return-value]
    _evolution_stop_event = False
    _evolution_task = asyncio.get_event_loop().create_task(run_template_evolution())
    return _evolution_task


_evolution_thread = None  # type: ignore[var-annotated]


def run_template_evolution_threaded() -> None:
    """Run run_template_evolution on a dedicated daemon thread with its own
    asyncio event loop. Returns immediately.

    Isolates the long sync `time.sleep(cooldown)` (workaround for the
    Python 3.13 asyncio deque bug) from the main event loop so the agent,
    scorer, and heartbeat keep ticking.
    """
    import threading

    global _evolution_thread
    if _evolution_thread is not None and _evolution_thread.is_alive():
        return

    def _thread_target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_template_evolution())
        except Exception:
            logger.exception("Template evolution thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                pass

    _evolution_thread = threading.Thread(
        target=_thread_target, name="template_evolution", daemon=True
    )
    _evolution_thread.start()


async def run_template_evolution() -> None:
    """
    Continuous evolution loop — runs forever alongside agent and scorer.

    Cooldown: 30min during market hours, 5min outside.
    Each round:
      1. Load historical composites + price outcomes
      2. For each template, try boundary mutations
      3. Walk-forward validate: train 70% / OOS 30%
      4. Keep mutations that beat current on OOS
      5. Persist winning boundaries + update performance
    """
    global _evolution_stop_event, _evolution_paused
    logger.info("Template evolution loop started")
    mhp = get_market_hours_provider()
    conn = get_db()
    init_default_boundaries(conn)

    while True:
        if _evolution_stop_event:
            logger.info("Template evolution stopped by request")
            break
        if _evolution_paused:
            await asyncio.sleep(5)
            continue
        try:
            # Determine cooldown based on market hours
            is_open = mhp.is_market_open(include_extended=False)
            cooldown = (
                EVOLUTION_COOLDOWN_MARKET_HOURS
                if is_open
                else EVOLUTION_COOLDOWN_OFF_HOURS
            )

            await _evolution_round(conn)

            # Sync sleep — Python 3.13 asyncio deque bug crashes event loop
            # on long awaits. This loop runs in its own thread (see
            # run_template_evolution_threaded) so blocking is local.
            time.sleep(cooldown)
        except asyncio.CancelledError:
            logger.info("Template evolution cancelled")
            break
        except Exception as e:
            logger.error("Template evolution error: %s", e, exc_info=True)
            time.sleep(60)


# ---------------------------------------------------------------------------
# Single evolution round
# ---------------------------------------------------------------------------

async def _evolution_round(conn) -> None:
    """Run one round of template evolution."""
    t0 = time.time()

    # Load historical data
    composites = _load_historical_composites(conn)
    if len(composites) < TEMPLATE_EVOLUTION_MIN_TRADES:
        logger.debug(
            "Not enough composite history (%d < %d) — skipping evolution",
            len(composites), TEMPLATE_EVOLUTION_MIN_TRADES,
        )
        return

    # Walk-forward split
    split_idx = int(len(composites) * TEMPLATE_EVOLUTION_TRAIN_PCT)
    train_data = composites[:split_idx]
    oos_data = composites[split_idx:]

    if len(oos_data) < 5:
        logger.debug("OOS set too small (%d) — skipping", len(oos_data))
        return

    current_boundaries = load_boundaries(conn)
    improved = 0

    for tname, tdef in TEMPLATE_DEFS.items():
        current_b = _normalize_boundary_pairs(current_boundaries.get(tname, {}))
        if not current_b:
            continue

        # Evaluate current boundaries on OOS
        current_oos_metrics = _evaluate_boundaries(tname, current_b, oos_data)

        # Generate and test mutations
        best_mutation = None
        best_fitness = current_oos_metrics.get("search_fitness", 0.0)

        for _ in range(5):  # 5 mutations per template per round
            mutated = _mutate_boundaries(tname, current_b, tdef)

            # Quick reject on training data
            train_metrics = _evaluate_boundaries(tname, mutated, train_data)
            if train_metrics.get("search_fitness", 0.0) < best_fitness * 0.8:
                continue  # Not promising even on train

            # OOS validation
            oos_metrics = _evaluate_boundaries(tname, mutated, oos_data)
            mutation_fitness = oos_metrics.get("search_fitness", 0.0)

            if mutation_fitness > best_fitness:
                best_fitness = mutation_fitness
                best_mutation = mutated

        # Keep-gate: mutation must beat current on OOS
        if best_mutation is not None:
            generation = int(current_b.get("_generation", 0)) + 1
            save_boundaries(
                conn, tname, best_mutation,
                generation=generation, fitness=best_fitness,
            )
            _update_performance(conn, tname, oos_data, best_mutation)
            improved += 1
            logger.info(
                "Template %s evolved (gen %d, fitness %.3f)",
                tname, generation, best_fitness,
            )

    elapsed = time.time() - t0
    logger.info(
        "Evolution round complete: %d/%d templates improved in %.1fs",
        improved, len(TEMPLATE_DEFS), elapsed,
    )


# ---------------------------------------------------------------------------
# Boundary mutation
# ---------------------------------------------------------------------------

def _mutate_boundaries(
    tname: str,
    current: dict[str, float],
    tdef: Any,
) -> dict[str, float]:
    """Generate a mutated copy of boundary parameters."""
    mutated = dict(current)

    # Pick 1-3 parameters to mutate
    params = [k for k in current if not k.startswith("_")]
    n_mutations = random.randint(1, min(3, len(params)))
    targets = random.sample(params, n_mutations)

    for pname in targets:
        delta = _MUTATION_DELTA.get(pname, 0.05)
        change = random.uniform(-delta, delta)
        new_val = mutated[pname] + change

        # Clamp to valid range from template def
        bounds = tdef.default_boundaries.get(pname)
        if bounds:
            lo, hi = bounds
            new_val = max(lo, min(hi, new_val))

        mutated[pname] = round(new_val, 4)

    return _normalize_boundary_pairs(mutated)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_boundaries(
    template_name: str,
    boundaries: dict[str, float],
    data: list[dict],
) -> dict[str, float]:
    """
    Evaluate boundary configuration against historical composite data.

    Returns metrics dict with search_fitness (capped at 5.0).
    """
    b = _normalize_boundary_pairs(boundaries)
    comp_min = b.get("composite_min", 0.25)
    comp_max = b.get("composite_max", 1.0)
    iv_min = b.get("iv_rank_min", 0.0)
    iv_max = b.get("iv_rank_max", 100.0)

    trades = []
    for entry in data:
        abs_comp = abs(entry.get("composite_score", 0.0))
        iv = entry.get("iv_rank")
        fwd_ret = entry.get("forward_return", 0.0)

        # Check boundary match
        if not (comp_min <= abs_comp <= comp_max):
            continue
        if iv is not None and not (iv_min <= iv <= iv_max):
            continue

        # Score: did composite correctly predict direction?
        composite = entry.get("composite_score", 0.0)
        if composite > 0:
            trades.append(fwd_ret)  # Long trade
        else:
            trades.append(-fwd_ret)  # Short trade

    if len(trades) < 5:
        return {"search_fitness": 0.0, "trades": len(trades)}

    trades_arr = np.array(trades)
    avg_ret = float(np.mean(trades_arr))
    win_rate = float(np.sum(trades_arr > 0) / len(trades_arr))
    std_ret = float(np.std(trades_arr)) if len(trades_arr) > 1 else 1.0

    # Profit factor
    gains = trades_arr[trades_arr > 0].sum()
    losses = abs(trades_arr[trades_arr < 0].sum())
    pf = float(gains / max(losses, 1e-9))

    # Sharpe (annualized, ~252 trading days)
    sharpe = float(avg_ret / max(std_ret, 1e-9) * np.sqrt(252)) if std_ret > 0 else 0.0

    # Composite fitness: balanced metric capped at 5.0
    fitness = min(5.0, sharpe * 0.4 + win_rate * 2.0 + min(pf, 3.0) * 0.5)

    return {
        "search_fitness": max(0.0, fitness),
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_return_pct": avg_ret,
        "profit_factor": pf,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Data loading & persistence
# ---------------------------------------------------------------------------

def _load_historical_composites(conn) -> list[dict]:
    """Load composite scores + forward returns for evolution."""
    cur = conn.execute(
        """
        SELECT c.symbol, c.ts, c.composite_score, c.signal_breakdown_json
        FROM composite_scores c
        ORDER BY c.ts
        """
    )
    # Simplified: load composites and compute forward returns from adjacent rows
    rows = cur.fetchall()
    composites = []

    # Group by symbol and compute forward returns
    by_symbol: dict[str, list] = {}
    for row in rows:
        sym, ts, comp_score, breakdown_json, *_ = row
        by_symbol.setdefault(sym, []).append({
            "symbol": sym,
            "ts": ts,
            "composite_score": float(comp_score),
        })

    # For forward returns, use signal_returns averages as proxy.
    # signal_returns.ts matches signal_scores.ts, NOT composite_scores.ts,
    # so we build a (symbol → [(ts, fwd_ret)]) index and find the closest
    # signal ts that precedes each composite ts.
    cur2 = conn.execute(
        """
        SELECT symbol, ts, AVG(forward_return) as fwd_ret
        FROM signal_returns
        GROUP BY symbol, ts
        ORDER BY ts
        """
    )
    fwd_by_symbol: dict[str, list[tuple[float, float]]] = {}
    for row in cur2.fetchall():
        sym, ts, fwd = row
        fwd_by_symbol.setdefault(sym, []).append((float(ts), float(fwd)))

    for sym, entries in by_symbol.items():
        sym_fwds = fwd_by_symbol.get(sym, [])
        for entry in entries:
            comp_ts = entry["ts"]
            # Find the latest signal ts <= comp_ts
            best_fwd = None
            for sig_ts, fwd in sym_fwds:
                if sig_ts <= comp_ts:
                    best_fwd = fwd
                else:
                    break
            if best_fwd is not None:
                entry["forward_return"] = best_fwd
                composites.append(entry)

    composites.sort(key=lambda x: x["ts"])
    return composites


def _update_performance(
    conn,
    template_name: str,
    oos_data: list[dict],
    boundaries: dict[str, float],
) -> None:
    """Update template_performance table with OOS metrics."""
    metrics = _evaluate_boundaries(template_name, boundaries, oos_data)
    now = time.time()

    conn.execute(
        "INSERT OR REPLACE INTO template_performance "
        "(template_name, regime_key, composite_bucket, trades, wins, "
        "avg_return_pct, sharpe, updated_ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            template_name,
            "all",  # Aggregate across regimes for now
            "all",
            metrics.get("trades", 0),
            int(metrics.get("win_rate", 0) * metrics.get("trades", 0)),
            metrics.get("avg_return_pct", 0.0),
            metrics.get("sharpe"),
            now,
        ),
    )
    conn.commit()
