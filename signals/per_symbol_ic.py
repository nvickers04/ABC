"""
Per-symbol Information Coefficient (IC).

The combiner's existing ``_compute_signal_ic`` collapses across symbols
to produce one global IC per signal.  This module computes the same
Pearson(score_at_entry, forward_return) but bucketed per
``(signal_name, symbol, horizon_bars)``, so we can ask:

    "How well does momentum predict NVDA, specifically?"

vs the global "how well does momentum predict in general?".

It is purely derived state — every value here can be regenerated from
``signal_returns`` — so the table is safe to drop and rebuild.

This module DOES NOT modify combiner weights.  It just measures and
persists.  The cognitive / intuition layer will read these rows via
:func:`per_symbol_modifier` when fully wired in.  Until then the modifier returns
the conservative default of ``1.0`` whenever ``n_obs`` is below the
cadence-aware minimum, so callers can opt in early without changing
behavior.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Mirror the combiner's tunables so per-symbol IC stays consistent with
# the global IC pipeline.  Imported lazily inside functions to avoid a
# circular import (combiner -> per_symbol_ic -> combiner).
_DEFAULT_WINDOW_DAYS = 60
_DEFAULT_MIN_OBS = 30
_NOISE_THRESHOLD = 0.02   # |IC| below this on trusted n is "noise"
_PER_SYMBOL_MIN_OBS = 20  # floor for modifier — needs less than retirement
# Modifier amplification.  modifier = clip(per_symbol_ic_abs /
# global_ic_abs, MOD_LO, MOD_HI).  Bounded so a tiny global IC can't
# explode the multiplier and an unreliable per-symbol IC can't zero out
# a globally-trusted signal entirely.
_MOD_LO = 0.25
_MOD_HI = 2.0


def _min_obs_for_signal(signal_name: str) -> int:
    """Cadence-aware minimum observation count for a signal.

    Mirrors ``signals.combiner._min_obs_for`` but resilient to a missing
    registry entry (returns the default 30).
    """
    try:
        from signals.base import SIGNAL_REGISTRY
        from signals.combiner import _min_obs_for
        sig = SIGNAL_REGISTRY.get(signal_name)
        if sig is None:
            return _DEFAULT_MIN_OBS
        return _min_obs_for(sig)
    except Exception:
        return _DEFAULT_MIN_OBS


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute_per_symbol_ic(
    db_conn,
    signal_names: Iterable[str],
    window_days: int = _DEFAULT_WINDOW_DAYS,
) -> dict[tuple[str, str, int], dict[str, float]]:
    """Compute per-(signal, symbol, horizon) IC over the rolling window.

    Returns
    -------
    {(signal_name, symbol, horizon_bars): {"ic", "t", "n"}}

    Buckets with fewer than 3 observations are omitted.  Buckets with
    zero variance in either score or return are reported with ic=0.
    """
    names = list(signal_names)
    if not names:
        return {}

    cutoff = time.time() - max(0.0, float(window_days)) * 86400.0
    placeholders = ",".join("?" * len(names))
    cur = db_conn.execute(
        f"""
        SELECT signal_name, symbol, horizon_bars, score_at_entry, forward_return
          FROM signal_returns
         WHERE ts >= ? AND signal_name IN ({placeholders})
        """,
        (cutoff, *names),
    )

    buckets: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    for sig, sym, horizon, s, r in cur.fetchall():
        try:
            key = (str(sig), str(sym), int(horizon))
            buckets.setdefault(key, []).append((float(s), float(r)))
        except (TypeError, ValueError):
            continue

    out: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, pairs in buckets.items():
        n = len(pairs)
        if n < 3:
            continue
        arr = np.asarray(pairs, dtype=float)
        s, r = arr[:, 0], arr[:, 1]
        if float(s.std()) < 1e-12 or float(r.std()) < 1e-12:
            out[key] = {"ic": 0.0, "t": 0.0, "n": n}
            continue
        ic = float(np.corrcoef(s, r)[0, 1])
        if not np.isfinite(ic):
            ic = 0.0
        denom = max(1e-12, 1.0 - ic * ic)
        t_stat = float(ic * np.sqrt(max(n - 2, 0)) / np.sqrt(denom))
        out[key] = {"ic": ic, "t": t_stat, "n": n}
    return out


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def update_per_symbol_ic(
    db_conn,
    signal_names: Iterable[str],
    window_days: int = _DEFAULT_WINDOW_DAYS,
) -> int:
    """Recompute per-symbol IC and upsert into ``signal_symbol_ic``.

    Also tracks a noise streak per (signal, symbol, horizon): incremented
    whenever |IC| < ``_NOISE_THRESHOLD`` on a row that meets the
    cadence-aware min-obs threshold; reset to 0 otherwise.  Streaks let
    the cognitive layer down-weight stubbornly noisy combos without
    relying on a single round.

    Returns the number of rows upserted.  Persistence failures are
    swallowed and logged at DEBUG so a busted IC update can never crash
    a combiner round.
    """
    try:
        ic = compute_per_symbol_ic(db_conn, signal_names, window_days=window_days)
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("compute_per_symbol_ic failed: %s", e)
        return 0
    if not ic:
        return 0

    now_ts = float(time.time())
    rows: list[tuple[str, str, int, float, float, int, int, float]] = []
    try:
        # Pull existing streaks in one query so we don't issue 25*50 SELECTs.
        cur = db_conn.execute(
            "SELECT signal_name, symbol, horizon_bars, ic_neg_streak FROM signal_symbol_ic"
        )
        prev_streak: dict[tuple[str, str, int], int] = {
            (str(r[0]), str(r[1]), int(r[2])): int(r[3] or 0) for r in cur.fetchall()
        }
    except Exception:
        prev_streak = {}

    for (sig, sym, horizon), d in ic.items():
        n = int(d["n"])
        ic_v = float(d["ic"])
        t_v = float(d["t"])
        # Streak only ticks when n is past the cadence-aware threshold —
        # otherwise we'd flag every newly-tracked symbol as noisy.
        if n >= _min_obs_for_signal(sig) and abs(ic_v) < _NOISE_THRESHOLD:
            streak = prev_streak.get((sig, sym, horizon), 0) + 1
        else:
            streak = 0
        rows.append((sig, sym, horizon, ic_v, t_v, n, streak, now_ts))

    try:
        db_conn.executemany(
            """
            INSERT OR REPLACE INTO signal_symbol_ic
                (signal_name, symbol, horizon_bars, ic, t_stat, n_obs,
                 ic_neg_streak, last_updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        db_conn.commit()
    except Exception as e:
        logger.debug("signal_symbol_ic upsert failed: %s", e)
        return 0

    logger.info(
        "PER_SYMBOL_IC upserted=%d window_days=%d signals=%d",
        len(rows), int(window_days), len(set(r[0] for r in rows)),
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Read accessors
# ---------------------------------------------------------------------------

def get_per_symbol_ic(
    db_conn,
    signal_name: str,
    symbol: str,
    horizon_bars: int | None = None,
) -> dict | None:
    """Read the latest per-symbol IC row for the given key, or None.

    If ``horizon_bars`` is None, returns the row with the most observations
    (most reliable horizon for that combo).
    """
    try:
        if horizon_bars is None:
            cur = db_conn.execute(
                """
                SELECT signal_name, symbol, horizon_bars, ic, t_stat, n_obs,
                       ic_neg_streak, last_updated_ts
                  FROM signal_symbol_ic
                 WHERE signal_name = ? AND symbol = ?
                 ORDER BY n_obs DESC
                 LIMIT 1
                """,
                (signal_name, symbol),
            )
        else:
            cur = db_conn.execute(
                """
                SELECT signal_name, symbol, horizon_bars, ic, t_stat, n_obs,
                       ic_neg_streak, last_updated_ts
                  FROM signal_symbol_ic
                 WHERE signal_name = ? AND symbol = ? AND horizon_bars = ?
                """,
                (signal_name, symbol, int(horizon_bars)),
            )
        row = cur.fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return {
        "signal_name": row[0],
        "symbol": row[1],
        "horizon_bars": int(row[2]),
        "ic": float(row[3]),
        "t_stat": float(row[4]),
        "n_obs": int(row[5]),
        "ic_neg_streak": int(row[6] or 0),
        "last_updated_ts": float(row[7]),
    }


def per_symbol_modifier(
    db_conn,
    signal_name: str,
    symbol: str,
    *,
    global_ic: float | None = None,
    horizon_bars: int | None = None,
    default: float = 1.0,
) -> float:
    """Return a per-symbol weight modifier in roughly ``[0.25, 2.0]``.

    The cognitive layer multiplies the signal's global weight by this
    modifier when scoring a specific symbol.

    Semantics:
      - If we don't have enough per-symbol observations (n < 20) → return
        ``default`` (typically 1.0). Conservative: don't claim per-symbol
        info we haven't earned.
      - If the global IC is unknown / ~0 → return ``default``. We can't
        meaningfully compare a per-symbol IC against a missing baseline.
      - Otherwise → ``clip(|ic_per_symbol| / |ic_global|, 0.25, 2.0)``.
        Same-sign agreement preserved by using absolute values; sign
        flipping is handled by the combiner's existing weight optimizer.
      - If the per-symbol noise streak is at/over the retirement
        threshold, the modifier is hard-capped at 0.25.

    All failures fall back to ``default`` — this function MUST NOT raise.
    """
    try:
        row = get_per_symbol_ic(db_conn, signal_name, symbol, horizon_bars)
    except Exception:
        return default
    if row is None or row["n_obs"] < _PER_SYMBOL_MIN_OBS:
        return default

    g = abs(float(global_ic)) if global_ic is not None else 0.0
    if g < _NOISE_THRESHOLD:
        return default

    raw = abs(row["ic"]) / g
    mod = max(_MOD_LO, min(_MOD_HI, raw))

    # Hard cap when per-symbol noise streak suggests this combo is dead.
    try:
        from signals.combiner import _IC_RETIRE_STREAK
        if row["ic_neg_streak"] >= _IC_RETIRE_STREAK:
            mod = min(mod, _MOD_LO)
    except Exception:
        pass
    return float(mod)
