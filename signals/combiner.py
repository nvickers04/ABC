"""
Combination engine — article's 11-step alpha combination procedure.

Produces per-signal weights and composite scores per symbol.
Operates on R(i,s) = score × forward_return (not raw signal scores).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from research.config import (
    FORWARD_RETURN_HORIZON,
    MIN_SHARED_PERIODS_FOR_COMBINATION,
    SIGNAL_WEIGHT_LOOKBACK_DAYS,
)
from signals.base import SIGNAL_REGISTRY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def combine_signals(
    db_conn,
    signal_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the 11-step alpha combination procedure.

    Returns
    -------
    {
        "weights": {signal_name: float},
        "n_eff": float,
        "status": "ok" | "insufficient_data" | "fallback_equal",
    }
    """
    if signal_names is None:
        signal_names = sorted(SIGNAL_REGISTRY.keys())

    N = len(signal_names)
    if N == 0:
        return {"weights": {}, "n_eff": 0.0, "status": "no_signals"}

    # ── Load R(i,s) matrix from signal_returns ──────────────────
    R, periods = _load_return_matrix(db_conn, signal_names)

    M = R.shape[1] if R.ndim == 2 else 0
    if M < MIN_SHARED_PERIODS_FOR_COMBINATION:
        logger.info(
            "Insufficient shared periods (%d < %d) — using equal weights",
            M, MIN_SHARED_PERIODS_FOR_COMBINATION,
        )
        weights = {s: 1.0 / N for s in signal_names}
        _persist_weights(db_conn, weights, n_eff=float(N))
        return {"weights": weights, "n_eff": float(N), "status": "insufficient_data"}

    # ── Steps 1-11 ──────────────────────────────────────────────
    try:
        w_dict, n_eff = _run_11_steps(R, signal_names)
    except Exception as e:
        logger.error("Combination engine failed (%s) — falling back to equal weights", e)
        w_dict = {s: 1.0 / N for s in signal_names}
        n_eff = float(N)
        _persist_weights(db_conn, w_dict, n_eff)
        return {"weights": w_dict, "n_eff": n_eff, "status": "fallback_equal"}

    _persist_weights(db_conn, w_dict, n_eff)
    return {"weights": w_dict, "n_eff": n_eff, "status": "ok"}


def compute_composite_scores(
    db_conn,
    weights: dict[str, float],
    symbols: list[str],
    ts: float | None = None,
) -> dict[str, float]:
    """
    Compute composite_score per symbol from current signal_scores + weights.

    composite(sym) = sum_i w(i) * score(i, sym, now)
    Returns {symbol: composite_score} and persists to DB.
    """
    if ts is None:
        ts = time.time()

    # Latest score + confidence per signal/symbol
    scores_by_sym: dict[str, dict[str, float]] = {}
    confidence_by_sym: dict[str, dict[str, float]] = {}
    cur = db_conn.execute(
        """
        SELECT signal_name, symbol, score, confidence
        FROM signal_scores
        WHERE ts = (
            SELECT MAX(ts) FROM signal_scores s2
            WHERE s2.signal_name = signal_scores.signal_name
              AND s2.symbol = signal_scores.symbol
        )
        """
    )
    for row in cur.fetchall():
        sig, sym, sc, conf = row
        scores_by_sym.setdefault(sym, {})[sig] = float(sc)
        confidence_by_sym.setdefault(sym, {})[sig] = float(conf)

    composites: dict[str, float] = {}
    breakdowns: dict[str, dict[str, float]] = {}
    for sym in symbols:
        sym_scores = scores_by_sym.get(sym, {})
        if not sym_scores:
            continue

        sym_conf = confidence_by_sym.get(sym, {})
        raw_composite = 0.0
        active_weight_sum = 0.0
        breakdown = {}
        for sig_name, w in weights.items():
            s = sym_scores.get(sig_name, 0.0)
            raw_composite += w * s
            # Count signal as active if it has confidence > 0
            if sym_conf.get(sig_name, 0.0) > 0:
                active_weight_sum += abs(w)
            sig_obj = SIGNAL_REGISTRY.get(sig_name)
            cat = sig_obj.category if sig_obj else "unknown"
            breakdown.setdefault(cat, 0.0)
            breakdown[cat] += w * s

        # Normalize by active weight sum so zero-confidence signals
        # (no data / failed) don't dilute the composite.
        if active_weight_sum > 0:
            composite = raw_composite / active_weight_sum
        else:
            composite = 0.0

        composites[sym] = max(-1.0, min(1.0, composite))
        breakdowns[sym] = breakdown

    # Persist
    now = time.time()
    rows = [
        (sym, now, composites[sym], json.dumps(breakdowns.get(sym, {})))
        for sym in composites
    ]
    if rows:
        db_conn.executemany(
            "INSERT OR REPLACE INTO composite_scores (symbol, ts, composite_score, signal_breakdown_json) VALUES (?, ?, ?, ?)",
            rows,
        )
        db_conn.commit()

    return composites


def compute_n_eff(db_conn, signal_names: list[str] | None = None) -> float:
    """Compute effective number of independent signals from R(i,s) correlation."""
    if signal_names is None:
        signal_names = sorted(SIGNAL_REGISTRY.keys())

    R, _ = _load_return_matrix(db_conn, signal_names)
    N = R.shape[0] if R.ndim == 2 else 0
    M = R.shape[1] if R.ndim == 2 else 0
    if N < 2 or M < 3:
        return float(N)

    # Exclude zero-variance signals (constant R → NaN in corrcoef)
    row_var = np.var(R, axis=1)
    active_mask = row_var > 1e-14
    n_active = int(active_mask.sum())
    if n_active < 2:
        return float(n_active) if n_active > 0 else 1.0

    C = np.corrcoef(R[active_mask])
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.maximum(eigvals, 0.0)  # Clip numerical negatives
    sum_eig = eigvals.sum()
    sum_eig_sq = (eigvals ** 2).sum()
    if sum_eig_sq < 1e-12:
        return float(n_active)
    return float(sum_eig ** 2 / sum_eig_sq)


# ---------------------------------------------------------------------------
# Internal — 11-Step Procedure
# ---------------------------------------------------------------------------

def _run_11_steps(
    R: np.ndarray,  # shape (N, M)
    signal_names: list[str],
) -> tuple[dict[str, float], float]:
    """Execute the 11-step alpha combination and return (weights, n_eff)."""
    N, M = R.shape

    # Step 1: R(i,s) already loaded — rows are signals, columns are periods

    # Step 2: Serial demean — remove each signal's average drift
    means = R.mean(axis=1, keepdims=True)
    X = R - means  # shape (N, M)

    # Step 3: Sample variance per signal
    sigma_sq = (X ** 2).mean(axis=1)  # shape (N,)
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-12))  # avoid /0

    # Step 4: Normalize to common scale
    Y = X / sigma[:, np.newaxis]  # shape (N, M)

    # Step 5: Drop the most recent observation
    if M <= 2:
        # Not enough data after dropping
        equal_w = {s: 1.0 / N for s in signal_names}
        return equal_w, float(N)
    Y_trimmed = Y[:, :-1]  # shape (N, M-1)

    # Step 6: Cross-sectional demean at each time period
    cross_mean = Y_trimmed.mean(axis=0, keepdims=True)  # shape (1, M-1)
    Lambda = Y_trimmed - cross_mean  # shape (N, M-1)

    # Step 7: Drop last period from Lambda
    if Lambda.shape[1] <= 1:
        equal_w = {s: 1.0 / N for s in signal_names}
        return equal_w, float(N)
    Lambda = Lambda[:, :-1]  # shape (N, M-2)

    # Step 8: Forward expected return for each signal
    d = min(SIGNAL_WEIGHT_LOOKBACK_DAYS, M)
    E = R[:, -d:].mean(axis=1)  # shape (N,)
    E_normalized = E / sigma  # shape (N,)

    # Step 9: Regress out shared variance
    # OLS: E_normalized ~ Lambda (no intercept)
    # Lambda: (N, M-2), E_normalized: (N,)
    # Want: E_normalized = Lambda @ beta + epsilon
    # epsilon = E_normalized - Lambda @ beta
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(Lambda, E_normalized, rcond=None)
        epsilon = E_normalized - Lambda @ beta
    except (np.linalg.LinAlgError, ValueError):
        logger.warning("OLS regression failed — using E_normalized directly")
        epsilon = E_normalized

    # Step 10: Portfolio weight for each signal
    # w(i) = eta * epsilon(i) / sigma(i)
    raw_w = epsilon / sigma  # shape (N,)

    # Step 11: Normalize so sum|w| = 1
    abs_sum = np.abs(raw_w).sum()
    if abs_sum < 1e-12:
        weights = np.ones(N) / N
    else:
        weights = raw_w / abs_sum

    # N_eff from correlation matrix — exclude zero-variance signals
    # (signals with constant R produce NaN rows in corrcoef)
    row_var = np.var(R, axis=1)
    active_mask = row_var > 1e-14
    n_active = int(active_mask.sum())

    if n_active >= 2:
        C = np.corrcoef(R[active_mask])
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)  # ensure diagonal is 1
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.maximum(eigvals, 0.0)
        sum_eig = eigvals.sum()
        sum_eig_sq = (eigvals ** 2).sum()
        n_eff = (sum_eig ** 2 / sum_eig_sq) if sum_eig_sq > 1e-12 else float(n_active)
    else:
        n_eff = float(n_active) if n_active > 0 else 1.0

    logger.info(
        "N_eff=%.1f (%d active / %d total signals, %d periods)",
        n_eff, n_active, N, M,
    )

    # N_eff circuit breaker — threshold 2: below this, there is
    # effectively only one independent factor (pure market beta)
    if n_eff < 2:
        logger.warning(
            "N_eff %.1f < 2 — single dominant factor, falling back to equal weights",
            n_eff,
        )
        weights = np.ones(N) / N
    elif n_eff < 4:
        logger.info("N_eff %.1f < 4 — few independent signal dimensions", n_eff)
    elif n_eff < 8:
        logger.info("N_eff %.1f < 8 — moderate signal correlation", n_eff)

    w_dict = {signal_names[i]: float(weights[i]) for i in range(N)}
    return w_dict, float(n_eff)


# ---------------------------------------------------------------------------
# Internal — Data Loading
# ---------------------------------------------------------------------------

def _load_return_matrix(
    db_conn,
    signal_names: list[str],
) -> tuple[np.ndarray, list[float]]:
    """
    Load R(i,s) matrix from signal_returns table.

    R(i,s) is averaged across all symbols per signal per timestamp.
    Returns (R, periods) where R.shape = (N, M) and periods is list of timestamps.
    """
    N = len(signal_names)
    name_to_idx = {name: i for i, name in enumerate(signal_names)}

    # Get all unique timestamps across all signals
    cur = db_conn.execute(
        "SELECT DISTINCT ts FROM signal_returns ORDER BY ts"
    )
    all_periods = [row[0] for row in cur.fetchall()]

    if not all_periods:
        return np.empty((N, 0)), []

    period_to_idx = {ts: j for j, ts in enumerate(all_periods)}
    M = len(all_periods)

    # Allocate and fill
    R = np.full((N, M), np.nan)
    cur = db_conn.execute(
        "SELECT signal_name, ts, AVG(r_value) FROM signal_returns GROUP BY signal_name, ts"
    )
    for row in cur.fetchall():
        sig, ts, avg_r = row
        i = name_to_idx.get(sig)
        j = period_to_idx.get(ts)
        if i is not None and j is not None:
            R[i, j] = float(avg_r)

    # Find shared periods (all signals have data)
    valid_cols = ~np.any(np.isnan(R), axis=0)
    R_shared = R[:, valid_cols]
    shared_periods = [all_periods[j] for j in range(M) if valid_cols[j]]

    return R_shared, shared_periods


def _persist_weights(
    db_conn,
    weights: dict[str, float],
    n_eff: float,
) -> None:
    """Write signal weights to DB."""
    now = time.time()
    rows = []
    for sig_name, w in weights.items():
        sig_obj = SIGNAL_REGISTRY.get(sig_name)
        cat = sig_obj.category if sig_obj else "unknown"
        rows.append((sig_name, w, n_eff, cat, now))

    db_conn.executemany(
        "INSERT OR REPLACE INTO signal_weights (signal_name, weight, n_eff, category, updated_ts) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    db_conn.commit()
