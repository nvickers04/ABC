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
        "status": "ok" | "insufficient_data" | "fallback_equal"
                  | "circuit_breaker_neff",
    }

    Circuit breaker: if N_eff stays below ``_NEFF_CIRCUIT_THRESHOLD`` for
    ``_NEFF_CIRCUIT_STREAK`` consecutive successful runs we switch to equal
    weights (``status = "circuit_breaker_neff"``) — the combination engine
    has lost its structural advantage and equal weights are safer than
    weights derived from highly correlated signals.  The streak counter is
    persisted across process restarts via ``research_config``.
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

    # ── N_eff monitoring + streak-based circuit breaker ─────────
    # With 50 signals in 5 categories (price/fundamental/macro/volatility/
    # microstructure), and categories themselves correlated (e.g. market-
    # direction dominates both "price" and "macro"), the realistic N_eff
    # ceiling is ~5-6.  We only warn/trip below ~3 where the combiner has
    # genuinely lost structural advantage.
    if n_eff < 3:
        logger.warning("N_eff %.1f < 3 — very few independent dimensions", n_eff)

    streak = _update_neff_streak(db_conn, n_eff)
    status = "ok"
    if streak >= _NEFF_CIRCUIT_STREAK:
        logger.warning(
            "Circuit breaker tripped: N_eff < %.1f for %d consecutive rounds — "
            "falling back to equal weights",
            _NEFF_CIRCUIT_THRESHOLD, streak,
        )
        w_dict = {s: 1.0 / N for s in signal_names}
        status = "circuit_breaker_neff"

    _persist_weights(db_conn, w_dict, n_eff)

    # ── Signal attribution: per-signal IC (article Part 1 homework) ──
    # IC = corr(score_at_entry, forward_return).  Logs top/bottom 5 and
    # tracks consecutive-round streaks of IC ≤ 0 for retirement review.
    try:
        _compute_and_log_ic_attribution(db_conn, signal_names)
    except Exception as e:  # non-fatal — never break the combiner
        logger.debug("IC attribution failed: %s", e)

    return {"weights": w_dict, "n_eff": n_eff, "status": status}


# ── N_eff circuit-breaker tunables ──────────────────────────────
# Calibrated for a 50-signal / 5-category universe where the realistic
# N_eff ceiling is ~5-6 (categories themselves share market-direction
# variance).  Threshold 2.5 = "single factor drowning out everything".
_NEFF_CIRCUIT_THRESHOLD = 2.5
_NEFF_CIRCUIT_STREAK = 3
_NEFF_STREAK_KEY = "n_eff_low_streak"

# ── Structural tunables (improvements #1-6) ─────────────────────
# SNR = |mean(R)| / std(R).  Below this, signal × forward-return is
# pure noise — drop from weighting.
_SNR_FLOOR = 0.05

# Number of periods a signal must be constant (zero-variance) before
# it gets flagged as "dead" with a WARNING.
_DEAD_SIGNAL_PERIODS = 100

# EWMA half-life for correlation weighting (in periods).  Clamped to
# >= 30 so we don't over-react on thin histories.
_EWMA_HALFLIFE_FRACTION = 0.25
_EWMA_HALFLIFE_MIN = 30

# Max |weight| per category (after step-11 normalization).  Prevents a
# single category (e.g. "price" with 15 correlated signals) from
# dominating the portfolio.  0.40 allows 2x the equal-category share.
_CATEGORY_WEIGHT_CAP = 0.40

# ── IC attribution tunables ─────────────────────────────────────
# Rolling window for computing per-signal Information Coefficient
# (Pearson corr between score_at_entry and forward_return).
_IC_WINDOW_DAYS = 60
# Minimum observations before IC / t-stat are trusted.  Below this we
# still log the IC but don't count it toward retirement streaks.
_IC_MIN_OBS = 30
# Consecutive rounds of IC ≤ 0 (with enough obs) before we emit a
# RETIRE_CANDIDATE warning.  Each round = one combine_signals() call.
_IC_RETIRE_STREAK = 5
# How many signals to log in top/bottom attribution lists each round.
_IC_ATTRIBUTION_TOP_K = 5


def _update_neff_streak(db_conn, n_eff: float) -> int:
    """Increment / reset the persistent low-N_eff streak counter and return
    the new value.

    The counter is stored in the ``research_config`` table via the existing
    ``get_research_config`` / ``set_research_config`` helpers — no schema
    change required.  Storage failures are non-fatal: the function falls
    back to in-process tracking so the combiner keeps working.
    """
    try:
        from memory import get_research_config, set_research_config
        prev = int(get_research_config(_NEFF_STREAK_KEY, 0))
        new = prev + 1 if n_eff < _NEFF_CIRCUIT_THRESHOLD else 0
        if new != prev:
            set_research_config(
                _NEFF_STREAK_KEY,
                float(new),
                reason=f"combiner: n_eff={n_eff:.2f}",
            )
        return new
    except Exception as e:
        logger.debug("N_eff streak persistence failed: %s", e)
        return 1 if n_eff < _NEFF_CIRCUIT_THRESHOLD else 0


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
    """Execute the 11-step alpha combination and return (weights, n_eff).

    Enhancements over the raw procedure:
    - Zero-variance signals forced to weight 0 (no information).
    - Low SNR signals (|mean(R)| / std(R) < floor) forced to weight 0.
    - N_eff uses Ledoit-Wolf shrinkage + EWMA-weighted correlation
      (more stable than raw sample correlation when T ~ N).
    - Per-category weight caps prevent single-category dominance.
    - Per-category N_eff is logged for diagnostics.
    - Persistent dead signals are flagged with a WARNING.
    """
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
    try:
        beta, _residuals, _rank, _sv = np.linalg.lstsq(Lambda, E_normalized, rcond=None)
        epsilon = E_normalized - Lambda @ beta
    except (np.linalg.LinAlgError, ValueError):
        logger.warning("OLS regression failed — using E_normalized directly")
        epsilon = E_normalized

    # Step 10: Portfolio weight for each signal
    raw_w = epsilon / sigma  # shape (N,)

    # ── Filters (improvements #1 + #5) ──────────────────────────
    # Zero-variance signals: constant R produces near-zero sigma and
    # degenerate epsilon/sigma → ∞ that dominate after normalization.
    zero_var_mask = sigma_sq <= 1e-14
    raw_w[zero_var_mask] = 0.0

    # Low-SNR signals: |mean(R)| / std(R) below floor means the signal
    # has no directional edge regardless of how uncorrelated it is.
    abs_mean = np.abs(R.mean(axis=1))
    std = np.sqrt(np.maximum(R.var(axis=1), 1e-24))
    snr = abs_mean / std
    low_snr_mask = (~zero_var_mask) & (snr < _SNR_FLOOR)
    raw_w[low_snr_mask] = 0.0

    # Step 11: Normalize so sum|w| = 1
    abs_sum = np.abs(raw_w).sum()
    if abs_sum < 1e-12:
        weights = np.ones(N) / N
    else:
        weights = raw_w / abs_sum

    # ── Per-category weight caps (improvement #6) ───────────────
    weights = _apply_category_caps(weights, signal_names)

    # ── N_eff with Ledoit-Wolf shrinkage + EWMA (improvements #3 + #4) ──
    active_mask = ~zero_var_mask
    n_active = int(active_mask.sum())

    if n_active >= 2:
        R_active = R[active_mask]
        halflife = max(_EWMA_HALFLIFE_MIN, int(M * _EWMA_HALFLIFE_FRACTION))
        C = _ewma_shrunk_correlation(R_active, halflife=halflife)
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

    # ── Per-category N_eff diagnostics (improvement #2) ─────────
    _log_category_n_eff(R, active_mask, signal_names)

    # ── Dead-signal diagnostics (improvement #1) ────────────────
    if M >= _DEAD_SIGNAL_PERIODS:
        dead = [
            signal_names[i] for i in range(N)
            if zero_var_mask[i] and not np.all(np.isnan(R[i]))
        ]
        if dead:
            logger.warning(
                "Dead signals (zero variance over %d periods): %s",
                M, ", ".join(sorted(dead)),
            )

    # ── Hard circuit breaker — single dominant factor ───────────
    if n_eff < 2:
        logger.warning(
            "N_eff %.1f < 2 — single dominant factor, falling back to equal weights",
            n_eff,
        )
        weights = np.ones(N) / N
    elif n_eff < 4:
        logger.debug("N_eff %.1f < 4 — few independent signal dimensions", n_eff)

    w_dict = {signal_names[i]: float(weights[i]) for i in range(N)}
    return w_dict, float(n_eff)


# ---------------------------------------------------------------------------
# Internal — Correlation estimators (Ledoit-Wolf + EWMA)
# ---------------------------------------------------------------------------

def _ewma_shrunk_correlation(R: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA-weighted correlation with Ledoit-Wolf shrinkage toward identity.

    R: (n_active, M) signal return rows.  Returns (n_active, n_active) PSD
    correlation matrix.  Recent periods dominate according to half-life.
    Shrinkage intensity is chosen to minimize MSE vs the identity target
    (a simplified LW estimator suitable for correlation matrices).
    """
    n, M = R.shape
    if n < 2 or M < 2:
        return np.eye(max(n, 1))

    # EWMA weights normalized to sum to M (so effective sample size is M).
    decay = np.log(2.0) / max(halflife, 1.0)
    t = np.arange(M)
    w = np.exp(-decay * (M - 1 - t))
    w = w * (M / w.sum())  # normalize so sum(w) = M
    w_row = w[np.newaxis, :]  # (1, M)

    # Weighted mean & centered data
    mu = (R * w_row).sum(axis=1, keepdims=True) / M
    Rc = R - mu

    # Weighted covariance
    Sw = (Rc * w_row) @ Rc.T / M  # (n, n)

    # Convert to correlation
    d = np.sqrt(np.maximum(np.diag(Sw), 1e-24))
    C = Sw / np.outer(d, d)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)

    # Ledoit-Wolf shrinkage toward identity on the correlation matrix.
    # Intensity ∝ 1/M (more shrinkage when sample is short).  This is a
    # simplified LW: the full formula requires 4th-order moments but the
    # 1/M approximation is close in practice and much cheaper.
    off_diag_sq = (C ** 2).sum() - n  # excludes diagonal (which is 1)
    if off_diag_sq < 1e-12:
        return C
    lam = min(1.0, max(0.0, (n + n * n) / (M * off_diag_sq)))
    C_shrunk = (1.0 - lam) * C + lam * np.eye(n)
    return C_shrunk


def _log_category_n_eff(
    R: np.ndarray,
    active_mask: np.ndarray,
    signal_names: list[str],
) -> None:
    """Log per-category N_eff so we can see which category is saturated."""
    cats: dict[str, list[int]] = {}
    for i, name in enumerate(signal_names):
        if not active_mask[i]:
            continue
        sig = SIGNAL_REGISTRY.get(name)
        cat = sig.category if sig else "unknown"
        cats.setdefault(cat, []).append(i)

    parts = []
    for cat in sorted(cats.keys()):
        idx = cats[cat]
        if len(idx) < 2:
            parts.append(f"{cat}={len(idx)}")
            continue
        C = np.corrcoef(R[idx])
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)
        ev = np.maximum(np.linalg.eigvalsh(C), 0.0)
        s1, s2 = ev.sum(), (ev ** 2).sum()
        n_eff_cat = (s1 * s1 / s2) if s2 > 1e-12 else float(len(idx))
        parts.append(f"{cat}={n_eff_cat:.1f}/{len(idx)}")
    if parts:
        logger.info("N_eff by category: %s", " ".join(parts))


def _apply_category_caps(
    weights: np.ndarray,
    signal_names: list[str],
) -> np.ndarray:
    """Cap |weight| sum per category at `_CATEGORY_WEIGHT_CAP`.

    If a category's total |weight| exceeds the cap, scale all weights in
    that category down proportionally.  After capping, rescale the full
    vector so sum|w| = 1 again.  Categories under the cap are unchanged.
    """
    if weights.size == 0:
        return weights

    cat_indices: dict[str, list[int]] = {}
    for i, name in enumerate(signal_names):
        sig = SIGNAL_REGISTRY.get(name)
        cat = sig.category if sig else "unknown"
        cat_indices.setdefault(cat, []).append(i)

    w = weights.copy()
    abs_w = np.abs(w)

    for cat, idx in cat_indices.items():
        cat_sum = abs_w[idx].sum()
        if cat_sum > _CATEGORY_WEIGHT_CAP:
            scale = _CATEGORY_WEIGHT_CAP / cat_sum
            w[idx] *= scale

    abs_total = np.abs(w).sum()
    if abs_total > 1e-12:
        w = w / abs_total
    return w


# ---------------------------------------------------------------------------
# Internal — Information Coefficient (IC) attribution
# ---------------------------------------------------------------------------

def _compute_signal_ic(
    db_conn,
    signal_names: list[str],
    window_days: int = _IC_WINDOW_DAYS,
) -> dict[str, dict[str, float]]:
    """Per-signal IC = Pearson corr(score_at_entry, forward_return).

    Uses all (symbol, ts) observations in the rolling window.  Also
    returns the IC t-statistic (two-sided) and observation count.

    Returns: {signal_name: {"ic": float, "t": float, "n": int}}
    Signals with no data in the window are omitted.
    """
    cutoff = time.time() - window_days * 86400.0
    placeholders = ",".join("?" * len(signal_names))
    cur = db_conn.execute(
        f"""
        SELECT signal_name, score_at_entry, forward_return
          FROM signal_returns
         WHERE ts >= ? AND signal_name IN ({placeholders})
        """,
        (cutoff, *signal_names),
    )

    buckets: dict[str, list[tuple[float, float]]] = {}
    for sig, s, r in cur.fetchall():
        buckets.setdefault(sig, []).append((float(s), float(r)))

    out: dict[str, dict[str, float]] = {}
    for sig, pairs in buckets.items():
        n = len(pairs)
        if n < 3:
            continue
        arr = np.asarray(pairs, dtype=float)
        s, r = arr[:, 0], arr[:, 1]
        s_std = float(s.std())
        r_std = float(r.std())
        if s_std < 1e-12 or r_std < 1e-12:
            out[sig] = {"ic": 0.0, "t": 0.0, "n": n}
            continue
        ic = float(np.corrcoef(s, r)[0, 1])
        if not np.isfinite(ic):
            ic = 0.0
        # Two-sided t-stat: t = IC * sqrt(n-2) / sqrt(1 - IC^2)
        denom = max(1e-12, 1.0 - ic * ic)
        t_stat = float(ic * np.sqrt(max(n - 2, 0)) / np.sqrt(denom))
        out[sig] = {"ic": ic, "t": t_stat, "n": n}
    return out


def _log_ic_attribution(ic_stats: dict[str, dict[str, float]]) -> None:
    """Log top-K and bottom-K signals by IC each round.

    Format chosen for easy grep: ``IC_TOP`` / ``IC_BOT`` / ``IC_STAT``.
    """
    trusted = [
        (name, d["ic"], d["t"], d["n"])
        for name, d in ic_stats.items()
        if d["n"] >= _IC_MIN_OBS
    ]
    if not trusted:
        logger.info("IC_STAT no_signals_with_min_obs=%d", _IC_MIN_OBS)
        return

    trusted.sort(key=lambda x: x[1], reverse=True)
    k = _IC_ATTRIBUTION_TOP_K
    top = trusted[:k]
    bot = list(reversed(trusted[-k:]))  # worst first

    ics = np.asarray([x[1] for x in trusted])
    logger.info(
        "IC_STAT mean=%.4f median=%.4f std=%.4f n_signals=%d min_obs=%d",
        float(ics.mean()), float(np.median(ics)), float(ics.std()),
        len(trusted), _IC_MIN_OBS,
    )
    for name, ic, t, n in top:
        logger.info("IC_TOP %-32s ic=%+.4f t=%+.2f n=%d", name, ic, t, n)
    for name, ic, t, n in bot:
        logger.info("IC_BOT %-32s ic=%+.4f t=%+.2f n=%d", name, ic, t, n)


def _update_ic_retirement(
    db_conn,
    ic_stats: dict[str, dict[str, float]],
) -> None:
    """Track per-signal streaks of non-positive IC and warn on retirement.

    Streaks are persisted via ``research_config`` under key
    ``ic_neg_streak:{signal}``.  A signal with IC ≤ 0 and n >= min-obs
    for ``_IC_RETIRE_STREAK`` consecutive combiner rounds is logged as a
    RETIRE_CANDIDATE (WARNING).  Positive IC resets the streak.

    Persistence failures are non-fatal.
    """
    try:
        from memory import get_research_config, set_research_config
    except Exception as e:
        logger.debug("IC retirement persistence unavailable: %s", e)
        return

    for name, d in ic_stats.items():
        if d["n"] < _IC_MIN_OBS:
            continue
        key = f"ic_neg_streak:{name}"
        prev = int(get_research_config(key, 0))
        new = prev + 1 if d["ic"] <= 0.0 else 0
        if new != prev:
            try:
                set_research_config(
                    key,
                    float(new),
                    reason=f"combiner: ic={d['ic']:+.4f} n={d['n']}",
                )
            except Exception:
                pass
        if new >= _IC_RETIRE_STREAK and (new == _IC_RETIRE_STREAK or new % 5 == 0):
            logger.warning(
                "RETIRE_CANDIDATE %s — IC ≤ 0 for %d consecutive rounds "
                "(ic=%+.4f t=%+.2f n=%d)",
                name, new, d["ic"], d["t"], d["n"],
            )


def _compute_and_log_ic_attribution(
    db_conn,
    signal_names: list[str],
) -> None:
    """One-shot: compute IC, log top/bottom, update retirement streaks."""
    ic_stats = _compute_signal_ic(db_conn, signal_names)
    if not ic_stats:
        return
    _log_ic_attribution(ic_stats)
    _update_ic_retirement(db_conn, ic_stats)


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
