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
    MIN_SHARED_PERIODS_FOR_COMBINATION,
    SIGNAL_WEIGHT_LOOKBACK_DAYS,
)
from signals.base import SIGNAL_REGISTRY
from signals.candidate_lifecycle import (
    apply_candidate_weight_caps,
    filter_live_signal_names,
    update_candidate_lifecycle,
)

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
    signal_names = filter_live_signal_names(signal_names)

    N = len(signal_names)
    if N == 0:
        return {"weights": {}, "n_eff": 0.0, "status": "no_signals"}

    # ── Steps 1-11 (bucket-stratified by horizon) ───────────────
    # Signals with different return cadences/horizons are benchmarked
    # inside their own (resolution, horizon) bucket first, then merged.
    # This avoids a 1h/h=4 stack being penalized against D/h=10 dynamics.
    try:
        if _STRATIFY_BY_HORIZON_BUCKET:
            w_dict, combine_status = _run_11_steps_stratified(db_conn, signal_names)
            # Keep structural N_eff measured on the full shared matrix so
            # the existing low-N_eff monitoring/circuit semantics remain
            # comparable to prior rounds.
            n_eff = float(compute_n_eff(db_conn, signal_names=signal_names))
            if combine_status == "insufficient_data":
                logger.info(
                    "Insufficient data across all horizon buckets — using equal weights"
                )
                try:
                    ic_stats = _compute_and_log_ic_attribution(db_conn, signal_names)
                    _publish_ir_snapshot(
                        db_conn,
                        ic_stats,
                        n_eff=float(N),
                        signal_names=signal_names,
                    )
                    _safe_update_per_symbol_ic(db_conn, signal_names)
                except Exception as e:
                    logger.debug("IC attribution failed on insufficient-data path: %s", e)
                _persist_weights(db_conn, w_dict, n_eff=float(N))
                return {"weights": w_dict, "n_eff": float(N), "status": "insufficient_data"}
        else:
            # Legacy pooled mode (all signals share one return matrix).
            R, _periods = _load_return_matrix(db_conn, signal_names)
            M = R.shape[1] if R.ndim == 2 else 0
            if M < MIN_SHARED_PERIODS_FOR_COMBINATION:
                logger.info(
                    "Insufficient shared periods (%d < %d) — using equal weights",
                    M, MIN_SHARED_PERIODS_FOR_COMBINATION,
                )
                weights = {s: 1.0 / N for s in signal_names}
                try:
                    ic_stats = _compute_and_log_ic_attribution(db_conn, signal_names)
                    _publish_ir_snapshot(
                        db_conn,
                        ic_stats,
                        n_eff=float(N),
                        signal_names=signal_names,
                    )
                    _safe_update_per_symbol_ic(db_conn, signal_names)
                except Exception as e:
                    logger.debug("IC attribution failed on insufficient-data path: %s", e)
                _persist_weights(db_conn, weights, n_eff=float(N))
                return {"weights": weights, "n_eff": float(N), "status": "insufficient_data"}
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

    # ── Signal attribution: per-signal IC (article Part 1 homework) ──
    # Computed BEFORE persisting weights so retired signals can be zeroed
    # and estimated IR can be published to the briefing in the same round.
    ic_stats: dict[str, dict[str, float]] = {}
    try:
        ic_stats = _compute_and_log_ic_attribution(db_conn, signal_names)
        update_candidate_lifecycle(ic_stats, evaluated_signal_names=signal_names)
        _safe_update_per_symbol_ic(db_conn, signal_names)
    except Exception as e:  # non-fatal — never break the combiner
        logger.debug("IC attribution failed: %s", e)

    # ── Auto-zero RETIRE_CANDIDATE signals ──────────────────────
    # Any signal whose `ic_neg_streak:{name}` >= _IC_RETIRE_STREAK has
    # demonstrated IC ≤ 0 for several consecutive rounds on a trusted
    # sample size — we stop giving it weight until its IC recovers.
    if status == "ok":
        w_dict = _apply_ic_retirement_mask(db_conn, w_dict)
        w_dict = apply_candidate_weight_caps(w_dict)

    # ── Estimated IR + gate ─────────────────────────────────────
    # IR ≈ mean(positive IC) × √N_eff  (Fundamental Law of Active Mgmt).
    # Persist snapshot to research_config so the briefing can surface it
    # to the trading agent as an ADVISORY conviction multiplier — not a
    # binary trade permission.
    estimated_ir, gate_open = _publish_ir_snapshot(
        db_conn,
        ic_stats,
        n_eff,
        signal_names=signal_names,
    )
    if not gate_open:
        logger.info(
            "Quant edge currently weak (IR=%.4f < %.4f): advisory — "
            "agent should prefer position management, hedges, and smaller size.",
            estimated_ir, _IR_GATE_MIN,
        )

    _persist_weights(db_conn, w_dict, n_eff)
    return {
        "weights": w_dict,
        "n_eff": n_eff,
        "status": status,
        "estimated_ir": estimated_ir,
        "ir_gate_open": gate_open,
    }


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

# Horizon-bucket combiner: run step-1..11 inside each
# (return_resolution, return_horizon) bucket, then merge.
_STRATIFY_BY_HORIZON_BUCKET = True
# Merge mode is read from research_config key
# ``combiner_horizon_bucket_mass_mode`` (numeric enum):
#   0 = signal_count (default): mass ∝ number of signals in bucket
#   1 = equal: equal mass per non-empty bucket
#   2 = n_eff: mass ∝ bucket effective breadth
_HORIZON_BUCKET_MASS_MODE_KEY = "combiner_horizon_bucket_mass_mode"
_HORIZON_BUCKET_MASS_MODE_DEFAULT = 0

# ── IC attribution tunables ─────────────────────────────────────
# Rolling window for computing per-signal Information Coefficient
# (Pearson corr between score_at_entry and forward_return).
_IC_WINDOW_DAYS = 60
# Minimum observations before IC / t-stat are trusted.  Below this we
# still log the IC but don't count it toward retirement streaks.
# DEFAULT for daily-cadence signals; sub-daily signals need more rows
# because their bars are more autocorrelated -- see _min_obs_for(sig).
_IC_MIN_OBS = 30
# Per-resolution minimum observations.  Sub-daily bars are tightly
# autocorrelated so |IC| stabilises only after many rows -- but we
# also want the IR gate to *open* during warmup rather than waiting
# weeks. The IR gate itself is t-stat / sample-size aware (it down-
# weights short-history signals), so these min_obs values are the
# WARMUP floor: "enough rows that IC isn't pure noise".  As the
# system collects more data the gate naturally tightens via the IR
# confidence math without us having to raise these thresholds.
_MIN_OBS_BY_RES = {
    "1min":  60,   # ~1h of valid intraday rounds
    "5min":  40,   # ~3-4h of 5-min bars
    "15min": 25,
    "1h":    20,   # ~3 trading days
    "D":     12,   # ~2.5 trading weeks
}


def _min_obs_for(sig) -> int:
    """Cadence-aware IC minimum-observation threshold for a signal.

    A None signal (e.g. retired/renamed and missing from the registry)
    falls back to the conservative ``_IC_MIN_OBS`` default rather than
    the looser daily-bar warmup floor.
    """
    if sig is None:
        return _IC_MIN_OBS
    res = getattr(sig, "return_resolution", "D")
    return _MIN_OBS_BY_RES.get(res, _IC_MIN_OBS)
# A signal whose |IC| falls below this threshold (on a trusted sample)
# is considered effectively noise. Negative IC is NOT a disqualifier --
# the weight optimizer can flip the sign. Only |IC| ≈ 0 means no info.
_IC_NOISE_THRESHOLD = 0.02
# Consecutive rounds of |IC| < noise threshold (with enough obs) before
# we emit a RETIRE_CANDIDATE warning.
_IC_RETIRE_STREAK = 5
# How many signals to log in top/bottom attribution lists each round.
_IC_ATTRIBUTION_TOP_K = 5

# ── IR gate tunables ────────────────────────────────────────────
# Minimum estimated IR = mean(positive IC) * sqrt(N_eff) before we
# consider the signal stack "live".  Below this the briefing exposes
# ir_gate_open=False and the agent is instructed not to open new
# positions (existing positions are unaffected).  0.05 matches the
# article's institutional baseline IC floor.
_IR_GATE_MIN = 0.05
# Config keys for briefing consumption.
_IR_ESTIMATE_KEY = "estimated_ir"
_IR_GATE_OPEN_KEY = "ir_gate_open"


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


def _group_signals_by_horizon_bucket(
    signal_names: list[str],
) -> dict[tuple[str, int], list[str]]:
    """Group signals by their (return_resolution, return_horizon) bucket."""
    buckets: dict[tuple[str, int], list[str]] = {}
    for name in signal_names:
        buckets.setdefault(_signal_horizon_bucket(name), []).append(name)
    return buckets


def _resolve_horizon_bucket_mass_mode(db_conn) -> str:
    """Resolve bucket-mass mode from research_config.

    Enum values:
      0 -> signal_count
      1 -> equal
      2 -> n_eff
    """
    mode_map = {0: "signal_count", 1: "equal", 2: "n_eff"}
    mode_raw = _HORIZON_BUCKET_MASS_MODE_DEFAULT
    try:
        from memory import get_research_config
        mode_raw = int(
            get_research_config(
                _HORIZON_BUCKET_MASS_MODE_KEY,
                float(_HORIZON_BUCKET_MASS_MODE_DEFAULT),
            )
        )
    except Exception:
        pass
    return mode_map.get(mode_raw, mode_map[_HORIZON_BUCKET_MASS_MODE_DEFAULT])


def _run_11_steps_stratified(
    db_conn,
    signal_names: list[str],
) -> tuple[dict[str, float], str]:
    """Run the combiner per horizon bucket, then merge bucket weights.

    Returns:
        (weights_dict, status)
        status in {"ok", "insufficient_data"}
    """
    if not signal_names:
        return {}, "insufficient_data"

    buckets = _group_signals_by_horizon_bucket(signal_names)
    if not buckets:
        return {s: 1.0 / len(signal_names) for s in signal_names}, "insufficient_data"

    any_bucket_ok = False
    mode = _resolve_horizon_bucket_mass_mode(db_conn)
    bucket_rows: list[dict[str, Any]] = []

    for bucket in sorted(buckets.keys(), key=lambda b: (b[0], b[1])):
        names = buckets[bucket]
        R_b, _periods_b = _load_return_matrix(db_conn, names)
        M_b = R_b.shape[1] if R_b.ndim == 2 else 0

        if M_b < MIN_SHARED_PERIODS_FOR_COMBINATION:
            w_b = {s: 1.0 / len(names) for s in names}
            status_b = "insufficient"
            try:
                n_eff_b = float(compute_n_eff(db_conn, signal_names=names))
            except Exception:
                n_eff_b = float(len(names))
        else:
            try:
                w_b, n_eff_b = _run_11_steps(R_b, names)
                status_b = "ok"
                any_bucket_ok = True
            except Exception as e:
                logger.warning(
                    "Bucket %s combine failed (%s) — equal weights inside bucket",
                    _bucket_label(bucket), e,
                )
                w_b = {s: 1.0 / len(names) for s in names}
                status_b = "fallback_equal"
                try:
                    n_eff_b = float(compute_n_eff(db_conn, signal_names=names))
                except Exception:
                    n_eff_b = float(len(names))

        bucket_rows.append(
            {
                "bucket": bucket,
                "names": names,
                "weights": w_b,
                "status": status_b,
                "M": M_b,
                "n_eff": max(float(n_eff_b), 0.0),
            }
        )

    # Allocate mass across buckets according to configured mode.
    n_buckets = len(bucket_rows)
    if mode == "equal":
        masses = [1.0 / max(n_buckets, 1)] * n_buckets
    elif mode == "n_eff":
        total_neff = sum(r["n_eff"] for r in bucket_rows)
        if total_neff > 1e-12:
            masses = [r["n_eff"] / total_neff for r in bucket_rows]
        else:
            masses = [1.0 / max(n_buckets, 1)] * n_buckets
    else:  # signal_count
        total_signals = sum(len(r["names"]) for r in bucket_rows)
        masses = [
            len(r["names"]) / max(total_signals, 1)
            for r in bucket_rows
        ]

    merged: dict[str, float] = {s: 0.0 for s in signal_names}
    bucket_log_parts: list[str] = []
    for row, mass in zip(bucket_rows, masses):
        for s, w in row["weights"].items():
            merged[s] += mass * float(w)
        bucket_log_parts.append(
            f"{_bucket_label(row['bucket'])}:{row['status']},n={len(row['names'])},"
            f"M={row['M']},n_eff={row['n_eff']:.2f},mass={mass:.3f}"
        )

    # Numerical hygiene: enforce full allocation (sum|w|=1).
    abs_sum = sum(abs(v) for v in merged.values())
    if abs_sum > 1e-12:
        merged = {k: (v / abs_sum) for k, v in merged.items()}
    else:
        merged = {s: 1.0 / len(signal_names) for s in signal_names}

    logger.info(
        "HORIZON_BUCKET_COMBINE mode=%s %s",
        mode,
        " | ".join(bucket_log_parts),
    )
    status = "ok" if any_bucket_ok else "insufficient_data"
    return merged, status


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
    min_symbols_per_period: int = 3,
) -> dict[str, dict[str, float]]:
    """Per-signal IC via the institutional cross-sectional method.

    For each ``(signal, ts)`` we compute the Pearson correlation of
    ``score_at_entry`` against ``forward_return`` ACROSS SYMBOLS at
    that timestamp.  The signal's IC is the *mean* of those per-period
    correlations.  This matches the procedure described in the article
    (Part 2/3) and avoids the inflation that pooled (symbol \u00d7 ts)
    correlations suffer from when within-symbol time-series autocorrelation
    is mistaken for cross-sectional edge.

    Outputs per signal:
        ic            mean per-period cross-sectional Pearson correlation
        t             standard t = IC / SE,  SE = std(IC_t) / sqrt(n_periods)
        n             number of valid periods that contributed
                      (each period needs >= ``min_symbols_per_period`` symbols
                      with non-degenerate score AND return variance)

    Signals that produce a constant value across symbols at every period
    (e.g. a regime tag) yield zero usable periods and are omitted from
    the output -- their |IC| is genuinely undefined cross-sectionally.

    Returns: ``{signal_name: {\"ic\": float, \"t\": float, \"n\": int}}``
    """
    cutoff = time.time() - window_days * 86400.0
    placeholders = ",".join("?" * len(signal_names))
    cur = db_conn.execute(
        f"""
        SELECT signal_name, ts, symbol, score_at_entry, forward_return
          FROM signal_returns
         WHERE ts >= ? AND signal_name IN ({placeholders})
        """,
        (cutoff, *signal_names),
    )

    # Group: signal -> ts -> [(score, return), ...] across symbols.
    by_sig: dict[str, dict[float, list[tuple[float, float]]]] = {}
    for sig, ts, _sym, s, r in cur.fetchall():
        by_sig.setdefault(sig, {}).setdefault(float(ts), []).append((float(s), float(r)))

    out: dict[str, dict[str, float]] = {}
    for sig, periods in by_sig.items():
        per_period_ic: list[float] = []
        for _ts, pairs in periods.items():
            if len(pairs) < min_symbols_per_period:
                continue
            arr = np.asarray(pairs, dtype=float)
            s, r = arr[:, 0], arr[:, 1]
            if float(s.std()) < 1e-12 or float(r.std()) < 1e-12:
                # Constant-across-symbols at this period contributes
                # no cross-sectional information; skip rather than
                # injecting a zero that biases the mean toward 0.
                continue
            ic_t = float(np.corrcoef(s, r)[0, 1])
            if np.isfinite(ic_t):
                per_period_ic.append(ic_t)

        n = len(per_period_ic)
        if n < 1:
            continue
        ics = np.asarray(per_period_ic, dtype=float)
        ic_mean = float(ics.mean())
        # Standard error of the mean per-period IC.
        if n >= 2:
            se = float(ics.std(ddof=1) / np.sqrt(n))
        else:
            # Single period: no SE estimate possible; report t=0 so the
            # signal contributes its IC but earns no statistical trust.
            se = 0.0
        t_stat = ic_mean / se if se > 1e-12 else 0.0
        out[sig] = {"ic": ic_mean, "t": float(t_stat), "n": n}
    return out


def _log_ic_attribution(ic_stats: dict[str, dict[str, float]]) -> None:
    """Log top-K and bottom-K signals by IC each round.

    Format chosen for easy grep: ``IC_TOP`` / ``IC_BOT`` / ``IC_STAT``.
    """
    trusted = [
        (name, d["ic"], d["t"], d["n"])
        for name, d in ic_stats.items()
        if d["n"] >= _min_obs_for(SIGNAL_REGISTRY.get(name))
    ]
    if not trusted:
        logger.info("IC_STAT no_signals_with_min_obs")
        return

    trusted.sort(key=lambda x: x[1], reverse=True)
    k = _IC_ATTRIBUTION_TOP_K
    top = trusted[:k]
    bot = list(reversed(trusted[-k:]))  # worst first

    ics = np.asarray([x[1] for x in trusted])
    logger.info(
        "IC_STAT mean=%.4f median=%.4f std=%.4f n_signals=%d",
        float(ics.mean()), float(np.median(ics)), float(ics.std()),
        len(trusted),
    )
    for name, ic, t, n in top:
        logger.info("IC_TOP %-32s ic=%+.4f t=%+.2f n=%d", name, ic, t, n)
    for name, ic, t, n in bot:
        logger.info("IC_BOT %-32s ic=%+.4f t=%+.2f n=%d", name, ic, t, n)


def _update_ic_retirement(
    db_conn,
    ic_stats: dict[str, dict[str, float]],
) -> None:
    """Track per-signal streaks of |IC| near zero and warn on retirement.

    Streaks are persisted via ``research_config`` under key
    ``ic_noise_streak:{signal}``.  A signal with |IC| < noise threshold
    and n >= min-obs for ``_IC_RETIRE_STREAK`` consecutive rounds is
    logged as a RETIRE_CANDIDATE (WARNING).  Any informative |IC|
    (positive OR negative -- the weight engine can flip the sign)
    resets the streak.

    Persistence failures are non-fatal.
    """
    try:
        from memory import get_research_config, set_research_config
    except Exception as e:
        logger.debug("IC retirement persistence unavailable: %s", e)
        return

    # Cross-sectional IC is undefined for signals that produce the same
    # value across all symbols at a given timestamp (macro/environment
    # signals).  Skip retirement tracking for them — their |IC| will
    # always be ~0 and we'd flood the log with spurious RETIRE_CANDIDATE
    # warnings.  These signals contribute via their time-series weight
    # in the regime layer, not the per-symbol IC layer.
    try:
        from signals.base import SIGNAL_REGISTRY
        _macro_signals = {
            n for n, s in SIGNAL_REGISTRY.items()
            if getattr(s, "category", "") == "macro"
        }
    except Exception:
        _macro_signals = set()

    for name, d in ic_stats.items():
        sig = SIGNAL_REGISTRY.get(name)
        if d["n"] < _min_obs_for(sig):
            continue
        if name in _macro_signals:
            continue
        key = f"ic_noise_streak:{name}"
        prev = int(get_research_config(key, 0))
        new = prev + 1 if abs(d["ic"]) < _IC_NOISE_THRESHOLD else 0
        if new != prev:
            try:
                set_research_config(
                    key,
                    float(new),
                    reason=f"combiner: |ic|={abs(d['ic']):.4f} n={d['n']}",
                )
            except Exception:
                pass
        if new >= _IC_RETIRE_STREAK and (new == _IC_RETIRE_STREAK or new % 5 == 0):
            logger.warning(
                "RETIRE_CANDIDATE %s -- |IC| < %.3f for %d consecutive rounds "
                "(ic=%+.4f t=%+.2f n=%d)",
                name, _IC_NOISE_THRESHOLD, new, d["ic"], d["t"], d["n"],
            )


def _compute_and_log_ic_attribution(
    db_conn,
    signal_names: list[str],
) -> dict[str, dict[str, float]]:
    """One-shot: compute IC, log top/bottom, update retirement streaks.

    Returns the full ic_stats dict so callers can reuse it for IR
    estimation without a second DB scan.
    """
    ic_stats = _compute_signal_ic(db_conn, signal_names)
    if not ic_stats:
        return {}
    _log_ic_attribution(ic_stats)
    _update_ic_retirement(db_conn, ic_stats)
    return ic_stats


def _safe_update_per_symbol_ic(db_conn, signal_names: list[str]) -> None:
    """Refresh per-(signal, symbol, horizon) IC; swallow all errors.

    Per-symbol IC is purely measured here — the combiner does NOT yet
    apply a per-symbol modifier to weights.  The cognitive layer reads
    the resulting ``signal_symbol_ic`` rows once it's wired in.  Keeping
    the call here means per-symbol IC stays in lockstep with the global
    IC's window/cadence and gets refreshed on every combiner round.
    """
    try:
        from signals.per_symbol_ic import update_per_symbol_ic
        update_per_symbol_ic(db_conn, signal_names, window_days=_IC_WINDOW_DAYS)
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("per-symbol IC update failed: %s", e)


def _apply_ic_retirement_mask(
    db_conn,
    weights: dict[str, float],
) -> dict[str, float]:
    """Zero any signal whose ic_neg_streak has reached the retirement
    threshold, then re-normalize so sum|w| = 1.

    Retirement flags are read from ``research_config`` (written by
    ``_update_ic_retirement``).  If no signals are retired the input is
    returned unchanged.  Persistence failures are non-fatal.
    """
    try:
        from memory import get_research_config
    except Exception:
        return weights

    retired: list[str] = []
    for name in weights:
        try:
            streak = int(get_research_config(f"ic_noise_streak:{name}", 0))
        except Exception:
            streak = 0
        if streak >= _IC_RETIRE_STREAK:
            retired.append(name)

    if not retired:
        return weights

    logger.info(
        "IC_RETIRE zeroing %d signal(s) with ic_neg_streak >= %d: %s",
        len(retired), _IC_RETIRE_STREAK, ", ".join(sorted(retired)),
    )
    new_w = dict(weights)
    for name in retired:
        new_w[name] = 0.0

    abs_sum = sum(abs(v) for v in new_w.values())
    if abs_sum > 1e-12:
        new_w = {k: v / abs_sum for k, v in new_w.items()}
    return new_w


def _signal_horizon_bucket(signal_name: str) -> tuple[str, int]:
    """Return the (resolution, horizon) bucket for a signal.

    Missing registry entries fall back to the base Signal defaults to keep
    diagnostics stable during rename / retire transitions.
    """
    sig = SIGNAL_REGISTRY.get(signal_name)
    if sig is None:
        return ("D", 5)
    res = str(getattr(sig, "return_resolution", "D"))
    try:
        horizon = int(getattr(sig, "return_horizon", 5))
    except Exception:
        horizon = 5
    return (res, horizon)


def _bucket_label(bucket: tuple[str, int]) -> str:
    res, horizon = bucket
    return f"{res}/h={horizon}"


def _publish_ir_snapshot(
    db_conn,
    ic_stats: dict[str, dict[str, float]],
    n_eff: float,
    signal_names: list[str] | None = None,
) -> tuple[float, bool]:
    """Compute estimated IR and publish gate state to research_config.

    Backward-compatible behavior (``signal_names is None``): estimate a
    single IR over the entire trusted set:

        estimated_ir = mean(|IC|) * sqrt(N_eff)

    Stratified behavior (``signal_names`` provided): first group signals by
    ``(return_resolution, return_horizon)`` and compute an IR per bucket,
    then aggregate bucket IRs by trusted-signal count:

        ir_bucket = mean(|IC|_bucket) * sqrt(N_eff_bucket)
        estimated_ir = weighted_mean(ir_bucket, weight=n_trusted_bucket)

    This prevents a short-horizon stack (e.g. 1h/h=4) from being benchmarked
    directly against long-horizon stacks (e.g. D/h=10) when forming the
    confidence snapshot used by briefing / gate logic.

    Stores three numeric keys for briefing consumption:
      - estimated_ir
      - ir_gate_open (1.0 / 0.0)
      - ir_snapshot_ts

    Returns (estimated_ir, gate_open).
    """
    try:
        from memory import set_research_config
    except Exception:
        return 0.0, True  # fail-open so the combiner still works

    trusted = [
        (name, d) for name, d in ic_stats.items()
        if d["n"] >= _min_obs_for(SIGNAL_REGISTRY.get(name))
        and abs(d["ic"]) >= _IC_NOISE_THRESHOLD
    ]

    if signal_names is None:
        trusted_abs_ic = [abs(d["ic"]) for _name, d in trusted]
        mean_ic = float(np.mean(trusted_abs_ic)) if trusted_abs_ic else 0.0
        ir = mean_ic * float(np.sqrt(max(n_eff, 0.0)))
        bucket_parts: list[str] = []
    else:
        # Stratify by per-signal return horizon bucket.
        by_bucket: dict[tuple[str, int], list[str]] = {}
        for name, _d in trusted:
            by_bucket.setdefault(_signal_horizon_bucket(name), []).append(name)

        bucket_irs: list[tuple[tuple[str, int], float, int, float, float]] = []
        # tuple = (bucket, ir_bucket, n_trusted, mean_abs_ic, n_eff_bucket)
        for bucket, names in by_bucket.items():
            abs_ics = [abs(ic_stats[n]["ic"]) for n in names]
            mean_abs_ic = float(np.mean(abs_ics)) if abs_ics else 0.0
            try:
                n_eff_bucket = float(compute_n_eff(db_conn, signal_names=names))
            except Exception:
                n_eff_bucket = float(len(names))
            ir_bucket = mean_abs_ic * float(np.sqrt(max(n_eff_bucket, 0.0)))
            bucket_irs.append((bucket, ir_bucket, len(names), mean_abs_ic, n_eff_bucket))

        total_weight = sum(n_trusted for _b, _ir_b, n_trusted, _m, _n in bucket_irs)
        if total_weight > 0:
            ir = float(
                sum(ir_b * n_trusted for _b, ir_b, n_trusted, _m, _n in bucket_irs)
                / total_weight
            )
        else:
            ir = 0.0
        mean_ic = (
            float(np.mean([abs(d["ic"]) for _name, d in trusted]))
            if trusted else 0.0
        )
        bucket_parts = [
            (
                f"{_bucket_label(bucket)}:ir={ir_b:.4f},mean_abs_ic={mean_abs_ic:.4f},"
                f"n_eff={n_eff_bucket:.2f},n={n_trusted}"
            )
            for bucket, ir_b, n_trusted, mean_abs_ic, n_eff_bucket in sorted(
                bucket_irs,
                key=lambda x: (x[0][0], x[0][1]),
            )
        ]

    gate_open = ir >= _IR_GATE_MIN

    try:
        set_research_config(_IR_ESTIMATE_KEY, ir,
                            reason=f"mean_abs_ic={mean_ic:.4f} n_eff={n_eff:.2f}")
        set_research_config(_IR_GATE_OPEN_KEY, 1.0 if gate_open else 0.0,
                            reason=f"ir={ir:.4f} min={_IR_GATE_MIN}")
        set_research_config("ir_snapshot_ts", float(time.time()),
                            reason="combiner round complete")
    except Exception as e:
        logger.debug("IR snapshot persistence failed: %s", e)

    logger.info(
        "IR_ESTIMATE ir=%.4f mean_abs_ic=%.4f n_eff=%.2f gate_open=%s",
        ir, mean_ic, n_eff, gate_open,
    )
    if bucket_parts:
        logger.info("IR_BUCKETS %s", " | ".join(bucket_parts))
    return ir, gate_open


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
