"""PR20 - Combiner edge-case characterization.

Complements the existing combiner tests in ``tests/test_signals.py``
(which cover identity / perfect-corr / singular-matrix / lookahead via
``_run_11_steps`` on hand-built R matrices). This file pins:

  * ``compute_n_eff`` DB path: degenerate cases (N=0, N=1, no rows in
    ``signal_returns``, all-zero R), and the "independent signals
    yields N_eff close to N" property end-to-end through the loader.
  * ``_run_11_steps`` invariants:
      - sum(|w|) == 1 in normal cases (Step 11 normalization)
      - all-zero R yields equal weights (no information)
      - low-SNR signals are forced to weight 0 (improvement #5)
"""

from __future__ import annotations

import numpy as np
import pytest

# _isolated_db (autouse) and db fixtures are provided by tests/conftest.py.


def _seed_signal_returns(db, signal_names, n_periods=20, *, correlation=0.0,
                         seed=42):
    """Insert synthetic signal_returns rows. correlation=0 -> independent;
    correlation=1 -> all signals identical."""
    rng = np.random.default_rng(seed)
    ts_base = 1_700_000_000.0
    base_series = rng.standard_normal(n_periods)

    for sig in signal_names:
        if correlation >= 1.0:
            series = base_series.copy()
        elif correlation <= 0.0:
            series = rng.standard_normal(n_periods)
        else:
            noise = rng.standard_normal(n_periods)
            series = (correlation * base_series
                      + np.sqrt(1 - correlation ** 2) * noise)
        for j in range(n_periods):
            ts = ts_base + j * 300
            r_val = float(series[j])
            db.execute(
                "INSERT INTO signal_returns (signal_name, symbol, ts, "
                "score_at_entry, forward_return, r_value, horizon_bars) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sig, "AAPL", ts, 0.5, r_val * 2, r_val, 4),
            )
    db.commit()


# ── compute_n_eff DB path ─────────────────────────────────────


class TestComputeNEffDB:
    def test_no_rows_returns_n(self, db):
        from signals.combiner import compute_n_eff
        # Empty DB, fresh signals -- no rows in signal_returns.
        # The loader returns an N x 0 matrix; M=0 < 3 -> the function
        # short-circuits to float(N).
        out = compute_n_eff(db, signal_names=["a", "b", "c"])
        assert out == 3.0

    def test_single_signal_returns_one(self, db):
        from signals.combiner import compute_n_eff
        _seed_signal_returns(db, ["only"], n_periods=20)
        out = compute_n_eff(db, signal_names=["only"])
        # With N=1 the function returns float(N) = 1.0 before the
        # eigvalsh path.
        assert out == 1.0

    def test_independent_signals_close_to_n(self, db):
        from signals.combiner import compute_n_eff
        names = [f"ind_{i}" for i in range(5)]
        _seed_signal_returns(db, names, n_periods=40, correlation=0.0)
        out = compute_n_eff(db, signal_names=names)
        # Independent N=5 -> N_eff should be at least 3 with this many
        # periods; existing test_signals.py uses the same threshold.
        assert out > 3.0

    def test_perfectly_correlated_collapses_to_one(self, db):
        from signals.combiner import compute_n_eff
        names = [f"corr_{i}" for i in range(5)]
        _seed_signal_returns(db, names, n_periods=40, correlation=1.0)
        out = compute_n_eff(db, signal_names=names)
        # Identical series -> N_eff ~ 1.
        assert out < 1.5


# ── _run_11_steps invariants and edge cases ───────────────────


class TestRun11StepsInvariants:
    def test_weight_sum_abs_equals_one_in_normal_case(self):
        from signals.combiner import _run_11_steps
        rng = np.random.default_rng(42)
        # Use enough periods to bypass M<=2 / Lambda<=1 short-circuits
        # AND avoid the equal-weights N_eff fallback in _run_11_steps
        # by giving each signal a distinct mean (so SNR isn't zero).
        N, M = 5, 60
        R = rng.standard_normal((N, M))
        # Add per-signal drift so each has nonzero mean -> nonzero raw_w.
        R += np.array([0.1, -0.1, 0.2, -0.2, 0.05])[:, None]
        names = [f"s{i}" for i in range(N)]
        w_dict, _ = _run_11_steps(R, names)

        # Step 11 normalizes so sum(|w|) == 1, OR equal weights when
        # abs_sum was below 1e-12.  Either way the absolute sum must be
        # very close to 1.
        s = sum(abs(v) for v in w_dict.values())
        assert s == pytest.approx(1.0, abs=1e-9)

    def test_all_zero_R_yields_equal_weights(self):
        from signals.combiner import _run_11_steps
        N, M = 4, 50
        R = np.zeros((N, M))
        names = [f"s{i}" for i in range(N)]
        w_dict, _ = _run_11_steps(R, names)
        # Every signal is zero-variance -> raw_w masked to 0 -> Step 11
        # falls back to equal weights.
        expected = 1.0 / N
        for w in w_dict.values():
            assert w == pytest.approx(expected, abs=1e-9)

    def test_low_snr_signal_zeroed(self):
        """A signal with high variance but near-zero mean (SNR < floor)
        is masked to weight 0 by improvement #5."""
        from signals.combiner import _run_11_steps, _SNR_FLOOR
        N, M = 3, 80
        rng = np.random.default_rng(7)
        R = rng.standard_normal((N, M))
        # Force signal 0 to have a strong directional drift (high SNR).
        R[0] += 1.0
        # Force signal 1 to have zero mean (low SNR -> below floor).
        R[1] -= R[1].mean()
        # Signal 2 left as random noise.

        names = [f"s{i}" for i in range(N)]
        w_dict, _ = _run_11_steps(R, names)

        # The SNR floor is _SNR_FLOOR (default 0.05). Signal 1 has
        # mean exactly 0 by construction -> SNR = 0 -> below floor ->
        # weight masked to 0. Pin it.
        assert w_dict[names[1]] == 0.0
        # Sanity: the high-SNR signal should keep nonzero weight (after
        # normalization), unless the equal-weights fallback fires.
        # In either case the masked one is 0.
        assert _SNR_FLOOR > 0  # documents the test's premise

    def test_single_signal_returns_equal_weight_one(self):
        """N=1 still produces a valid weights dict."""
        from signals.combiner import _run_11_steps
        R = np.array([[0.1, 0.2, -0.1, 0.05, 0.15] * 6])  # N=1, M=30
        w_dict, _ = _run_11_steps(R, ["only"])
        assert list(w_dict.keys()) == ["only"]
        # With N=1 the equal-weights path (1/N = 1.0) and the
        # normalization path (|w|/|w| = 1.0) both yield exactly 1.0.
        assert abs(w_dict["only"]) == pytest.approx(1.0, abs=1e-9)
