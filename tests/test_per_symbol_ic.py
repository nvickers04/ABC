"""Tests for signals.per_symbol_ic — per-(signal, symbol, horizon) IC."""

from __future__ import annotations

import time

import numpy as np
import pytest


# DB isolation: uses ``tests/conftest.py`` autouse ``_isolated_db``.


@pytest.fixture
def db():
    from memory import get_db
    return get_db()


# ── Helpers ──────────────────────────────────────────────────────

def _seed_pairs(db, signal_name, symbol, pairs, *, horizon=1, ts_base=None):
    """Insert (score, forward_return) pairs into signal_returns."""
    ts_base = ts_base if ts_base is not None else time.time() - 3600.0
    for j, (s, r) in enumerate(pairs):
        ts = ts_base + j * 60.0
        db.execute(
            "INSERT INTO signal_returns "
            "(signal_name, symbol, ts, score_at_entry, forward_return, r_value, horizon_bars) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (signal_name, symbol, ts, float(s), float(r), float(s) * float(r), int(horizon)),
        )
    db.commit()


def _make_correlated(n, target_corr, seed=0):
    """Generate (s, r) arrays with sample Pearson correlation ≈ target_corr."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    # r = a*s + b*noise normalized to target correlation
    a = float(target_corr)
    b = float(np.sqrt(max(0.0, 1.0 - target_corr ** 2)))
    r = a * s + b * noise
    return list(zip(s.tolist(), r.tolist()))


# ── Schema tests ────────────────────────────────────────────────

def test_signal_symbol_ic_table_exists(db):
    cur = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_symbol_ic'"
    )
    assert cur.fetchone() is not None


def test_signal_symbol_ic_pk_columns(db):
    cur = db.execute("PRAGMA table_info(signal_symbol_ic)")
    cols = {row[1]: row for row in cur.fetchall()}
    for required in ("signal_name", "symbol", "horizon_bars", "ic", "t_stat",
                     "n_obs", "ic_neg_streak", "last_updated_ts"):
        assert required in cols


# ── compute_per_symbol_ic ───────────────────────────────────────

def test_compute_returns_empty_when_no_signal_names(db):
    from signals.per_symbol_ic import compute_per_symbol_ic
    assert compute_per_symbol_ic(db, []) == {}


def test_compute_returns_empty_when_no_rows(db):
    from signals.per_symbol_ic import compute_per_symbol_ic
    assert compute_per_symbol_ic(db, ["momentum"]) == {}


def test_compute_skips_buckets_with_too_few_obs(db):
    from signals.per_symbol_ic import compute_per_symbol_ic
    _seed_pairs(db, "momentum", "AAPL", [(0.1, 0.01), (0.2, 0.02)])  # n=2
    out = compute_per_symbol_ic(db, ["momentum"])
    assert out == {}


def test_compute_correlation_matches_numpy(db):
    from signals.per_symbol_ic import compute_per_symbol_ic
    pairs = _make_correlated(50, target_corr=0.6, seed=7)
    _seed_pairs(db, "momentum", "AAPL", pairs)
    out = compute_per_symbol_ic(db, ["momentum"])
    assert ("momentum", "AAPL", 1) in out
    s, r = zip(*pairs)
    expected = float(np.corrcoef(s, r)[0, 1])
    got = out[("momentum", "AAPL", 1)]["ic"]
    assert got == pytest.approx(expected, abs=1e-9)
    assert out[("momentum", "AAPL", 1)]["n"] == 50


def test_compute_separates_symbols(db):
    """Same signal, different symbols → distinct buckets with distinct ICs."""
    from signals.per_symbol_ic import compute_per_symbol_ic
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.5, seed=1))
    _seed_pairs(db, "momentum", "NVDA", _make_correlated(40, -0.3, seed=2))
    out = compute_per_symbol_ic(db, ["momentum"])
    assert ("momentum", "AAPL", 1) in out
    assert ("momentum", "NVDA", 1) in out
    assert out[("momentum", "AAPL", 1)]["ic"] > 0.2
    assert out[("momentum", "NVDA", 1)]["ic"] < -0.05


def test_compute_separates_horizons(db):
    from signals.per_symbol_ic import compute_per_symbol_ic
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(30, 0.4, seed=3), horizon=1)
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(30, -0.2, seed=4), horizon=12)
    out = compute_per_symbol_ic(db, ["momentum"])
    assert ("momentum", "AAPL", 1) in out
    assert ("momentum", "AAPL", 12) in out


def test_compute_zero_variance_returns_zero_ic(db):
    from signals.per_symbol_ic import compute_per_symbol_ic
    # All scores identical → s.std() == 0
    _seed_pairs(db, "momentum", "AAPL", [(0.5, float(i) * 0.01) for i in range(20)])
    out = compute_per_symbol_ic(db, ["momentum"])
    assert out[("momentum", "AAPL", 1)]["ic"] == 0.0
    assert out[("momentum", "AAPL", 1)]["t"] == 0.0


def test_compute_window_filter_excludes_old_rows(db):
    from signals.per_symbol_ic import compute_per_symbol_ic
    # Old rows (10 days ago) with n=30
    old_ts = time.time() - 10 * 86400.0
    _seed_pairs(db, "momentum", "AAPL",
                _make_correlated(30, 0.5, seed=5), ts_base=old_ts)
    # Window of 5 days excludes them all
    out = compute_per_symbol_ic(db, ["momentum"], window_days=5)
    assert out == {}


# ── update_per_symbol_ic ────────────────────────────────────────

def test_update_writes_rows(db):
    from signals.per_symbol_ic import update_per_symbol_ic, get_per_symbol_ic
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.5, seed=11))
    n = update_per_symbol_ic(db, ["momentum"])
    assert n == 1
    row = get_per_symbol_ic(db, "momentum", "AAPL")
    assert row is not None
    assert row["n_obs"] == 40
    assert row["ic"] != 0.0


def test_update_is_idempotent_upsert(db):
    from signals.per_symbol_ic import update_per_symbol_ic
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.5, seed=12))
    update_per_symbol_ic(db, ["momentum"])
    update_per_symbol_ic(db, ["momentum"])
    cur = db.execute(
        "SELECT COUNT(*) FROM signal_symbol_ic "
        "WHERE signal_name='momentum' AND symbol='AAPL'"
    )
    assert cur.fetchone()[0] == 1


def test_update_tracks_neg_streak_when_noisy(db, monkeypatch):
    """|IC| < 0.02 with enough obs increments streak; non-noise resets it."""
    from signals import per_symbol_ic as psm
    # Force min-obs threshold to 30 regardless of registry
    monkeypatch.setattr(psm, "_min_obs_for_signal", lambda _name: 30)

    # Round 1: deterministic |IC|=0 (constant score → zero variance branch)
    # n=40 ≥ 30 min-obs → streak goes 0 → 1
    constant_score_pairs = [(0.5, float(i % 7) - 3.0) for i in range(40)]
    _seed_pairs(db, "momentum", "AAPL", constant_score_pairs)
    psm.update_per_symbol_ic(db, ["momentum"])
    cur = db.execute(
        "SELECT ic, ic_neg_streak FROM signal_symbol_ic "
        "WHERE signal_name='momentum' AND symbol='AAPL'"
    )
    ic1, streak1 = cur.fetchone()
    assert ic1 == 0.0
    assert int(streak1) == 1

    # Round 2: still noisy → streak ticks again
    psm.update_per_symbol_ic(db, ["momentum"])
    cur = db.execute(
        "SELECT ic_neg_streak FROM signal_symbol_ic WHERE signal_name='momentum' AND symbol='AAPL'"
    )
    assert int(cur.fetchone()[0]) == 2

    # Round 3: replace data with strong correlation → streak resets
    db.execute("DELETE FROM signal_returns")
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.6, seed=22))
    psm.update_per_symbol_ic(db, ["momentum"])
    cur = db.execute(
        "SELECT ic_neg_streak FROM signal_symbol_ic WHERE signal_name='momentum' AND symbol='AAPL'"
    )
    assert int(cur.fetchone()[0]) == 0


def test_update_does_not_streak_when_below_min_obs(db, monkeypatch):
    from signals import per_symbol_ic as psm
    # Min-obs floor of 100 → 40 rows can't drive a streak even with ic=0
    monkeypatch.setattr(psm, "_min_obs_for_signal", lambda _name: 100)
    constant_score_pairs = [(0.5, float(i % 7) - 3.0) for i in range(40)]
    _seed_pairs(db, "momentum", "AAPL", constant_score_pairs)
    psm.update_per_symbol_ic(db, ["momentum"])
    cur = db.execute(
        "SELECT ic_neg_streak FROM signal_symbol_ic WHERE signal_name='momentum' AND symbol='AAPL'"
    )
    assert int(cur.fetchone()[0]) == 0


# ── get_per_symbol_ic ────────────────────────────────────────────

def test_get_returns_none_when_missing(db):
    from signals.per_symbol_ic import get_per_symbol_ic
    assert get_per_symbol_ic(db, "momentum", "AAPL") is None


def test_get_picks_highest_n_when_horizon_unspecified(db):
    from signals.per_symbol_ic import update_per_symbol_ic, get_per_symbol_ic
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(20, 0.5, seed=31), horizon=1)
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(50, 0.3, seed=32), horizon=12)
    update_per_symbol_ic(db, ["momentum"])
    row = get_per_symbol_ic(db, "momentum", "AAPL")
    assert row["horizon_bars"] == 12
    assert row["n_obs"] == 50


def test_get_with_horizon_filter(db):
    from signals.per_symbol_ic import update_per_symbol_ic, get_per_symbol_ic
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(30, 0.5, seed=41), horizon=1)
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.2, seed=42), horizon=12)
    update_per_symbol_ic(db, ["momentum"])
    row = get_per_symbol_ic(db, "momentum", "AAPL", horizon_bars=1)
    assert row["horizon_bars"] == 1
    assert row["n_obs"] == 30


# ── per_symbol_modifier ─────────────────────────────────────────

def test_modifier_returns_default_when_no_row(db):
    from signals.per_symbol_ic import per_symbol_modifier
    assert per_symbol_modifier(db, "momentum", "AAPL", global_ic=0.05) == 1.0


def test_modifier_returns_default_when_below_min_obs(db):
    from signals.per_symbol_ic import update_per_symbol_ic, per_symbol_modifier
    # 10 rows < _PER_SYMBOL_MIN_OBS=20
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(10, 0.5, seed=51))
    update_per_symbol_ic(db, ["momentum"])
    assert per_symbol_modifier(db, "momentum", "AAPL", global_ic=0.05) == 1.0


def test_modifier_returns_default_when_global_ic_unknown_or_tiny(db):
    from signals.per_symbol_ic import update_per_symbol_ic, per_symbol_modifier
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.5, seed=52))
    update_per_symbol_ic(db, ["momentum"])
    assert per_symbol_modifier(db, "momentum", "AAPL", global_ic=None) == 1.0
    assert per_symbol_modifier(db, "momentum", "AAPL", global_ic=0.001) == 1.0


def test_modifier_amplifies_when_per_symbol_ic_exceeds_global(db):
    from signals.per_symbol_ic import update_per_symbol_ic, per_symbol_modifier
    # Per-symbol IC ~0.5; pretend global IC = 0.1 → ratio = 5 → clipped to 2.0
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.5, seed=53))
    update_per_symbol_ic(db, ["momentum"])
    mod = per_symbol_modifier(db, "momentum", "AAPL", global_ic=0.1)
    assert mod == pytest.approx(2.0)


def test_modifier_dampens_when_per_symbol_ic_weaker_than_global(db):
    from signals.per_symbol_ic import update_per_symbol_ic, per_symbol_modifier
    # Per-symbol IC ~0.05; pretend global IC = 0.4 → ratio = 0.125 → clipped 0.25
    _seed_pairs(db, "momentum", "AAPL", _make_correlated(40, 0.05, seed=54))
    update_per_symbol_ic(db, ["momentum"])
    mod = per_symbol_modifier(db, "momentum", "AAPL", global_ic=0.4)
    assert mod == pytest.approx(0.25)


def test_modifier_swallows_db_errors(db):
    """Broken connection must NOT raise — modifier returns default."""
    from signals.per_symbol_ic import per_symbol_modifier

    class BrokenConn:
        def execute(self, *a, **k):
            raise RuntimeError("nope")

    assert per_symbol_modifier(BrokenConn(), "momentum", "AAPL", global_ic=0.1) == 1.0


# ── Combiner integration smoke test ──────────────────────────────

def test_combiner_round_populates_per_symbol_ic(db):
    """Running combine_signals on seeded data should populate signal_symbol_ic."""
    from signals.combiner import combine_signals

    sigs = ["momentum", "mean_reversion"]
    rng = np.random.default_rng(123)
    for sig in sigs:
        for sym in ("AAPL", "NVDA"):
            pairs = _make_correlated(40, 0.3, seed=int(rng.integers(0, 10_000)))
            _seed_pairs(db, sig, sym, pairs)

    combine_signals(db, signal_names=sigs)

    cur = db.execute("SELECT COUNT(*) FROM signal_symbol_ic")
    count = cur.fetchone()[0]
    # 2 signals × 2 symbols × 1 horizon = 4 buckets
    assert count == 4
