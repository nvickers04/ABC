"""PR16 - Characterization tests for ``signals.scorer._compute_forward_returns``.

Locks in the R(i, s) writer's contract:

  * **Maturity** - a score with at least ``horizon_bars`` of subsequent
    candle bars yields a row in ``signal_returns`` keyed by the entry-bar
    timestamp (NOT the original score timestamp).
  * **Pending** - a score whose horizon has not elapsed is left in
    ``signal_scores`` for a future round (no row written, score retained).
  * **Universe-drift prune** - any score whose symbol is missing from EVERY
    resolution's candle map is deleted from ``signal_scores`` at the top
    of the function.
  * **TTL purge** - scores older than 30 days are deleted unconditionally.
  * **R(i, s) value** - ``r_value = score * (exit_close / entry_close - 1)``,
    where entry is the most recent bar with ``ts <= score_ts`` and exit is
    ``entry + horizon`` bars later.
  * **Per-resolution dispatch** - signals are bucketed by
    ``return_resolution`` and resolved against the matching candle map.

These tests register one-off ``Signal`` subclasses, clean up the global
``SIGNAL_REGISTRY`` after each test, and stub the candle objects with
``close`` + ``timestamps`` lists (the only attributes the function reads).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest


# ── Fixtures ───────────────────────────────────────────────────


# _isolated_db (autouse) and db fixtures come from tests/conftest.py.


@pytest.fixture
def clean_registry():
    """Snapshot SIGNAL_REGISTRY around each test to avoid cross-test leaks."""
    from signals.base import SIGNAL_REGISTRY
    saved = dict(SIGNAL_REGISTRY)
    SIGNAL_REGISTRY.clear()
    yield SIGNAL_REGISTRY
    SIGNAL_REGISTRY.clear()
    SIGNAL_REGISTRY.update(saved)


def _register_signal(reg, *, name: str, resolution: str, horizon: int,
                     category: str = "price"):
    """Build a one-off Signal subclass, register, and return the instance."""
    from signals.base import Signal, SignalResult

    class _Sig(Signal):
        def compute(self, symbol, data):
            return SignalResult(score=0.0, confidence=0.0, components={})

    _Sig.name = name
    _Sig.category = category
    _Sig.data_source = "candles"
    _Sig.return_resolution = resolution
    _Sig.return_horizon = horizon

    inst = _Sig()
    reg[name] = inst
    return inst


def _make_candles(closes, ts_list):
    """Minimal candles object — only ``close`` and ``timestamps`` are read."""
    return SimpleNamespace(close=list(closes), timestamps=list(ts_list))


def _insert_score(db, signal, symbol, ts, score=0.5, conf=0.8):
    db.execute(
        "INSERT OR REPLACE INTO signal_scores "
        "(signal_name, symbol, ts, score, confidence, components_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (signal, symbol, ts, score, conf, "{}"),
    )
    db.commit()


# ── Tests ─────────────────────────────────────────────────────


class TestForwardReturnsMaturity:
    def test_matured_score_writes_signal_returns_row(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        # Hourly signal, 4-bar horizon.
        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4, category="price")

        # 10 hourly bars at $100, $101, $102, ..., $109; bar timestamps 1..10h.
        ts_bars = [3600.0 * i for i in range(1, 11)]
        closes = [100.0 + i for i in range(10)]
        candles = _make_candles(closes, ts_bars)

        # Score recorded one second after bar 1 -> entry bar = bar index 0
        # (close $100). Exit bar = 0 + 4 = bar index 4 (close $104).
        # Forward return = 4 / 100 = 0.04. r_value = 0.5 * 0.04 = 0.02.
        score_ts = ts_bars[0] + 1.0
        _insert_score(db, "px_h4", "AAPL", score_ts, score=0.5)

        candles_by_res = {"1h": {"AAPL": candles}}
        # current_ts must be > score_ts (the SQL filters ts < current_ts).
        _compute_forward_returns(db, dp=None,
                                 candles_by_res=candles_by_res,
                                 current_ts=ts_bars[-1] + 1.0)

        rows = db.execute(
            "SELECT signal_name, symbol, ts, score_at_entry, forward_return, "
            "r_value, horizon_bars FROM signal_returns"
        ).fetchall()
        assert len(rows) == 1
        r = rows[0]
        assert r["signal_name"] == "px_h4"
        assert r["symbol"] == "AAPL"
        # The signal_returns row is keyed by the ENTRY-BAR ts, not the
        # original score ts -- this is critical for the bar-grid dedup
        # described in the function's docstring.
        assert r["ts"] == ts_bars[0]
        assert r["score_at_entry"] == 0.5
        assert r["forward_return"] == pytest.approx(0.04, abs=1e-9)
        assert r["r_value"] == pytest.approx(0.02, abs=1e-9)
        assert r["horizon_bars"] == 4

    def test_pending_score_not_written_and_retained(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        # Only 3 bars of history -> horizon=4 not yet matured.
        ts_bars = [3600.0, 7200.0, 10800.0]
        candles = _make_candles([100.0, 101.0, 102.0], ts_bars)

        score_ts = ts_bars[0] + 1.0
        _insert_score(db, "px_h4", "AAPL", score_ts, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        # Nothing written.
        n_returns = db.execute(
            "SELECT COUNT(*) AS n FROM signal_returns"
        ).fetchone()["n"]
        assert n_returns == 0
        # Score still pending in signal_scores.
        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores "
            "WHERE signal_name = 'px_h4' AND symbol = 'AAPL'"
        ).fetchone()["n"]
        assert n_scores == 1


class TestForwardReturnsRValueSign:
    def test_negative_score_positive_return_yields_negative_r(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="bear", resolution="1h",
                         horizon=2)

        # Price rises 10% over 2 bars; bearish signal of -0.5 ->
        # r_value = -0.5 * 0.10 = -0.05.
        ts_bars = [3600.0, 7200.0, 10800.0, 14400.0]
        candles = _make_candles([100.0, 105.0, 110.0, 115.0], ts_bars)

        _insert_score(db, "bear", "AAPL", ts_bars[0] + 1.0, score=-0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        row = db.execute("SELECT * FROM signal_returns").fetchone()
        assert row is not None
        assert row["forward_return"] == pytest.approx(0.10, abs=1e-9)
        assert row["r_value"] == pytest.approx(-0.05, abs=1e-9)


class TestForwardReturnsPruning:
    def test_universe_drift_symbol_pruned(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        ts_bars = [3600.0 * i for i in range(1, 11)]
        candles = _make_candles([100.0 + i for i in range(10)], ts_bars)

        # AAPL is in the candle map; XYZZY is not anywhere -> should be pruned.
        _insert_score(db, "px_h4", "AAPL", ts_bars[0] + 1.0, score=0.5)
        _insert_score(db, "px_h4", "XYZZY", ts_bars[0] + 1.0, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        remaining = sorted(
            row["symbol"] for row in db.execute(
                "SELECT symbol FROM signal_scores"
            ).fetchall()
        )
        # AAPL row was consumed (matured -> wrote signal_returns row, then
        # deleted from signal_scores so it doesn't dominate the LIMIT
        # queue on subsequent rounds).  XYZZY was deleted as universe
        # drift.
        assert remaining == []

        return_syms = sorted(
            row["symbol"] for row in db.execute(
                "SELECT symbol FROM signal_returns"
            ).fetchall()
        )
        assert return_syms == ["AAPL"]

    def test_ttl_purge_removes_old_scores(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        # Score ts is "now"; TTL cutoff is now - 30 days. Anchor things
        # accordingly so the old row is unambiguously past TTL.
        now = 1_800_000_000.0
        old_ts = now - 31 * 86400  # 31 days old
        recent_ts = now - 60.0     # 1 minute old (well within TTL)

        ts_bars = [now - 3600.0 * (5 - i) for i in range(6)]  # 5 bars before now
        candles = _make_candles([100.0 + i for i in range(6)], ts_bars)

        _insert_score(db, "px_h4", "AAPL", old_ts, score=0.5)
        _insert_score(db, "px_h4", "AAPL", recent_ts, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=now)

        # Old row deleted by TTL purge; recent row may be matured/pending
        # depending on bar alignment, but the OLD ts must be gone.
        ts_left = sorted(
            row["ts"] for row in db.execute(
                "SELECT ts FROM signal_scores"
            ).fetchall()
        )
        assert old_ts not in ts_left


class TestForwardReturnsPerResolution:
    def test_different_resolutions_use_different_candle_maps(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="micro", resolution="5min",
                         horizon=2, category="microstructure")
        _register_signal(clean_registry, name="daily_macro", resolution="D",
                         horizon=1, category="macro")

        # 5-min candles for AAPL: closes step +1 each bar.
        ts_5m = [300.0 * i for i in range(1, 11)]
        candles_5m = _make_candles([50.0 + i for i in range(10)], ts_5m)
        # Daily candles for AAPL: closes step +10 each bar.
        ts_d = [86400.0 * i for i in range(1, 6)]
        candles_d = _make_candles([200.0 + 10 * i for i in range(5)], ts_d)

        # Both signals scored at t=600 (after first 5m bar, after first daily bar).
        _insert_score(db, "micro", "AAPL", 600.0 + 1.0, score=1.0)
        _insert_score(db, "daily_macro", "AAPL",
                      ts_d[0] + 1.0, score=1.0)

        _compute_forward_returns(
            db, dp=None,
            candles_by_res={
                "5min": {"AAPL": candles_5m},
                "D": {"AAPL": candles_d},
            },
            current_ts=max(ts_5m[-1], ts_d[-1]) + 1.0,
        )

        rows = db.execute(
            "SELECT signal_name, ts, score_at_entry, horizon_bars "
            "FROM signal_returns ORDER BY signal_name"
        ).fetchall()
        assert len(rows) == 2
        names = {r["signal_name"] for r in rows}
        assert names == {"micro", "daily_macro"}
        # micro horizon=2, entry bar ts in 5m grid (300s steps).
        micro = next(r for r in rows if r["signal_name"] == "micro")
        assert micro["horizon_bars"] == 2
        assert micro["ts"] in ts_5m
        # daily_macro horizon=1, entry bar ts in D grid (86400s steps).
        daily = next(r for r in rows if r["signal_name"] == "daily_macro")
        assert daily["horizon_bars"] == 1
        assert daily["ts"] in ts_d

    def test_score_with_no_candles_for_its_resolution_skipped(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        # AAPL exists in DAILY map (so it's not a universe-drift prune),
        # but the signal needs HOURLY -> no_candles path.
        ts_d = [86400.0 * i for i in range(1, 4)]
        candles_d = _make_candles([100.0, 110.0, 120.0], ts_d)

        _insert_score(db, "px_h4", "AAPL", ts_d[0] + 1.0, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"D": {"AAPL": candles_d}},
                                 current_ts=ts_d[-1] + 1.0)

        # No row written.
        n = db.execute("SELECT COUNT(*) AS n FROM signal_returns").fetchone()["n"]
        assert n == 0
        # Score still in signal_scores (not purged - AAPL is in some map).
        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores"
        ).fetchone()["n"]
        assert n_scores == 1


class TestForwardReturnsUpsertByEntryBar:
    def test_two_scores_same_bar_collapse_to_one_row(self, db, clean_registry):
        """Multiple intraday scores that resolve to the same entry bar
        must collapse via INSERT OR REPLACE so signal_returns stays
        bar-aligned (one row per (signal, symbol, entry_bar))."""
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        ts_bars = [3600.0 * i for i in range(1, 11)]
        candles = _make_candles([100.0 + i for i in range(10)], ts_bars)

        # Two scores both fall WITHIN the first hourly bar (between
        # ts_bars[0] and ts_bars[1]) -> both resolve to entry bar 0.
        _insert_score(db, "px_h4", "AAPL", ts_bars[0] + 100.0, score=0.3)
        _insert_score(db, "px_h4", "AAPL", ts_bars[0] + 200.0, score=0.8)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        rows = db.execute(
            "SELECT score_at_entry, ts FROM signal_returns "
            "WHERE signal_name = 'px_h4' AND symbol = 'AAPL'"
        ).fetchall()
        assert len(rows) == 1
        # The function processes scores ORDER BY ts ASC and writes via
        # INSERT OR REPLACE keyed by entry-bar ts -- so the LATER score
        # (0.8) wins.
        assert rows[0]["score_at_entry"] == 0.8
        assert rows[0]["ts"] == ts_bars[0]


class TestForwardReturnsMaturedRowsDeleted:
    def test_matured_rows_removed_from_signal_scores(self, db, clean_registry):
        """PR18: matured signal_scores rows are deleted after their R(i,s)
        row is written, so they don't dominate the ORDER BY ts ASC LIMIT
        queue on subsequent rounds. Pending rows remain."""
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        ts_bars = [3600.0 * i for i in range(1, 11)]
        candles = _make_candles([100.0 + i for i in range(10)], ts_bars)

        # Score 1: matured (entry bar 0, exit bar 4 -- both available).
        _insert_score(db, "px_h4", "AAPL", ts_bars[0] + 1.0, score=0.5)
        # Score 2: pending (entry bar 8, exit bar 12 -- exit not yet available).
        _insert_score(db, "px_h4", "AAPL", ts_bars[8] + 1.0, score=0.7)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        scores_left = sorted(
            row["ts"] for row in db.execute(
                "SELECT ts FROM signal_scores"
            ).fetchall()
        )
        # Matured row deleted, pending row retained.
        assert scores_left == [ts_bars[8] + 1.0]
        # And the matured one wrote its R(i,s) row.
        n_returns = db.execute(
            "SELECT COUNT(*) AS n FROM signal_returns"
        ).fetchone()["n"]
        assert n_returns == 1

    def test_idempotent_second_round_no_double_write(self, db, clean_registry):
        """A second call to _compute_forward_returns with no new data must
        be a no-op (no scores left to mature). Before PR18 the same matured
        rows would be re-read every round."""
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        ts_bars = [3600.0 * i for i in range(1, 11)]
        candles = _make_candles([100.0 + i for i in range(10)], ts_bars)
        _insert_score(db, "px_h4", "AAPL", ts_bars[0] + 1.0, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)
        # Second call: nothing left to mature.
        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores"
        ).fetchone()["n"]
        n_returns = db.execute(
            "SELECT COUNT(*) AS n FROM signal_returns"
        ).fetchone()["n"]
        assert n_scores == 0
        assert n_returns == 1


class TestForwardReturnsZombieRowsDeleted:
    """PR18b: extend matured-row cleanup to rows that can never mature.

    PR18 deleted matured rows (score successfully resolved into an
    R(i,s) row).  Three additional classes of unresolvable rows still
    accumulated until the 30-day TTL purge fired, crowding out fresh
    scores from the per-round LIMIT budget:

      1. Retired-signal rows  (signal removed from SIGNAL_REGISTRY)
      2. Pre-window rows      (score_ts < every candle ts -> i_entry < 0)
      3. Bad-price rows       (entry_price <= 0 at the matched bar)

    All three are now purged from signal_scores in the same DELETE pass.
    """

    def test_retired_signal_score_deleted(self, db, clean_registry):
        from signals.scorer import _compute_forward_returns

        # Insert a score for a signal that is NOT in the registry.
        # (clean_registry yields an empty registry.)
        ts_bars = [3600.0 * i for i in range(1, 6)]
        candles = _make_candles([100.0] * 5, ts_bars)
        _insert_score(db, "retired_sig", "AAPL", ts_bars[0] + 1.0, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        # Zombie row purged; no signal_returns written.
        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores"
        ).fetchone()["n"]
        n_returns = db.execute(
            "SELECT COUNT(*) AS n FROM signal_returns"
        ).fetchone()["n"]
        assert n_scores == 0
        assert n_returns == 0

    def test_pre_window_score_deleted(self, db, clean_registry):
        """Score whose ts predates every candle in the current window
        (i_entry < 0) can never resolve once the window slides forward."""
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        # Candle window starts at hour 100; score is at hour 1 (way before).
        ts_bars = [3600.0 * i for i in range(100, 110)]
        candles = _make_candles([100.0 + i for i in range(10)], ts_bars)

        score_ts = 3600.0  # one hour after epoch -- before all candles
        _insert_score(db, "px_h4", "AAPL", score_ts, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores"
        ).fetchone()["n"]
        n_returns = db.execute(
            "SELECT COUNT(*) AS n FROM signal_returns"
        ).fetchone()["n"]
        assert n_scores == 0
        assert n_returns == 0

    def test_bad_entry_price_deleted(self, db, clean_registry):
        """A score whose matched entry bar has a zero/negative close is
        unrecoverable -- the bar history wont change retroactively."""
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        ts_bars = [3600.0 * i for i in range(1, 11)]
        # Entry bar (index 0) has close 0.0 -> bad-price branch.
        closes = [0.0] + [100.0 + i for i in range(1, 10)]
        candles = _make_candles(closes, ts_bars)

        score_ts = ts_bars[0] + 1.0
        _insert_score(db, "px_h4", "AAPL", score_ts, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores"
        ).fetchone()["n"]
        n_returns = db.execute(
            "SELECT COUNT(*) AS n FROM signal_returns"
        ).fetchone()["n"]
        assert n_scores == 0
        assert n_returns == 0

    def test_no_candles_score_retained(self, db, clean_registry):
        """Sanity check: a score whose symbol is missing from the candle
        map for its resolution (but present elsewhere) must NOT be
        deleted -- the no_candles path is recoverable on later rounds."""
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        # AAPL present in DAILY but signal needs HOURLY.
        ts_d = [86400.0 * i for i in range(1, 4)]
        candles_d = _make_candles([100.0, 110.0, 120.0], ts_d)

        _insert_score(db, "px_h4", "AAPL", ts_d[0] + 1.0, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"D": {"AAPL": candles_d}},
                                 current_ts=ts_d[-1] + 1.0)

        # Score retained (recoverable on later rounds when 1h candles arrive).
        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores"
        ).fetchone()["n"]
        assert n_scores == 1

    def test_pending_score_retained(self, db, clean_registry):
        """Sanity check: pending (horizon not elapsed) is also recoverable
        and must NOT be deleted as a zombie."""
        from signals.scorer import _compute_forward_returns

        _register_signal(clean_registry, name="px_h4", resolution="1h",
                         horizon=4)

        # Only 3 bars of history; horizon=4 -> i_exit out of range -> pending.
        ts_bars = [3600.0 * i for i in range(1, 4)]
        candles = _make_candles([100.0, 101.0, 102.0], ts_bars)

        _insert_score(db, "px_h4", "AAPL", ts_bars[0] + 1.0, score=0.5)

        _compute_forward_returns(db, dp=None,
                                 candles_by_res={"1h": {"AAPL": candles}},
                                 current_ts=ts_bars[-1] + 1.0)

        n_scores = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores"
        ).fetchone()["n"]
        assert n_scores == 1
