"""Validation tests for the per-signal forward-return pipeline.

Covers the Phase D rewrite of ``signals.scorer._compute_forward_returns``:
- entry-bar keyed dedup (intraday rounds collapse to one row per bar)
- per-signal horizon (signal's ``return_horizon`` attribute is honoured)
- per-resolution candle routing (signals only see their own resolution)
- never-mature row pruning (orphan symbols + 30-day TTL)
"""
from __future__ import annotations

import sqlite3
import time

import pytest


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """In-memory DB with the minimal schema used by _compute_forward_returns."""
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE signal_scores (
          signal_name TEXT, symbol TEXT, ts REAL, score REAL,
          PRIMARY KEY (signal_name, symbol, ts)
        );
        CREATE TABLE signal_returns (
          signal_name TEXT, symbol TEXT, ts REAL,
          score_at_entry REAL, forward_return REAL, r_value REAL,
          horizon_bars INTEGER,
          PRIMARY KEY (signal_name, symbol, ts)
        );
        """
    )
    yield c
    c.close()


def _candles(timestamps, closes):
    class _C:
        pass

    obj = _C()
    obj.timestamps = list(timestamps)
    obj.close = list(closes)
    return obj


# ── Tests ───────────────────────────────────────────────────────

class TestEntryBarKeying:
    """Multiple intraday score rounds resolving to the same bar must
    collapse to ONE signal_returns row (the latest-scored INSERT OR
    REPLACE wins).  This is the core fix for IC-magnitude inflation."""

    def test_three_intraday_scores_collapse_to_one_row(self, conn):
        from signals.scorer import _compute_forward_returns
        import signals.market_momentum  # noqa: F401  (registers signal)
        from signals.base import SIGNAL_REGISTRY

        sig = SIGNAL_REGISTRY["market_momentum"]
        assert sig.return_resolution == "D"
        assert sig.return_horizon == 1

        now = time.time()
        ts = [now - (60 - i) * 86400 for i in range(60)]
        closes = [100.0 + i for i in range(60)]
        cmap = {"D": {"AAPL": _candles(ts, closes)}}

        # Three intraday scores all map to bar 57 (exit=58 is in-range).
        for offset, score in [(1000, 1.0), (5000, 0.5), (10000, -0.5)]:
            conn.execute(
                "INSERT INTO signal_scores VALUES (?, ?, ?, ?)",
                ("market_momentum", "AAPL", ts[57] + offset, score),
            )
        conn.commit()

        _compute_forward_returns(conn, dp=None, candles_by_res=cmap, current_ts=now)

        rows = conn.execute(
            "SELECT signal_name, symbol, ts, score_at_entry, horizon_bars "
            "FROM signal_returns"
        ).fetchall()
        assert len(rows) == 1, "intraday rounds should collapse to one row per entry bar"
        assert rows[0][2] == pytest.approx(ts[57], abs=1e-3), \
            "row must be keyed by entry_bar_ts not score_ts"
        # Last write wins (score=-0.5 was the most recent).
        assert rows[0][3] == -0.5
        assert rows[0][4] == 1


class TestPerSignalHorizon:
    """Each signal's own return_horizon must drive the bar lookahead."""

    def test_horizon_5_skips_5_bars(self, conn):
        from signals.scorer import _compute_forward_returns
        import signals.cash_flow_yield  # noqa: F401
        from signals.base import SIGNAL_REGISTRY

        sig = SIGNAL_REGISTRY["cash_flow_yield"]
        assert sig.return_resolution == "D"
        assert sig.return_horizon == 5

        now = time.time()
        ts = [now - (60 - i) * 86400 for i in range(60)]
        # Linear ramp: forward return over 5 bars = 5 / entry_price
        closes = [100.0 + i for i in range(60)]
        cmap = {"D": {"AAPL": _candles(ts, closes)}}

        # Score at bar 50 → exit bar 55, both in range.
        conn.execute(
            "INSERT INTO signal_scores VALUES (?, ?, ?, ?)",
            ("cash_flow_yield", "AAPL", ts[50] + 100, 1.0),
        )
        conn.commit()

        _compute_forward_returns(conn, dp=None, candles_by_res=cmap, current_ts=now)

        row = conn.execute(
            "SELECT forward_return, horizon_bars FROM signal_returns"
        ).fetchone()
        assert row is not None
        expected = (closes[55] - closes[50]) / closes[50]
        assert row[0] == pytest.approx(expected, rel=1e-9)
        assert row[1] == 5


class TestPerResolutionRouting:
    """A signal must only see candles from its own resolution map."""

    def test_daily_signal_ignored_when_only_subdaily_candles_present(self, conn):
        from signals.scorer import _compute_forward_returns
        import signals.market_momentum  # noqa: F401  (D, h=1)

        now = time.time()
        ts = [now - (60 - i) * 86400 for i in range(60)]
        closes = [100.0 + i for i in range(60)]

        # Sub-daily candles present, but the daily map is empty: signal
        # should produce zero rows even though scores exist.
        cmap = {"D": {}, "5min": {"AAPL": _candles(ts, closes)}}

        conn.execute(
            "INSERT INTO signal_scores VALUES (?, ?, ?, ?)",
            ("market_momentum", "AAPL", ts[50] + 100, 1.0),
        )
        conn.commit()

        _compute_forward_returns(conn, dp=None, candles_by_res=cmap, current_ts=now)
        n = conn.execute("SELECT COUNT(*) FROM signal_returns").fetchone()[0]
        assert n == 0, "daily signal must not consume sub-daily candles"


class TestPruneAndTTL:
    """Orphan-symbol prune and 30-day TTL keep the per-round LIMIT
    budget unblocked so fresh scores actually mature into IC."""

    def test_orphan_symbol_score_deleted(self, conn):
        from signals.scorer import _compute_forward_returns
        import signals.market_momentum  # noqa: F401

        now = time.time()
        ts = [now - (60 - i) * 86400 for i in range(60)]
        closes = [100.0 + i for i in range(60)]
        cmap = {"D": {"AAPL": _candles(ts, closes)}}

        # XYZ is not in any candle map → orphan → must be deleted.
        conn.execute(
            "INSERT INTO signal_scores VALUES (?, ?, ?, ?)",
            ("market_momentum", "XYZ", ts[50] + 100, 1.0),
        )
        conn.commit()

        _compute_forward_returns(conn, dp=None, candles_by_res=cmap, current_ts=now)

        n = conn.execute(
            "SELECT COUNT(*) FROM signal_scores WHERE symbol='XYZ'"
        ).fetchone()[0]
        assert n == 0, "orphan-symbol score must be pruned"

    def test_thirty_day_ttl_drops_ancient_scores(self, conn):
        from signals.scorer import _compute_forward_returns
        import signals.market_momentum  # noqa: F401

        now = time.time()
        ts = [now - (60 - i) * 86400 for i in range(60)]
        closes = [100.0 + i for i in range(60)]
        cmap = {"D": {"AAPL": _candles(ts, closes)}}

        ancient_ts = now - 45 * 86400  # > 30 day cutoff
        conn.execute(
            "INSERT INTO signal_scores VALUES (?, ?, ?, ?)",
            ("market_momentum", "AAPL", ancient_ts, 1.0),
        )
        conn.commit()

        _compute_forward_returns(conn, dp=None, candles_by_res=cmap, current_ts=now)

        n = conn.execute(
            "SELECT COUNT(*) FROM signal_scores WHERE ts=?", (ancient_ts,),
        ).fetchone()[0]
        assert n == 0, "scores older than 30d must be TTL-purged"
