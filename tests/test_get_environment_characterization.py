"""PR19 - Characterization tests for ``signals.scorer._get_environment``.

The environment dict feeds every macro signal in the registry, so its
contract is critical: this test pins the existing behavior so future
refactors of the env builder, the regime-duration logic, or the
macro-proximity enrichment will trip a clear failure.

Coverage:
  * **Empty input** -> returns ``None`` (no symbols to assess).
  * **Cross-asset correlation** is added when >= 3 symbols each have
    >= 5 candles, and is the mean of the off-diagonal Pearson corr.
  * **Cross-asset correlation absent** when fewer than 3 usable
    symbols are present.
  * **Macro proximity** keys (``days_to_fomc``, ``days_to_nfp``,
    ``days_to_cpi``) are populated from upcoming economic events when
    matching name keywords appear; events with no keyword match are
    ignored; only the FIRST event of each kind wins.
  * **Regime duration** counts consecutive same-regime snapshots from
    ``environment_snapshots`` (DESC by id) -- the run is broken on
    first mismatch and starts at 1 for an empty table.
  * **Persistence** writes one row to ``environment_snapshots`` per
    successful call.
  * **Outer try/except**: any exception inside the function body
    yields ``None`` (per ``except Exception``) without raising.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest


# ── Fixtures ───────────────────────────────────────────────────


# _isolated_db (autouse) and db fixtures come from tests/conftest.py.


def _candles(closes, *, n_volume=None):
    """Minimal candles object; ``_get_environment`` only reads OHLCV
    lists and ``len(candles)``."""
    n = len(closes)
    high = [c * 1.01 for c in closes]
    low = [c * 0.99 for c in closes]
    open_ = [c * 0.999 for c in closes]
    vol = [1_000_000] * (n_volume if n_volume is not None else n)

    class _Candles:
        def __len__(self):
            return len(self.close)

    obj = _Candles()
    obj.open = open_
    obj.high = high
    obj.low = low
    obj.close = list(closes)
    obj.volume = vol
    return obj


# ── Tests ─────────────────────────────────────────────────────


class TestGetEnvironmentEmpty:
    def test_empty_candles_map_returns_none(self):
        from signals.scorer import _get_environment
        out = _get_environment(dp=None, universe=[], candles_map={})
        assert out is None

    def test_only_zero_length_candles_returns_none(self):
        from signals.scorer import _get_environment
        # candles with len() == 0 are filtered out before df_map is built.
        empty = _candles([])
        out = _get_environment(dp=None, universe=["AAPL"],
                               candles_map={"AAPL": empty})
        assert out is None


class TestCrossAssetCorrelation:
    def test_correlation_added_when_three_symbols_with_history(self, db):
        from signals.scorer import _get_environment
        # Three symbols each with 10 daily-ish closes -> 9 daily returns.
        # Build identical series so correlation = 1.0.
        cl = [100.0 + i for i in range(10)]
        cmap = {"AAPL": _candles(cl), "MSFT": _candles(cl), "NVDA": _candles(cl)}
        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        assert "cross_asset_correlation" in env
        # Identical series -> correlation 1.0.
        assert env["cross_asset_correlation"] == pytest.approx(1.0, abs=1e-9)

    def test_correlation_absent_with_two_symbols(self, db):
        from signals.scorer import _get_environment
        cl = [100.0 + i for i in range(10)]
        cmap = {"AAPL": _candles(cl), "MSFT": _candles(cl)}
        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        # Need >=3 symbols for correlation; 2 means key absent.
        assert "cross_asset_correlation" not in env

    def test_correlation_absent_when_history_too_short(self, db):
        from signals.scorer import _get_environment
        # 4 closes -> 3 returns, but the threshold is len(candles) >= 5.
        # Three symbols, but each has only 4 closes -> no symbol qualifies.
        cl = [100.0, 101.0, 102.0, 103.0]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}
        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        assert "cross_asset_correlation" not in env


class TestMacroProximity:
    """The function calls ``data.economic_calendar.get_upcoming_events``
    and pulls in the FIRST FOMC / NFP / CPI date that matches by
    keyword. We monkeypatch that call to control the data."""

    def _patch_events(self, monkeypatch, events):
        import data.economic_calendar as ec
        monkeypatch.setattr(ec, "get_upcoming_events",
                            lambda days=30, today=None: events)

    def test_keywords_populate_days_to_fields(self, db, monkeypatch):
        from signals.scorer import _get_environment
        today = datetime.date.today()
        Event = SimpleNamespace
        events = [
            Event(name="FOMC Meeting Decision", date=today + datetime.timedelta(days=7)),
            Event(name="Nonfarm Payrolls", date=today + datetime.timedelta(days=2)),
            Event(name="CPI release", date=today + datetime.timedelta(days=14)),
            Event(name="Some unrelated event", date=today + datetime.timedelta(days=1)),
        ]
        self._patch_events(monkeypatch, events)

        cl = [100.0 + i for i in range(10)]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}
        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        assert env["days_to_fomc"] == 7
        assert env["days_to_nfp"] == 2
        assert env["days_to_cpi"] == 14

    def test_first_match_wins_per_category(self, db, monkeypatch):
        """Multiple FOMC events -> only the first (earliest-iterated)
        one populates ``days_to_fomc``."""
        from signals.scorer import _get_environment
        today = datetime.date.today()
        Event = SimpleNamespace
        events = [
            Event(name="FOMC #1", date=today + datetime.timedelta(days=3)),
            Event(name="FOMC #2", date=today + datetime.timedelta(days=21)),
        ]
        self._patch_events(monkeypatch, events)

        cl = [100.0 + i for i in range(10)]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}
        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        assert env["days_to_fomc"] == 3

    def test_calendar_failure_silently_skipped(self, db, monkeypatch):
        """The economic calendar import/call is wrapped in try/except;
        a failure must not break the rest of the env build."""
        import data.economic_calendar as ec

        def _boom(days=30, today=None):
            raise RuntimeError("calendar offline")

        monkeypatch.setattr(ec, "get_upcoming_events", _boom)

        from signals.scorer import _get_environment
        cl = [100.0 + i for i in range(10)]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}
        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        # No proximity keys set, but env still built.
        assert "days_to_fomc" not in env
        assert "days_to_nfp" not in env
        assert "days_to_cpi" not in env


class TestRegimeDurationAndPersistence:
    def test_first_call_writes_snapshot_and_durations_start_at_1(self, db):
        from signals.scorer import _get_environment
        cl = [100.0 + i for i in range(10)]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}

        before = db.execute(
            "SELECT COUNT(*) AS n FROM environment_snapshots"
        ).fetchone()["n"]
        assert before == 0

        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        # Empty prior history -> durations start at 1.
        assert env["vol_regime_duration"] == 1
        assert env["trend_regime_duration"] == 1

        after = db.execute(
            "SELECT COUNT(*) AS n FROM environment_snapshots"
        ).fetchone()["n"]
        assert after == 1

    def test_consecutive_same_regime_increments_duration(self, db):
        from signals.scorer import _get_environment
        cl = [100.0 + i for i in range(10)]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}

        env1 = _get_environment(dp=None, universe=list(cmap.keys()),
                                candles_map=cmap)
        env2 = _get_environment(dp=None, universe=list(cmap.keys()),
                                candles_map=cmap)
        assert env1 is not None and env2 is not None
        # Same input -> same regimes -> duration increments by 1.
        # First call: duration=1 (no priors). Second call sees one prior
        # row matching its regime -> duration=2.
        assert env2["vol_regime_duration"] == env1["vol_regime_duration"] + 1
        assert env2["trend_regime_duration"] == env1["trend_regime_duration"] + 1

    def test_regime_break_resets_duration(self, db):
        from signals.scorer import _get_environment
        # Pre-seed the snapshots table with rows of a DIFFERENT regime
        # so the duration counter cannot match the new env's regimes.
        for _ in range(3):
            db.execute(
                "INSERT INTO environment_snapshots "
                "(ts, volatility_regime, trend_regime, breadth_regime, "
                " momentum_regime, volume_regime, avg_atr_pct, dispersion, "
                " raw_snapshot_json) "
                "VALUES (?, 'IMPOSSIBLE_REGIME_X', 'IMPOSSIBLE_TREND_Y', "
                " 'unknown', 'unknown', 'unknown', 0.0, 0.0, '{}')",
                (datetime.datetime.now(datetime.timezone.utc).isoformat(),),
            )
        db.commit()

        cl = [100.0 + i for i in range(10)]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}
        env = _get_environment(dp=None, universe=list(cmap.keys()),
                               candles_map=cmap)
        assert env is not None
        # No prior row matched -> duration stays at 1.
        assert env["vol_regime_duration"] == 1
        assert env["trend_regime_duration"] == 1


class TestRoundNumPersistence:
    def test_round_num_written_to_environment_snapshots(self, db):
        from signals.scorer import _get_environment

        cl = [100.0 + i for i in range(10)]
        cmap = {s: _candles(cl) for s in ("AAPL", "MSFT", "NVDA")}
        env = _get_environment(
            dp=None,
            universe=list(cmap.keys()),
            candles_map=cmap,
            round_num=185,
        )
        assert env is not None
        row = db.execute(
            "SELECT round_num FROM environment_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["round_num"] == 185
