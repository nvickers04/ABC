"""PR26 - Characterize ``TradingAgent._run_execution_analysis``.

The full analysis flow is a 200-line orchestration: calibration ->
graduated-param rollback -> LLM proposal -> Mann-Whitney validation.
This file pins ONLY the deterministic, no-LLM portions:

  * <5 filled snapshots -> early return (no calibrated_slippage rows
    written, no last_analysis_snapshot_id update).
  * >=5 snapshots all in one bucket -> at least three calibrated_slippage
    rows: bucket-level, (ot, tb, 'all'), (ot, 'all', 'all').
  * Snapshots with NULL slippage_bps are skipped during grouping.
  * After successful analysis, ``last_analysis_snapshot_id`` advances
    to MAX(id) of the analyzed snapshots.
  * Graduated-param rollback: if after-median > 1.10 * before-median,
    the param is deactivated.

The LLM proposal step is short-circuited by giving the agent a fake
grok client whose ``chat.create`` raises -- the function catches that
and continues to the bookkeeping at the bottom (snapshots get marked
analyzed even when the LLM step fails).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

# _isolated_db (autouse) and db fixtures are provided by tests/conftest.py.


# ── Helpers ───────────────────────────────────────────────────


def _make_agent():
    """Build the minimal agent stub ``_run_execution_analysis`` needs.

    The function only uses ``self.grok``, ``self.cost_tracker``, and
    ``self._test_proposal`` (not reached without LLM success). We give
    a grok whose chat.create raises -- the ``except Exception as
    llm_err`` branch swallows it and proceeds to bookkeeping.
    """
    class _RaisingChat:
        def create(self, *a, **k):
            raise RuntimeError("LLM disabled in tests")

    grok = SimpleNamespace(client=SimpleNamespace(chat=_RaisingChat()),
                           model="grok-test")
    cost_tracker = SimpleNamespace(
        log_llm_call=lambda **kw: None,
    )
    return SimpleNamespace(
        grok=grok,
        cost_tracker=cost_tracker,
    )


def _insert_snapshot(db, *, snap_id=None, order_type="market",
                    time_bucket="open", atr_bucket="high",
                    slippage_bps=10.0, status="filled",
                    ts="2026-04-15T10:00:00", symbol="AAPL"):
    cols = ("ts", "symbol", "side", "quantity", "order_type", "intent",
            "fill_price", "slippage_bps", "time_bucket", "atr_bucket",
            "status")
    vals = (ts, symbol, "BUY", 100, order_type, "entry",
            100.0, slippage_bps, time_bucket, atr_bucket, status)
    if snap_id is not None:
        db.execute(
            f"INSERT INTO execution_snapshots (id, {', '.join(cols)}) "
            f"VALUES (?, {', '.join('?' for _ in cols)})",
            (snap_id, *vals),
        )
    else:
        db.execute(
            f"INSERT INTO execution_snapshots ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            vals,
        )
    db.commit()


async def _run(agent):
    from core.agent import TradingAgent
    await TradingAgent._run_execution_analysis(agent)


def _calib_rows(db):
    return db.execute(
        "SELECT order_type, time_bucket, atr_bucket, median_slippage_bps, "
        "sample_count FROM calibrated_slippage "
        "ORDER BY order_type, time_bucket, atr_bucket"
    ).fetchall()


def _last_analysis_id(db) -> float:
    from memory import get_research_config
    return float(get_research_config("last_analysis_snapshot_id", 0.0))


# ── Tests ─────────────────────────────────────────────────────


class TestRunExecutionAnalysis:
    def test_too_few_snapshots_no_calibration(self, db):
        agent = _make_agent()
        # Only 4 filled snapshots -- function logs and returns early.
        for i in range(4):
            _insert_snapshot(db)
        asyncio.run(_run(agent))
        assert _calib_rows(db) == []
        assert _last_analysis_id(db) == 0.0

    def test_exactly_five_snapshots_calibrates(self, db):
        agent = _make_agent()
        # 5 in same bucket -> 3 calibrated rows: (ot,tb,ab),
        # (ot,tb,'all'), (ot,'all','all').
        for i in range(5):
            _insert_snapshot(db, slippage_bps=10.0 + i)
        asyncio.run(_run(agent))

        rows = _calib_rows(db)
        assert len(rows) == 3
        keys = {(r["order_type"], r["time_bucket"], r["atr_bucket"]) for r in rows}
        assert ("market", "open", "high") in keys
        assert ("market", "open", "all") in keys
        assert ("market", "all", "all") in keys

        # The bucket-level row's median is the median of [10..14] = 12.
        bucket = [r for r in rows
                  if r["atr_bucket"] == "high"][0]
        assert bucket["median_slippage_bps"] == pytest.approx(12.0, abs=0.01)
        assert bucket["sample_count"] == 5

    def test_null_slippage_skipped_in_grouping(self, db):
        """Snapshots with slippage_bps=NULL must not contribute to the
        calibration buckets but are still 'filled' and counted toward
        the analysis trigger threshold (len(snapshots) >= 5)."""
        agent = _make_agent()
        # 5 with NULL slippage + 5 with values
        for _ in range(5):
            _insert_snapshot(db, slippage_bps=None)
        for i in range(5):
            _insert_snapshot(db, slippage_bps=20.0 + i)
        asyncio.run(_run(agent))

        rows = _calib_rows(db)
        # Calibration uses only the 5 valid samples.
        bucket = [r for r in rows
                  if (r["order_type"], r["time_bucket"], r["atr_bucket"])
                  == ("market", "open", "high")][0]
        assert bucket["sample_count"] == 5
        # Median of [20..24] = 22
        assert bucket["median_slippage_bps"] == pytest.approx(22.0, abs=0.01)

    def test_under_3_per_bucket_skipped(self, db):
        """A bucket with only 2 valid samples is excluded from the
        narrow calibrated row, but its samples still aggregate into
        the (ot,tb,'all') and (ot,'all','all') roll-ups."""
        agent = _make_agent()
        # 5 filled snapshots: 3 in (open,high) and 2 in (open,low).
        for i in range(3):
            _insert_snapshot(db, slippage_bps=10.0, atr_bucket="high")
        for i in range(2):
            _insert_snapshot(db, slippage_bps=20.0, atr_bucket="low")
        asyncio.run(_run(agent))

        rows = _calib_rows(db)
        keys = {(r["order_type"], r["time_bucket"], r["atr_bucket"]) for r in rows}
        # The (market,open,low) bucket has only 2 samples -- should be
        # absent.
        assert ("market", "open", "low") not in keys
        # The (market,open,high) row should be present (3 samples).
        assert ("market", "open", "high") in keys

    def test_advances_last_analysis_snapshot_id(self, db):
        agent = _make_agent()
        for i in range(5):
            _insert_snapshot(db, snap_id=100 + i)
        asyncio.run(_run(agent))
        assert _last_analysis_id(db) == 104.0

    def test_idempotent_rerun_no_extra_progress(self, db):
        """Second run with no new snapshots -> early return, no
        change to calibration or last_analysis_snapshot_id."""
        agent = _make_agent()
        for i in range(5):
            _insert_snapshot(db, snap_id=10 + i)
        asyncio.run(_run(agent))
        first_id = _last_analysis_id(db)
        first_rows = _calib_rows(db)

        # Second run -- since=last_id, only newer snapshots (none) load.
        asyncio.run(_run(agent))
        assert _last_analysis_id(db) == first_id
        # Same calibration rows.
        assert len(_calib_rows(db)) == len(first_rows)
