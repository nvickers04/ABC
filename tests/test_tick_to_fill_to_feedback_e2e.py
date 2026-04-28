"""PR34 — End-to-end "tick → fill → feedback" regression test.

This is the one integration test that proves the data-flow seam between
the agent/tools layer (plan-time context), the execution layer (snapshot
+ fill), and the feedback layer (cost aggregation, calibration) is wired
correctly. Unit tests cover each stage in isolation; this test pins the
*composition*.

The flow under test (one symbol, one bracket order, one fill):

    1. TICK (plan)     tools.tools_executor publishes (graduated_param_id,
                       intent, atr_pct) for AAPL via the new accessors
                       added in PR35.

    2. SUBMIT          execution layer calls insert_execution_snapshot —
                       this MUST consume both pending dicts and link the
                       new snapshot row to the graduated_param.

    3. FILL            execution layer calls update_execution_snapshot_fill
                       with a worse-than-mid price → slippage_bps becomes
                       a positive number, status flips to 'filled'.

    4. FEEDBACK        feedback layer calls get_execution_cost and
                       get_filled_snapshots — the row from step 3 must
                       be visible with its computed slippage and the
                       graduated_param_id intact.

If any link in this chain breaks (private dicts renamed without updating
consumers, snapshot schema drift, fill update missing slippage math,
filtered-by-status query regression), this test will catch it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest


# Shared `_isolated_db` autouse fixture from tests/conftest.py gives us a
# fresh DB per test. `db` fixture is unused here (we go through the public
# memory functions, which open their own conn via memory.get_db()).


def test_tick_to_fill_to_feedback_roundtrip(db):
    """One symbol, one bracket, one fill — every linkage must survive."""
    import memory
    from memory import (
        # Tick (plan)
        set_pending_graduated_param,
        set_pending_order_context,
        get_pending_graduated_param,
        get_pending_order_context,
        # Submit
        insert_execution_snapshot,
        insert_graduated_param,
        # Fill
        update_execution_snapshot_fill,
        # Feedback
        get_filled_snapshots,
        get_new_snapshot_count,
        upsert_calibrated_slippage,
    )
    from memory.repos.execution_repo import get_calibrated_slippage

    symbol = "AAPL"

    # ─── Step 0: a graduated param exists (e.g. from a past LLM cycle) ──
    param_id = insert_graduated_param(
        param_key="market.entry.midday.medium",
        param_value="adaptive",
        previous_value=None,
        evidence_json=json.dumps({"n_before": 12, "n_after": 12, "improvement_bps": 30.0}),
        snapshots_analyzed=24,
        improvement_bps=30.0,
        p_value=0.01,
    )
    assert param_id is not None, "insert_graduated_param returned None — schema regression"

    # ─── Step 1: TICK / PLAN ─────────────────────────────────────────────
    # tools.tools_executor publishes plan-time context.
    set_pending_graduated_param(symbol, param_id)
    set_pending_order_context(symbol, {"intent": "entry", "atr_pct": 1.8})

    # Sanity: peek accessors return what was set (without consuming).
    assert get_pending_graduated_param(symbol) == param_id
    assert get_pending_order_context(symbol) == {"intent": "entry", "atr_pct": 1.8}

    # ─── Step 2: SUBMIT ──────────────────────────────────────────────────
    # execution layer calls insert_execution_snapshot.  We pass intent='unknown'
    # and atr_pct=None — the function MUST auto-consume the pending dicts
    # and substitute the planned values.  This is the actual production
    # call site contract (see memory.insert_execution_snapshot).
    snapshot_id = insert_execution_snapshot(
        symbol=symbol,
        side="BUY",
        quantity=10,
        order_type="LMT",
        intent="unknown",            # placeholder — should be overwritten
        bid=149.95, ask=150.05, mid=150.00, spread=0.10,
        volume=1_000_000,
        atr=2.7, atr_pct=None,       # placeholder — should be overwritten
    )
    assert snapshot_id is not None, "insert_execution_snapshot returned None"

    # Pending dicts MUST have been consumed (popped) — so a second submit
    # for the same symbol picks up *no* stale context.
    assert get_pending_graduated_param(symbol) is None
    assert get_pending_order_context(symbol) == {}

    # The snapshot row must reflect the planned values + linkage.
    row = db.execute(
        "SELECT symbol, intent, atr_at_submit, graduated_param_id, status, "
        "       order_type, mid_at_submit "
        "FROM execution_snapshots WHERE id = ?",
        (snapshot_id,),
    ).fetchone()
    assert row is not None, "snapshot was not persisted"
    assert row["symbol"] == symbol
    # Intent was overwritten from 'unknown' → 'entry' via _pending_order_context.
    assert row["intent"] == "entry"
    # graduated_param_id wired through from _pending_graduated_params.
    assert row["graduated_param_id"] == param_id
    assert row["status"] == "submitted"
    assert row["mid_at_submit"] == pytest.approx(150.00)

    # Before fill: filled-snapshot count is zero — feedback path sees nothing yet.
    assert get_new_snapshot_count() == 0
    assert get_filled_snapshots() == []

    # ─── Step 3: FILL ────────────────────────────────────────────────────
    submit_ts = row_ts = db.execute(
        "SELECT ts FROM execution_snapshots WHERE id = ?", (snapshot_id,)
    ).fetchone()["ts"]
    # Fill 50 ms later, 5 cents worse than mid (BUY @ 150.05 vs mid 150.00).
    fill_dt = datetime.fromisoformat(submit_ts) + timedelta(milliseconds=50)
    fill_time = fill_dt.isoformat()
    update_execution_snapshot_fill(
        snapshot_id=snapshot_id,
        fill_price=150.05,
        fill_time=fill_time,
        commission=1.00,
        partial_fills=0,
    )

    # ─── Step 4: FEEDBACK ────────────────────────────────────────────────
    # The fill must now be visible to the analysis pipeline.
    assert get_new_snapshot_count() == 1, "filled snapshot not counted"
    filled = get_filled_snapshots()
    assert len(filled) == 1, "expected exactly one filled snapshot"
    f = filled[0]
    assert f["id"] == snapshot_id
    assert f["graduated_param_id"] == param_id, \
        "graduated_param linkage lost between submit and fill"
    assert f["status"] == "filled"
    assert f["fill_price"] == pytest.approx(150.05)

    # Slippage math:  (150.05 - 150.00) / 150.00 * 10000 ≈ 3.33 bps.
    assert f["slippage_bps"] is not None
    assert f["slippage_bps"] == pytest.approx(3.33, abs=0.05)

    # Latency in ms — non-zero, plausible.
    assert f["latency_ms"] is not None
    assert 25 <= f["latency_ms"] <= 200

    # Aggregated feedback surface: drive the calibration upsert that the
    # autoresearch / scorer pipelines use after batches of fills.  This is
    # the actual closing edge of the feedback loop for the execution-cost
    # model — `get_execution_cost` reads from `trade_feedback` (a separate
    # pipeline populated by `record_trade`) and is not what consumes these
    # filled snapshots.
    upsert_calibrated_slippage(
        order_type="limit",
        time_bucket="midday",
        atr_bucket="medium",
        median_bps=float(f["slippage_bps"]),
        sample_count=1,
    )
    calibrated = get_calibrated_slippage()
    assert ("limit", "midday", "medium") in calibrated
    assert calibrated[("limit", "midday", "medium")] == pytest.approx(
        f["slippage_bps"], abs=0.05
    )


def test_two_concurrent_symbols_keep_pending_state_isolated(db):
    """Two symbols planned then submitted out-of-order — each one's
    pending context must reach its own snapshot, no cross-talk."""
    from memory import (
        set_pending_graduated_param,
        set_pending_order_context,
        insert_execution_snapshot,
        insert_graduated_param,
    )

    pid_aapl = insert_graduated_param(
        param_key="market.entry.open.high",
        param_value="adaptive",
        previous_value=None,
        evidence_json="{}",
        snapshots_analyzed=10,
        improvement_bps=20.0,
        p_value=0.02,
    )
    pid_msft = insert_graduated_param(
        param_key="market.entry.midday.low",
        param_value="adaptive",
        previous_value=None,
        evidence_json="{}",
        snapshots_analyzed=10,
        improvement_bps=15.0,
        p_value=0.03,
    )

    # Plan both first, then submit MSFT first (interleaved order).
    set_pending_graduated_param("AAPL", pid_aapl)
    set_pending_order_context("AAPL", {"intent": "entry", "atr_pct": 1.5})
    set_pending_graduated_param("MSFT", pid_msft)
    set_pending_order_context("MSFT", {"intent": "exit", "atr_pct": 2.5})

    sid_msft = insert_execution_snapshot(
        symbol="MSFT", side="SELL", quantity=5, order_type="LMT",
        intent="unknown", bid=300.0, ask=300.2, mid=300.1, spread=0.2,
        volume=500_000, atr=4.0, atr_pct=None,
    )
    sid_aapl = insert_execution_snapshot(
        symbol="AAPL", side="BUY", quantity=10, order_type="LMT",
        intent="unknown", bid=149.95, ask=150.05, mid=150.0, spread=0.1,
        volume=1_000_000, atr=2.7, atr_pct=None,
    )

    rows = {
        r["symbol"]: r for r in db.execute(
            "SELECT symbol, intent, atr_at_submit, graduated_param_id "
            "FROM execution_snapshots WHERE id IN (?, ?)",
            (sid_msft, sid_aapl),
        ).fetchall()
    }
    assert rows["AAPL"]["intent"] == "entry"
    assert rows["AAPL"]["graduated_param_id"] == pid_aapl
    assert rows["MSFT"]["intent"] == "exit"
    assert rows["MSFT"]["graduated_param_id"] == pid_msft
