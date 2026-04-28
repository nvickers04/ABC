"""PR22 - Characterization tests for ``compute_composite_scores``.

Existing ``tests/test_signals.py::test_compute_composite_scores`` only
asserts shape + range. This file pins the more subtle invariants:

  * Symbols with no signal_scores rows are omitted from the result dict.
  * raw_composite is normalized by ``active_weight_sum`` (signals with
    confidence > 0); zero-confidence signals don't dilute.
  * Output is clamped to [-1, 1].
  * Only the latest ts per (signal, symbol) is used.
  * Persisted ``composite_scores`` row contains the breakdown JSON keyed
    by signal *category* (not by signal name).
"""

from __future__ import annotations

import json
import time

import pytest

# _isolated_db (autouse) and db fixtures are provided by tests/conftest.py.


def _insert_score(db, signal_name, symbol, ts, score, confidence=1.0):
    db.execute(
        "INSERT OR REPLACE INTO signal_scores "
        "(signal_name, symbol, ts, score, confidence, components_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (signal_name, symbol, ts, score, confidence, "{}"),
    )
    db.commit()


# ── Tests ─────────────────────────────────────────────────────


class TestCompositeScoresEdgeCases:
    def test_symbol_with_no_scores_omitted(self, db):
        from signals.combiner import compute_composite_scores
        weights = {"momentum": 1.0}
        # Don't seed anything for "GHOST".
        out = compute_composite_scores(db, weights, ["GHOST"], ts=time.time())
        assert out == {}

    def test_weighted_sum_normalized_by_active_weight(self, db):
        """raw / active_weight_sum: equal weights with all-confident
        signals -> simple weighted average."""
        from signals.combiner import compute_composite_scores
        ts = time.time()
        # weights sum to 1.0; all confidences > 0 -> active_weight_sum = 1.0
        weights = {"momentum": 0.5, "breadth": 0.5}
        _insert_score(db, "momentum", "AAPL", ts, 0.4, confidence=1.0)
        _insert_score(db, "breadth", "AAPL", ts, 0.8, confidence=1.0)
        out = compute_composite_scores(db, weights, ["AAPL"], ts=ts)
        # raw = 0.5*0.4 + 0.5*0.8 = 0.6 ; active = 0.5+0.5 = 1.0
        assert out["AAPL"] == pytest.approx(0.6, abs=1e-9)

    def test_zero_confidence_signal_does_not_dilute(self, db):
        """A signal with confidence=0 still contributes to raw_composite
        but is excluded from active_weight_sum -> denominator shrinks
        instead of pulling the composite toward 0."""
        from signals.combiner import compute_composite_scores
        ts = time.time()
        weights = {"momentum": 0.5, "breadth": 0.5}
        # momentum: real score, confidence=1
        _insert_score(db, "momentum", "AAPL", ts, 0.4, confidence=1.0)
        # breadth: present row but confidence=0 -> not active
        _insert_score(db, "breadth", "AAPL", ts, 0.0, confidence=0.0)
        out = compute_composite_scores(db, weights, ["AAPL"], ts=ts)
        # raw = 0.5*0.4 + 0.5*0.0 = 0.2 ; active = 0.5 (only momentum)
        # composite = 0.2 / 0.5 = 0.4
        assert out["AAPL"] == pytest.approx(0.4, abs=1e-9)

    def test_all_zero_confidence_yields_zero(self, db):
        from signals.combiner import compute_composite_scores
        ts = time.time()
        weights = {"momentum": 1.0}
        _insert_score(db, "momentum", "AAPL", ts, 0.7, confidence=0.0)
        out = compute_composite_scores(db, weights, ["AAPL"], ts=ts)
        # active_weight_sum == 0 -> composite forced to 0.0
        assert out["AAPL"] == 0.0

    def test_output_clamped_to_unit_interval(self, db):
        from signals.combiner import compute_composite_scores
        ts = time.time()
        # Single signal with weight 1, score = 0.9 -> composite = 0.9.
        # Force clamp by giving score > 1 directly in the row (bypassing
        # the normal ``Signal.score`` clamp).
        weights = {"momentum": 1.0}
        _insert_score(db, "momentum", "AAPL", ts, 5.0, confidence=1.0)
        out = compute_composite_scores(db, weights, ["AAPL"], ts=ts)
        assert out["AAPL"] == 1.0  # clamped

        _insert_score(db, "momentum", "MSFT", ts, -5.0, confidence=1.0)
        out2 = compute_composite_scores(db, weights, ["MSFT"], ts=ts)
        assert out2["MSFT"] == -1.0  # clamped

    def test_only_latest_ts_used_per_signal(self, db):
        """Older rows with bigger scores are ignored; only the row at
        MAX(ts) per (signal, symbol) contributes."""
        from signals.combiner import compute_composite_scores
        old_ts = 1000.0
        new_ts = 2000.0
        weights = {"momentum": 1.0}
        # Old row: score=0.9
        _insert_score(db, "momentum", "AAPL", old_ts, 0.9, confidence=1.0)
        # New row: score=0.1 -- this should win.
        _insert_score(db, "momentum", "AAPL", new_ts, 0.1, confidence=1.0)
        out = compute_composite_scores(db, weights, ["AAPL"], ts=new_ts)
        assert out["AAPL"] == pytest.approx(0.1, abs=1e-9)

    def test_persisted_breakdown_keyed_by_category(self, db):
        """The breakdown JSON stored alongside composite_score is keyed
        by signal *category* (not by signal name). Unknown signals fall
        under 'unknown'."""
        from signals.combiner import compute_composite_scores
        ts = time.time()
        # 'momentum' is registered -> its category should appear.
        weights = {"momentum": 1.0}
        _insert_score(db, "momentum", "AAPL", ts, 0.5, confidence=1.0)
        compute_composite_scores(db, weights, ["AAPL"], ts=ts)

        row = db.execute(
            "SELECT signal_breakdown_json FROM composite_scores WHERE symbol = ?",
            ("AAPL",),
        ).fetchone()
        assert row is not None
        breakdown = json.loads(row[0])
        # Keys are categories. The momentum signal's category is whatever
        # ``signals.momentum`` declared; we only assert it's a non-empty
        # dict whose values sum to the raw weighted score.
        assert isinstance(breakdown, dict)
        assert len(breakdown) >= 1
        assert sum(breakdown.values()) == pytest.approx(0.5, abs=1e-9)

    def test_unknown_signal_category_is_unknown(self, db):
        """Signals not in SIGNAL_REGISTRY are bucketed under 'unknown'."""
        from signals.combiner import compute_composite_scores
        ts = time.time()
        weights = {"not_a_real_signal": 1.0}
        _insert_score(db, "not_a_real_signal", "AAPL", ts, 0.3,
                      confidence=1.0)
        compute_composite_scores(db, weights, ["AAPL"], ts=ts)
        row = db.execute(
            "SELECT signal_breakdown_json FROM composite_scores WHERE symbol = ?",
            ("AAPL",),
        ).fetchone()
        breakdown = json.loads(row[0])
        assert "unknown" in breakdown
        assert breakdown["unknown"] == pytest.approx(0.3, abs=1e-9)
