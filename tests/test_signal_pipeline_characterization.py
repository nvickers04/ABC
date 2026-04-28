"""PR15 — Signal-pipeline characterization tests.

Locks in the public contracts of the lower layers of the signal stack
that were previously only covered transitively through the combiner /
template tests. Each test documents the exact behavior currently
relied on by the agent, so future refactors of these modules will
trip a clear failure if they regress.

Covers:
  * ``signals.base.Signal.score`` — safe wrapper: clamping, exception
    capture, components passthrough.
  * ``signals.base.SignalResult.to_dict`` — round-trip shape.
  * ``signals.scorer._persist_scores`` — row layout, JSON encoding,
    INSERT OR REPLACE upsert behavior, no-op on empty input.
"""

from __future__ import annotations

import json
import time

import pytest


# ── Shared fixture (mirrors tests/test_signals.py) ─────────────


# _isolated_db (autouse) and db fixtures come from tests/conftest.py.


# ── Signal.score safe wrapper ──────────────────────────────────


class TestSignalSafeWrapper:
    def _make_signal(self, *, score=0.0, conf=0.0, raise_exc=None,
                     components=None):
        """Build a one-off Signal subclass that returns a fixed result."""
        from signals.base import Signal, SignalResult

        class _OneOff(Signal):
            name = ""  # empty -> not registered globally
            category = "price"
            data_source = "candles"

            def compute(self_inner, symbol, data):
                if raise_exc is not None:
                    raise raise_exc
                return SignalResult(
                    score=score,
                    confidence=conf,
                    components=components or {},
                )

        return _OneOff()

    def test_score_passthrough_in_range(self):
        sig = self._make_signal(score=0.42, conf=0.7,
                                components={"momentum": 0.42})
        out = sig.score("AAPL", {})
        assert out == {
            "score": 0.42,
            "confidence": 0.7,
            "components": {"momentum": 0.42},
        }

    def test_score_clamps_above_one(self):
        sig = self._make_signal(score=2.5, conf=0.5)
        out = sig.score("AAPL", {})
        assert out["score"] == 1.0

    def test_score_clamps_below_neg_one(self):
        sig = self._make_signal(score=-3.7, conf=0.5)
        out = sig.score("AAPL", {})
        assert out["score"] == -1.0

    def test_confidence_clamps_to_zero_one(self):
        sig_high = self._make_signal(score=0.0, conf=1.5)
        sig_low = self._make_signal(score=0.0, conf=-0.2)
        assert sig_high.score("AAPL", {})["confidence"] == 1.0
        assert sig_low.score("AAPL", {})["confidence"] == 0.0

    def test_exception_returns_zero_result_with_error(self):
        sig = self._make_signal(raise_exc=ValueError("boom"))
        out = sig.score("AAPL", {})
        # The agent relies on this exact shape: a failed signal
        # contributes 0 to the composite (zero score, zero confidence)
        # and surfaces the error message in components for debugging.
        assert out["score"] == 0.0
        assert out["confidence"] == 0.0
        assert out["components"] == {"error": "boom"}


class TestSignalResultToDict:
    def test_to_dict_shape(self):
        from signals.base import SignalResult
        r = SignalResult(score=0.1, confidence=0.2, components={"k": 1})
        d = r.to_dict()
        assert d == {"score": 0.1, "confidence": 0.2, "components": {"k": 1}}

    def test_to_dict_does_not_alias_components(self):
        # The dict carries the same components reference today (no copy);
        # this test pins that fact so any future copy-on-export change
        # is intentional and visible.
        from signals.base import SignalResult
        comp = {"k": 1}
        r = SignalResult(score=0.0, confidence=0.0, components=comp)
        assert r.to_dict()["components"] is comp


# ── scorer._persist_scores ─────────────────────────────────────


class TestPersistScores:
    def test_writes_one_row_per_signal_symbol_pair(self, db):
        from signals.scorer import _persist_scores
        ts = time.time()
        scores = {
            "AAPL": {
                "momentum":   {"score": 0.5,  "confidence": 0.8, "components": {"x": 1}},
                "iv_rank":    {"score": -0.2, "confidence": 0.6, "components": {}},
            },
            "MSFT": {
                "momentum":   {"score": 0.1,  "confidence": 0.9, "components": {"y": 2}},
            },
        }
        _persist_scores(db, scores, ts)

        rows = db.execute(
            "SELECT signal_name, symbol, ts, score, confidence, components_json "
            "FROM signal_scores ORDER BY symbol, signal_name"
        ).fetchall()
        assert len(rows) == 3
        # AAPL/iv_rank
        assert rows[0]["signal_name"] == "iv_rank"
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["score"] == -0.2
        assert json.loads(rows[0]["components_json"]) == {}
        # AAPL/momentum
        assert rows[1]["signal_name"] == "momentum"
        assert json.loads(rows[1]["components_json"]) == {"x": 1}
        # MSFT/momentum
        assert rows[2]["symbol"] == "MSFT"
        assert rows[2]["ts"] == ts

    def test_missing_keys_default_to_zero_and_empty(self, db):
        from signals.scorer import _persist_scores
        ts = time.time()
        # Score dict missing 'confidence' and 'components'.
        scores = {"AAPL": {"momentum": {"score": 0.3}}}
        _persist_scores(db, scores, ts)
        row = db.execute(
            "SELECT score, confidence, components_json FROM signal_scores"
        ).fetchone()
        assert row["score"] == 0.3
        assert row["confidence"] == 0.0
        assert json.loads(row["components_json"]) == {}

    def test_empty_input_is_noop(self, db):
        from signals.scorer import _persist_scores
        # Both shapes a caller might pass for "no scores".
        _persist_scores(db, {}, time.time())
        _persist_scores(db, {"AAPL": {}}, time.time())
        n = db.execute("SELECT COUNT(*) AS n FROM signal_scores").fetchone()["n"]
        assert n == 0

    def test_upsert_replaces_same_key(self, db):
        from signals.scorer import _persist_scores
        ts = 1700_000_000.0
        # First write
        _persist_scores(db, {
            "AAPL": {"momentum": {"score": 0.1, "confidence": 0.5, "components": {}}}
        }, ts)
        # Second write at the SAME (signal, symbol, ts) triggers
        # INSERT OR REPLACE — only the newer row survives.
        _persist_scores(db, {
            "AAPL": {"momentum": {"score": 0.9, "confidence": 0.99, "components": {"k": 1}}}
        }, ts)
        rows = db.execute(
            "SELECT score, confidence, components_json FROM signal_scores "
            "WHERE signal_name = 'momentum' AND symbol = 'AAPL' AND ts = ?",
            (ts,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["score"] == 0.9
        assert rows[0]["confidence"] == 0.99
        assert json.loads(rows[0]["components_json"]) == {"k": 1}

    def test_different_ts_keeps_history(self, db):
        from signals.scorer import _persist_scores
        for ts in (1700_000_000.0, 1700_000_300.0, 1700_000_600.0):
            _persist_scores(db, {
                "AAPL": {"momentum": {
                    "score": ts / 1e10, "confidence": 0.5, "components": {},
                }}
            }, ts)
        n = db.execute(
            "SELECT COUNT(*) AS n FROM signal_scores "
            "WHERE signal_name = 'momentum' AND symbol = 'AAPL'"
        ).fetchone()["n"]
        assert n == 3
