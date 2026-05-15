"""Tests for tools.tools_signal_breakdown.handle_signal_breakdown."""

from __future__ import annotations

import asyncio
import time

import pytest


# Local _isolated_db removed — now uses the single robust version from tests/conftest.py
# which does .env load + reset_state + init_db + graceful pytest.skip on Postgres connection/permission errors.

@pytest.fixture
def db():
    from memory import get_db
    return get_db()


class _StubExecutor:
    pass


def _run(coro):
    return asyncio.run(coro)


def _seed_signal_score(db, signal_name, symbol, score, ts=None):
    ts = ts if ts is not None else time.time()
    db.execute(
        "INSERT OR REPLACE INTO signal_scores "
        "(signal_name, symbol, ts, score, confidence, components_json) "
        "VALUES (?, ?, ?, ?, 1.0, NULL)",
        (signal_name, symbol, ts, score),
    )
    db.commit()


def _seed_weight(db, signal_name, weight, category="momentum"):
    db.execute(
        "INSERT OR REPLACE INTO signal_weights "
        "(signal_name, weight, n_eff, category, updated_ts) "
        "VALUES (?, ?, 10.0, ?, ?)",
        (signal_name, weight, category, time.time()),
    )
    db.commit()


def _seed_composite(db, symbol, score, ts=None):
    ts = ts if ts is not None else time.time()
    db.execute(
        "INSERT OR REPLACE INTO composite_scores "
        "(symbol, ts, composite_score, signal_breakdown_json) "
        "VALUES (?, ?, ?, NULL)",
        (symbol, ts, score),
    )
    db.commit()


# ── Validation ──────────────────────────────────────────────────


def test_missing_symbol_returns_error():
    from tools.tools_signal_breakdown import handle_signal_breakdown
    out = _run(handle_signal_breakdown(_StubExecutor(), {}))
    assert "error" in out


def test_bad_symbol_returns_error():
    from tools.tools_signal_breakdown import handle_signal_breakdown
    out = _run(handle_signal_breakdown(_StubExecutor(), {"symbol": "spaces here!"}))
    assert "error" in out


def test_unknown_symbol_returns_error(db):
    from tools.tools_signal_breakdown import handle_signal_breakdown
    out = _run(handle_signal_breakdown(_StubExecutor(), {"symbol": "ZZZ"}))
    assert "error" in out
    assert "ZZZ" in out["error"]


# ── Output shape ────────────────────────────────────────────────


def test_breakdown_returns_components_sorted_by_abs_contribution(db):
    from tools.tools_signal_breakdown import handle_signal_breakdown
    _seed_composite(db, "UNH", 0.42)
    # weak: weight=0.10, score=0.10 → contrib=0.01
    _seed_signal_score(db, "weak_sig", "UNH", 0.10)
    _seed_weight(db, "weak_sig", 0.10)
    # strong: weight=0.50, score=0.80 → contrib=0.40
    _seed_signal_score(db, "strong_sig", "UNH", 0.80)
    _seed_weight(db, "strong_sig", 0.50)

    out = _run(handle_signal_breakdown(_StubExecutor(), {"symbol": "unh"}))
    assert out["symbol"] == "UNH"
    assert out["composite"] == pytest.approx(0.42)
    assert out["n_contributing"] == 2
    comps = out["components"]
    assert comps[0]["signal"] == "strong_sig"
    assert comps[1]["signal"] == "weak_sig"
    assert comps[0]["contribution"] == pytest.approx(0.40)
    assert comps[1]["contribution"] == pytest.approx(0.01)
    # Each component has the documented keys.
    expected = {"signal", "score", "weight", "ic", "category",
                "contribution", "trust"}
    for c in comps:
        assert expected <= set(c.keys())


def test_breakdown_lists_missing_data(db):
    from tools.tools_signal_breakdown import handle_signal_breakdown
    # Force at least one signal into the registry.
    import signals.momentum  # noqa: F401
    from signals.combiner import SIGNAL_REGISTRY
    if not SIGNAL_REGISTRY:
        pytest.skip("no signals registered")
    sig = next(iter(SIGNAL_REGISTRY))

    _seed_composite(db, "AAPL", 0.10)
    # No signal_scores row for AAPL → registered signal is missing.
    out = _run(handle_signal_breakdown(_StubExecutor(), {"symbol": "AAPL"}))
    assert sig in out["missing_data"]


def test_breakdown_routes_through_executor_registry(db):
    """The new handler must be reachable via the central tools_executor registry."""
    from tools.tools_executor import _REGISTRY
    assert "signal_breakdown" in _REGISTRY
    _seed_composite(db, "AAPL", 0.20)
    out = _run(_REGISTRY["signal_breakdown"](_StubExecutor(), {"symbol": "AAPL"}))
    assert out["symbol"] == "AAPL"
