"""Tests for core.runtime.intuition — naive attention scorer + render."""

from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import memory
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "_DB_PATH", db_path)
    monkeypatch.setattr(memory, "_connection", None)
    monkeypatch.setattr(memory, "_calibration_version", 0)
    memory._pending_graduated_params.clear()
    memory._pending_order_context.clear()
    memory.init_db()
    yield
    if memory._connection:
        memory._connection.close()
    monkeypatch.setattr(memory, "_connection", None)


@pytest.fixture
def db():
    from memory import get_db
    return get_db()


def _seed_signal_score(db, signal_name, symbol, score, ts=None, confidence=1.0):
    ts = ts if ts is not None else time.time()
    db.execute(
        "INSERT OR REPLACE INTO signal_scores "
        "(signal_name, symbol, ts, score, confidence, components_json) "
        "VALUES (?, ?, ?, ?, ?, NULL)",
        (signal_name, symbol, ts, score, confidence),
    )
    db.commit()


def _seed_weight(db, signal_name, weight, n_eff=10.0, category="momentum"):
    db.execute(
        "INSERT OR REPLACE INTO signal_weights "
        "(signal_name, weight, n_eff, category, updated_ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (signal_name, weight, n_eff, category, time.time()),
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


def _seed_signal_returns(db, signal_name, pairs, *, horizon_bars=10):
    """Seed enough rows for IC computation. ``pairs`` is [(score, ret), ...]."""
    base_ts = time.time() - 100.0
    for i, (s, r) in enumerate(pairs):
        db.execute(
            "INSERT OR REPLACE INTO signal_returns "
            "(signal_name, symbol, ts, score_at_entry, forward_return, r_value, horizon_bars) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (signal_name, f"S{i}", base_ts + i, s, r, r, horizon_bars),
        )
    db.commit()


# ── compute_attention_scores ────────────────────────────────────


def test_compute_returns_empty_with_no_composites(db):
    from core.runtime.intuition import compute_attention_scores
    assert compute_attention_scores(db) == {}


def test_compute_returns_score_for_seeded_symbol(db):
    """Single symbol, single signal with positive IC → positive trust."""
    from core.runtime.intuition import compute_attention_scores
    # Need SIGNAL_REGISTRY to include the signal name; use a known one.
    from signals.combiner import SIGNAL_REGISTRY  # populates only if imports happened
    # Pick any registered signal name; if none, skip gracefully.
    if not SIGNAL_REGISTRY:
        # Trigger registration by importing one signal module.
        import signals.momentum  # noqa: F401
    sig_name = next(iter(SIGNAL_REGISTRY)) if SIGNAL_REGISTRY else "momentum"

    _seed_composite(db, "UNH", 0.5)
    _seed_signal_score(db, sig_name, "UNH", 0.8)
    _seed_weight(db, sig_name, 1.0)
    # Strong positive IC: monotonic (score, return) pairs.
    pairs = [(float(i), float(i)) for i in range(20)]
    _seed_signal_returns(db, sig_name, pairs)

    scored = compute_attention_scores(db)
    assert "UNH" in scored
    info = scored["UNH"]
    assert info["composite"] == pytest.approx(0.5)
    # |composite| * trust + |novelty| ; trust = |0.8| * |~1.0| * 1.0 ≈ 0.8
    assert info["trust"] > 0.0
    assert info["score"] > 0.0
    # First round → no prior → novelty = 0
    assert info["novelty"] == 0.0
    drivers = info["drivers"]
    assert len(drivers) == 1
    assert drivers[0][0] == sig_name


def test_novelty_is_delta_from_prior_composite(db):
    from core.runtime.intuition import compute_attention_scores
    now = time.time()
    _seed_composite(db, "AAPL", 0.20, ts=now - 600.0)  # prior
    _seed_composite(db, "AAPL", 0.55, ts=now)          # latest
    info = compute_attention_scores(db)["AAPL"]
    assert info["composite"] == pytest.approx(0.55)
    assert info["novelty"] == pytest.approx(0.35)


def test_universe_filter_restricts_results(db):
    from core.runtime.intuition import compute_attention_scores
    _seed_composite(db, "UNH", 0.5)
    _seed_composite(db, "AAPL", 0.6)
    out = compute_attention_scores(db, universe=["aapl"])
    assert set(out.keys()) == {"AAPL"}


def test_signal_with_no_weight_or_no_ic_does_not_contribute(db):
    from core.runtime.intuition import compute_attention_scores
    _seed_composite(db, "UNH", 0.5)
    _seed_signal_score(db, "ghost_signal", "UNH", 0.9)
    # No weight, no signal_returns for it.
    info = compute_attention_scores(db)["UNH"]
    assert info["trust"] == 0.0
    assert info["drivers"] == []
    # Score reduces to |composite|*0 + |0| = 0.
    assert info["score"] == 0.0


# ── render_intuition_block ──────────────────────────────────────


def test_render_empty_when_no_data(db):
    from core.runtime.intuition import render_intuition_block
    assert render_intuition_block(db) == ""


def test_render_top_n_with_drivers(db):
    from core.runtime.intuition import render_intuition_block
    # Two symbols with composites; no IC needed for render to produce
    # something — it'll just show empty drivers.
    _seed_composite(db, "UNH", 0.80)
    _seed_composite(db, "AAPL", 0.10)
    out = render_intuition_block(db, top_n=5, top_drivers=3)
    assert out.startswith("INTUITION")
    assert "UNH" in out
    assert "AAPL" in out
    # UNH (higher |composite|) ranks first.
    unh_pos = out.find("UNH")
    aapl_pos = out.find("AAPL")
    assert 0 < unh_pos < aapl_pos


def test_render_drivers_ordered_by_contribution(db):
    from core.runtime.intuition import render_intuition_block
    from signals.combiner import SIGNAL_REGISTRY
    # Register two signals.
    import signals.momentum  # noqa: F401
    import signals.gap  # noqa: F401
    names = [n for n in ("momentum", "gap") if n in SIGNAL_REGISTRY]
    if len(names) < 2:
        pytest.skip("registry didn't populate expected signals")
    a, b = names[0], names[1]

    _seed_composite(db, "UNH", 0.5)
    # a: small score, b: large score → b should rank higher.
    _seed_signal_score(db, a, "UNH", 0.10)
    _seed_signal_score(db, b, "UNH", 0.90)
    _seed_weight(db, a, 1.0)
    _seed_weight(db, b, 1.0)
    pairs = [(float(i), float(i)) for i in range(20)]
    _seed_signal_returns(db, a, pairs)
    _seed_signal_returns(db, b, pairs)

    out = render_intuition_block(db, top_drivers=2)
    # b should appear before a in the drivers list for UNH.
    pos_a = out.find(a + "(")
    pos_b = out.find(b + "(")
    assert 0 <= pos_b < pos_a
