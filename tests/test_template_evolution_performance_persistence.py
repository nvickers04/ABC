"""Characterize template_performance persistence in evolution rounds."""

from __future__ import annotations

import asyncio


def test_evolution_persists_baseline_performance_without_mutation_winner(db, monkeypatch):
    """template_performance should populate even when no mutation improves fitness."""
    import signals.template_evolution as te
    from signals.templates import TEMPLATE_DEFS

    template_name = "stock_market"
    monkeypatch.setattr(te, "TEMPLATE_DEFS", {template_name: TEMPLATE_DEFS[template_name]})

    fake_rows = [
        {"symbol": "AAPL", "ts": float(i), "composite_score": 0.35, "forward_return": 0.01}
        for i in range(40)
    ]
    monkeypatch.setattr(te, "_load_historical_composites", lambda _conn: fake_rows)

    # Current boundaries are valid; mutations don't beat baseline.
    base_boundaries = {
        template_name: {
            "composite_min": 0.25,
            "composite_max": 1.0,
            "iv_rank_min": 0.0,
            "iv_rank_max": 100.0,
            "atr_pct_min": 0.0,
            "atr_pct_max": 10.0,
        }
    }
    monkeypatch.setattr(te, "load_boundaries", lambda _conn: base_boundaries)
    monkeypatch.setattr(te, "_mutate_boundaries", lambda _t, current, _d: dict(current))

    def _fake_eval(_template, _bounds, _data):
        return {
            "search_fitness": 1.0,
            "trades": 12,
            "win_rate": 0.5,
            "avg_return_pct": 0.01,
            "sharpe": 0.6,
        }

    monkeypatch.setattr(te, "_evaluate_boundaries", _fake_eval)

    asyncio.run(te._evolution_round(db))

    row = db.execute(
        "SELECT template_name, trades, wins, avg_return_pct "
        "FROM template_performance WHERE template_name = ?",
        (template_name,),
    ).fetchone()
    assert row is not None
    assert row["template_name"] == template_name
    assert row["trades"] == 12
    assert row["wins"] == 6
    assert float(row["avg_return_pct"]) == 0.01
