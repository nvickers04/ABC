"""Anti-chop fitness adjustment in template boundary evaluation."""

from __future__ import annotations

from signals.template_evolution import _evaluate_boundaries


def _row(comp: float, fwd: float) -> dict:
    return {"composite_score": comp, "forward_return": fwd, "iv_rank": None}


def test_chop_penalty_when_edge_ratio_low():
    """Many tiny moves + fat tails → lower search_fitness than same win rate with fatter body."""
    bounds = {
        "composite_min": 0.2,
        "composite_max": 1.0,
        "iv_rank_min": 0.0,
        "iv_rank_max": 100.0,
    }
    chop_data = [_row(0.5, x) for x in [0.0002, -0.0002] * 9] + [_row(0.5, 0.8), _row(0.5, -0.8)]
    stable_data = [_row(0.5, x) for x in [0.02, -0.015, 0.018, -0.012] * 4]

    m_chop = _evaluate_boundaries("t", bounds, chop_data)
    m_stable = _evaluate_boundaries("t", bounds, stable_data)
    assert m_chop["trades"] >= 10
    assert m_stable["trades"] >= 10
    assert m_chop["search_fitness"] < m_stable["search_fitness"]
