"""Tests for core.runtime.intuition — naive attention scorer + render."""

from __future__ import annotations

import time

import pytest


# Local _isolated_db removed — uses shared robust conftest version (with .env + graceful Postgres skip).

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
