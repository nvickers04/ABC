"""PR23 - Characterize ``signals.templates.select_template`` edge cases.

Existing tests in ``tests/test_signals.py::TestTemplateSelector`` cover
the happy path (long/short, threshold, track-record tiebreaker, quote
pricing). This file pins the uncovered branches:

  * ``iv_rank=None`` and ``atr_pct=None`` bypass their respective
    boundary checks (no filtering).
  * Out-of-band ``iv_rank`` or ``atr_pct`` filters every template ->
    returns None.
  * Quote without ``last`` or ``mid`` -> entry/target/stop stay None.
  * Quote with last but ``atr_pct=None`` -> no entry price computed.
  * Short direction: stop > entry > target.
  * Returned dict carries ``template_track_record`` and
    ``max_hold_bars=60``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

# _isolated_db (autouse) and db fixtures are provided by tests/conftest.py.


def _seed_boundaries(db, *, composite_min=0.25, composite_max=1.0,
                     iv_min=0.0, iv_max=100.0,
                     atr_min=0.0, atr_max=5.0):
    from signals.templates import TEMPLATE_DEFS, save_boundaries
    for tname in TEMPLATE_DEFS:
        save_boundaries(db, tname, {
            "composite_min": composite_min,
            "composite_max": composite_max,
            "iv_rank_min": iv_min,
            "iv_rank_max": iv_max,
            "atr_pct_min": atr_min,
            "atr_pct_max": atr_max,
        })


# ── Tests ─────────────────────────────────────────────────────


class TestSelectTemplateEdgeCases:
    def test_none_iv_rank_bypasses_iv_filter(self, db):
        """When iv_rank is None, the iv_rank_min/max boundaries are
        ignored entirely (template passes even with iv_min=50)."""
        from signals.templates import select_template
        _seed_boundaries(db, iv_min=50.0, iv_max=100.0)
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=None,           # <-- bypass
            atr_pct=2.0,
        )
        assert result is not None  # not filtered

    def test_none_atr_bypasses_atr_filter(self, db):
        from signals.templates import select_template
        _seed_boundaries(db, atr_min=10.0, atr_max=20.0)  # would block atr=2
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=50.0,
            atr_pct=None,           # <-- bypass
        )
        assert result is not None

    def test_iv_rank_out_of_bounds_filters_all(self, db):
        from signals.templates import select_template
        _seed_boundaries(db, iv_min=80.0, iv_max=100.0)
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=10.0,           # below all templates' iv_min
            atr_pct=2.0,
        )
        assert result is None

    def test_atr_out_of_bounds_filters_all(self, db):
        from signals.templates import select_template
        _seed_boundaries(db, atr_min=0.0, atr_max=1.0)
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=50.0,
            atr_pct=10.0,           # above all templates' atr_max
        )
        assert result is None

    def test_composite_above_max_filters(self, db):
        """Templates have composite_max=0.5 -> abs(0.9) > 0.5 -> filter."""
        from signals.templates import select_template
        _seed_boundaries(db, composite_min=0.25, composite_max=0.5)
        result = select_template(
            db, "AAPL",
            composite_score=0.9,
            iv_rank=50.0,
            atr_pct=2.0,
        )
        assert result is None

    def test_quote_without_last_or_mid_no_prices(self, db):
        from signals.templates import select_template
        _seed_boundaries(db)
        # Quote with neither attribute -> entry stays None.
        quote = SimpleNamespace(bid=99.0, ask=101.0)
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=50.0,
            atr_pct=2.0,
            quote=quote,
        )
        assert result is not None
        assert result["entry_price"] is None
        assert result["target_price"] is None
        assert result["stop_price"] is None

    def test_quote_with_last_but_no_atr_no_prices(self, db):
        from signals.templates import select_template
        _seed_boundaries(db)
        quote = SimpleNamespace(last=100.0, mid=100.0)
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=50.0,
            atr_pct=None,           # entry computation gated on atr_pct
            quote=quote,
        )
        assert result is not None
        assert result["entry_price"] is None
        assert result["target_price"] is None
        assert result["stop_price"] is None

    def test_short_direction_stop_above_entry(self, db):
        from signals.templates import select_template
        _seed_boundaries(db)
        quote = SimpleNamespace(last=100.0, mid=100.0)
        result = select_template(
            db, "AAPL",
            composite_score=-0.7,   # short
            iv_rank=50.0,
            atr_pct=2.0,
            quote=quote,
        )
        assert result is not None
        assert result["direction"] == "short"
        assert result["entry_price"] == 100.0
        # Short: target below entry, stop above entry.
        assert result["target_price"] < result["entry_price"]
        assert result["stop_price"] > result["entry_price"]

    def test_long_target_stop_ratio_uses_atr(self, db):
        """Long: target = entry + 1.5*atr_$, stop = entry - 1.0*atr_$."""
        from signals.templates import select_template
        _seed_boundaries(db)
        quote = SimpleNamespace(last=100.0, mid=100.0)
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=50.0,
            atr_pct=2.0,            # 2% -> $2
            quote=quote,
        )
        assert result is not None
        # atr_$=2, target=100+1.5*2=103, stop=100-1*2=98
        assert result["target_price"] == pytest.approx(103.0, abs=1e-6)
        assert result["stop_price"] == pytest.approx(98.0, abs=1e-6)

    def test_result_carries_track_record_and_max_hold(self, db):
        from signals.templates import select_template
        _seed_boundaries(db)
        result = select_template(
            db, "AAPL",
            composite_score=0.7,
            iv_rank=50.0,
            atr_pct=2.0,
        )
        assert result is not None
        assert result["max_hold_bars"] == 60
        assert "template_track_record" in result
        assert isinstance(result["template_track_record"], dict)
        # legs_json is None at this stage (filled later for options).
        assert result["legs_json"] is None

    def test_threshold_exactly_zero_returns_none(self, db):
        """abs(composite=0) < threshold -> None regardless of regime."""
        from signals.templates import select_template
        _seed_boundaries(db)
        result = select_template(
            db, "AAPL",
            composite_score=0.0,
            iv_rank=50.0,
            atr_pct=2.0,
        )
        assert result is None
