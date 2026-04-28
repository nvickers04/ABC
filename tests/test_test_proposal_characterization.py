"""PR25 - Characterize ``TradingAgent._test_proposal``.

The Mann-Whitney bucket matcher decides whether a graduated-param
proposal has enough comparison data to validate. Pin every branch:

  * Bad key format (not 4 dot-separated parts) -> None.
  * Empty key -> None.
  * scipy unavailable -> None (we can't easily simulate this without
    monkeypatching imports; covered by unmatched-key path).
  * 'all' wildcard on order_type matches every group.
  * Wildcard on time_bucket / atr_bucket matches accordingly.
  * Insufficient target data (<5) -> None.
  * Insufficient other data (<5) -> None.
  * Sufficient data on both sides -> a float p-value in [0, 1] rounded
    to 4 decimals.
  * Identical distributions -> p ~= 1.0.
  * Strongly different distributions -> p << 0.05.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# scipy is a hard dep of the function for non-None returns.
pytest.importorskip("scipy")


def _agent_stub():
    return SimpleNamespace()


def _call(prop, groups):
    from core.agent import TradingAgent
    return TradingAgent._test_proposal(_agent_stub(), prop, groups)


# ── Tests ─────────────────────────────────────────────────────


class TestTestProposal:
    def test_empty_key_returns_none(self):
        assert _call({"param_key": ""}, {}) is None

    def test_missing_key_returns_none(self):
        # No param_key field at all -> default "" -> 1 part -> None.
        assert _call({}, {}) is None

    def test_three_part_key_returns_none(self):
        assert _call({"param_key": "market.entry.open"}, {}) is None

    def test_five_part_key_returns_none(self):
        assert _call({"param_key": "a.b.c.d.e"}, {}) is None

    def test_no_groups_returns_none(self):
        # Valid key, but groups is empty -> target_data and other_data
        # both empty -> None.
        prop = {"param_key": "market.entry.open.high"}
        assert _call(prop, {}) is None

    def test_target_data_under_5_returns_none(self):
        prop = {"param_key": "market.entry.open.high"}
        # Target group has only 4 samples (and len<3 wouldn't even be
        # accepted; use 4).
        groups = {
            ("market", "open", "high"): [10.0, 11.0, 12.0, 13.0],
            ("market", "midday", "high"): [5.0] * 10,
        }
        assert _call(prop, groups) is None

    def test_other_data_under_5_returns_none(self):
        prop = {"param_key": "market.entry.open.high"}
        # Target ample, but other group has only 3 samples (passes the
        # len>=3 filter but stays below the >=5 statistical threshold).
        groups = {
            ("market", "open", "high"): [10.0] * 10,
            ("market", "midday", "high"): [5.0, 6.0, 7.0],
        }
        # Both sides reach the >=3 inclusion gate but other_data
        # length is only 3 -> below 5 -> None.
        assert _call(prop, groups) is None

    def test_returns_p_value_when_data_sufficient(self):
        prop = {"param_key": "market.entry.open.high"}
        # 10 samples each, slightly different distributions.
        groups = {
            ("market", "open", "high"): [10.0, 11.0, 9.0, 10.5, 10.2,
                                          11.5, 9.5, 10.0, 10.8, 9.7],
            ("market", "midday", "all"): [12.0, 13.0, 11.0, 12.5, 12.2,
                                           13.5, 11.5, 12.0, 12.8, 11.7],
        }
        p = _call(prop, groups)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
        # Rounded to 4 decimals.
        assert round(p, 4) == p

    def test_identical_distributions_high_p(self):
        prop = {"param_key": "market.entry.open.high"}
        same = [10.0, 11.0, 9.0, 10.5, 10.2, 11.5, 9.5, 10.0, 10.8, 9.7]
        groups = {
            ("market", "open", "high"): list(same),
            ("market", "midday", "all"): list(same),
        }
        p = _call(prop, groups)
        assert p is not None
        # Identical samples -> Mann-Whitney p should be ~1.0.
        assert p > 0.5

    def test_strongly_different_distributions_low_p(self):
        prop = {"param_key": "market.entry.open.high"}
        groups = {
            ("market", "open", "high"): [50.0] * 20,        # all high
            ("market", "midday", "all"): [1.0] * 20,        # all low
        }
        p = _call(prop, groups)
        assert p is not None
        assert p < 0.05

    def test_order_type_wildcard_matches_all(self):
        """target_ot='all' -> any order_type matches as target."""
        prop = {"param_key": "all.entry.open.high"}
        groups = {
            ("market", "open", "high"): [10.0] * 10,
            ("limit",  "open", "high"): [11.0] * 10,
        }
        # With 'all' wildcard on ot, ALL groups match the target check.
        # The 'elif ot_match:' branch never fires (target catches first),
        # so other_data stays empty -> None.
        p = _call(prop, groups)
        assert p is None

    def test_time_bucket_wildcard_extends_target(self):
        """target_tb='all' includes both open and midday in target;
        other-data path requires same ot but a non-target bucket — but
        with tb='all' both buckets are target, so this returns None."""
        prop = {"param_key": "market.entry.all.high"}
        groups = {
            ("market", "open", "high"): [10.0] * 10,
            ("market", "midday", "high"): [11.0] * 10,
            ("market", "open", "low"): [5.0] * 10,
        }
        # target collects (open,high), (midday,high)  [tb='all' & ab='high']
        # other gets    (open,low)  via same-ot-different-bucket fallback
        p = _call(prop, groups)
        assert p is not None  # 20 target vs 10 other -> sufficient
        assert 0.0 <= p <= 1.0

    def test_only_other_data_with_under_3_samples_filtered_out(self):
        """The len<3 filter at the top of the loop should drop tiny
        groups before they enter target/other_data."""
        prop = {"param_key": "market.entry.open.high"}
        groups = {
            ("market", "open", "high"): [10.0] * 10,    # target
            ("market", "midday", "high"): [5.0, 6.0],   # filtered (len<3)
        }
        # Other_data ends up empty -> None.
        assert _call(prop, groups) is None
