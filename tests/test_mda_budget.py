"""Unit tests for MDA adaptive pacing helpers."""

from core.runtime import mda_budget


def test_credit_fraction_none_when_missing_headers():
    assert mda_budget.credit_fraction({}) is None
    assert mda_budget.credit_fraction({"mda_credits_remaining": 5000}) is None


def test_credit_fraction_ratio():
    assert mda_budget.credit_fraction(
        {"mda_credits_remaining": 25000, "mda_credits_limit": 100000}
    ) == 0.25


def test_cadence_multiplier_tiers():
    u_crit = {"mda_credits_remaining": 5000, "mda_credits_limit": 100000}
    assert mda_budget.mda_cadence_multiplier(u_crit) == 4.0

    u_low = {"mda_credits_remaining": 15000, "mda_credits_limit": 100000}
    assert mda_budget.mda_cadence_multiplier(u_low) == 2.0

    u_ok = {"mda_credits_remaining": 25000, "mda_credits_limit": 100000}
    assert mda_budget.mda_cadence_multiplier(u_ok) == 1.0


def test_skip_subdaily_unknown_fraction_false():
    assert mda_budget.should_skip_subdaily_candle_fetches({}) is False


def test_skip_subdaily_below_threshold():
    u = {"mda_credits_remaining": 10000, "mda_credits_limit": 100000}
    assert mda_budget.should_skip_subdaily_candle_fetches(u) is True
