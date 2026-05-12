"""Unit tests for MDA adaptive pacing helpers."""

import time

import pytest

from core.runtime import mda_budget


@pytest.fixture(autouse=True)
def _reset_mda_burn_state():
    mda_budget._prev_mda_rem = None  # type: ignore[attr-defined]
    mda_budget._prev_mda_ts = 0.0  # type: ignore[attr-defined]
    yield


def test_credit_fraction_none_when_missing_headers():
    assert mda_budget.credit_fraction({}) is None
    assert mda_budget.credit_fraction({"mda_credits_remaining": 5000}) is None


def test_credit_fraction_ratio():
    assert mda_budget.credit_fraction(
        {"mda_credits_remaining": 25000, "mda_credits_limit": 100000}
    ) == 0.25


def test_cadence_multiplier_tiers():
    u_crit = {"mda_credits_remaining": 17000, "mda_credits_limit": 100000}
    assert mda_budget.mda_cadence_multiplier(u_crit) == 4.0

    u_low = {"mda_credits_remaining": 25000, "mda_credits_limit": 100000}
    assert mda_budget.mda_cadence_multiplier(u_low) == 2.0

    u_soft = {"mda_credits_remaining": 40000, "mda_credits_limit": 100000}
    assert mda_budget.mda_cadence_multiplier(u_soft) == 1.35

    u_ok = {"mda_credits_remaining": 50000, "mda_credits_limit": 100000}
    assert mda_budget.mda_cadence_multiplier(u_ok) == 1.0


def test_skip_subdaily_unknown_fraction_false():
    assert mda_budget.should_skip_subdaily_candle_fetches({}) is False


def test_skip_subdaily_below_threshold():
    u = {"mda_credits_remaining": 35000, "mda_credits_limit": 100000}
    assert mda_budget.should_skip_subdaily_candle_fetches(u) is True


def test_seconds_until_reset_epoch():
    t0 = int(time.time()) + 7200
    u = {"mda_credits_reset_epoch": t0}
    sec = mda_budget.seconds_until_mda_reset(u)
    assert sec is not None
    assert 7100 < sec < 7300


def test_total_sleep_multiplier_first_sample_no_burn_spike():
    """First observation has no prior credits → burn leg stays 1.0."""
    t0 = int(time.time()) + 50000
    u = {
        "mda_credits_remaining": 60000,
        "mda_credits_limit": 100000,
        "mda_credits_reset_epoch": t0,
    }
    m, note = mda_budget.mda_total_sleep_multiplier(u)
    assert 1.0 <= m <= 8.0
    assert note is None


def test_runway_stretches_when_low_credits_and_reset_soon():
    t0 = int(time.time()) + 1800  # 30 min
    u = {
        "mda_credits_remaining": 22000,
        "mda_credits_limit": 100000,
        "mda_credits_reset_epoch": t0,
    }
    rwy = mda_budget.mda_runway_multiplier(u)
    assert rwy >= 1.6
