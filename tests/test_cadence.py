"""Tests for core.runtime.cadence — session-aware sleep duration."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from core.runtime import cadence


_ET = ZoneInfo("America/New_York")


def _et(year, month, day, hour, minute):
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


# 2026-04-28 is a Tuesday.


@pytest.mark.parametrize("dt,expected", [
    (_et(2026, 4, 28, 9, 30),  "regular"),
    (_et(2026, 4, 28, 12, 0),  "regular"),
    (_et(2026, 4, 28, 15, 59), "regular"),
    (_et(2026, 4, 28, 16, 0),  "active_extended"),
    (_et(2026, 4, 28, 9, 29),  "active_extended"),
    (_et(2026, 4, 28, 7, 0),   "active_extended"),
    (_et(2026, 4, 28, 17, 59), "active_extended"),
    (_et(2026, 4, 28, 18, 0),  "quiet_extended"),
    (_et(2026, 4, 28, 19, 59), "quiet_extended"),
    (_et(2026, 4, 28, 4, 0),   "quiet_extended"),
    (_et(2026, 4, 28, 6, 59),  "quiet_extended"),
    (_et(2026, 4, 28, 20, 0),  "overnight"),
    (_et(2026, 4, 28, 3, 59),  "overnight"),
    (_et(2026, 4, 28, 0, 0),   "overnight"),
])
def test_session_label_weekday(dt, expected):
    assert cadence.session_label(dt) == expected


@pytest.mark.parametrize("dt", [
    _et(2026, 5, 2, 12, 0),
    _et(2026, 5, 3, 9, 30),
    _et(2026, 5, 3, 12, 0),
])
def test_weekend_is_always_overnight(dt):
    assert cadence.session_label(dt) == "overnight"


def test_cadence_seconds_matches_label():
    regular         = _et(2026, 4, 28, 12, 0)
    active_extended = _et(2026, 4, 28, 17, 0)
    quiet_extended  = _et(2026, 4, 28, 19, 0)
    overnight       = _et(2026, 4, 28, 2, 0)
    assert cadence.cadence_seconds(regular)         == cadence.CADENCE_REGULAR_S
    assert cadence.cadence_seconds(active_extended) == cadence.CADENCE_ACTIVE_EXTENDED_S
    assert cadence.cadence_seconds(quiet_extended)  == cadence.CADENCE_QUIET_EXTENDED_S
    assert cadence.cadence_seconds(overnight)       == cadence.CADENCE_OVERNIGHT_S


def test_cadence_seconds_naive_datetime_assumed_et():
    naive = datetime(2026, 4, 28, 12, 0)
    assert cadence.session_label(naive) == "regular"


def test_cadence_seconds_default_now_returns_known_tier():
    val = cadence.cadence_seconds()
    assert val in (
        cadence.CADENCE_REGULAR_S,
        cadence.CADENCE_ACTIVE_EXTENDED_S,
        cadence.CADENCE_QUIET_EXTENDED_S,
        cadence.CADENCE_OVERNIGHT_S,
    )


def test_extended_alias_preserved_for_backcompat():
    assert cadence.CADENCE_EXTENDED_S == cadence.CADENCE_ACTIVE_EXTENDED_S


# ── base_universe_every_n_rounds ──────────────────────────────────────


def test_base_every_n_regular_skips_two_of_three():
    n = cadence.base_universe_every_n_rounds(_et(2026, 4, 28, 12, 0))
    assert n == 3


def test_base_every_n_extended_runs_every_round():
    assert cadence.base_universe_every_n_rounds(_et(2026, 4, 28, 17, 0)) == 1
    assert cadence.base_universe_every_n_rounds(_et(2026, 4, 28, 19, 0)) == 1
    assert cadence.base_universe_every_n_rounds(_et(2026, 4, 28, 2, 0))  == 1


def test_base_every_n_weekend_runs_every_round():
    assert cadence.base_universe_every_n_rounds(_et(2026, 5, 2, 12, 0)) == 1
