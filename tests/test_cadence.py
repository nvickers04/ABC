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
    (_et(2026, 4, 28, 9, 30),  "regular"),     # exact open
    (_et(2026, 4, 28, 12, 0),  "regular"),     # midday
    (_et(2026, 4, 28, 15, 59), "regular"),     # last minute regular
    (_et(2026, 4, 28, 16, 0),  "extended"),    # exact close → postmarket
    (_et(2026, 4, 28, 9, 29),  "extended"),    # 1 min pre-open
    (_et(2026, 4, 28, 4, 0),   "extended"),    # premarket open
    (_et(2026, 4, 28, 19, 59), "extended"),    # last minute postmarket
    (_et(2026, 4, 28, 20, 0),  "overnight"),   # postmarket close
    (_et(2026, 4, 28, 3, 59),  "overnight"),   # 1 min before premarket
    (_et(2026, 4, 28, 0, 0),   "overnight"),   # midnight
])
def test_session_label_weekday(dt, expected):
    assert cadence.session_label(dt) == expected


@pytest.mark.parametrize("dt", [
    _et(2026, 5, 2, 12, 0),   # Saturday noon
    _et(2026, 5, 3, 9, 30),   # Sunday "regular open" → still overnight
    _et(2026, 5, 3, 12, 0),
])
def test_weekend_is_always_overnight(dt):
    assert cadence.session_label(dt) == "overnight"


def test_cadence_seconds_matches_label():
    regular = _et(2026, 4, 28, 12, 0)
    extended = _et(2026, 4, 28, 18, 0)
    overnight = _et(2026, 4, 28, 2, 0)
    assert cadence.cadence_seconds(regular) == cadence.CADENCE_REGULAR_S
    assert cadence.cadence_seconds(extended) == cadence.CADENCE_EXTENDED_S
    assert cadence.cadence_seconds(overnight) == cadence.CADENCE_OVERNIGHT_S


def test_cadence_seconds_naive_datetime_assumed_et():
    """Naive datetimes are treated as ET (no surprise UTC interpretation)."""
    naive = datetime(2026, 4, 28, 12, 0)  # noon, no tz
    assert cadence.session_label(naive) == "regular"


def test_cadence_seconds_default_now_returns_int():
    """Smoke test: callable with no args returns one of the three tiers."""
    val = cadence.cadence_seconds()
    assert val in (cadence.CADENCE_REGULAR_S,
                   cadence.CADENCE_EXTENDED_S,
                   cadence.CADENCE_OVERNIGHT_S)
