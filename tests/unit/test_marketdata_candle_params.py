"""MarketData.app candle URL + countback semantics (intraday vs daily)."""

from __future__ import annotations

from data.marketdata_client import (
    intraday_countback_from_calendar_days,
    normalize_mda_candle_resolution,
)


def test_normalize_resolution_internal_to_mda_token():
    assert normalize_mda_candle_resolution("5min") == "5"
    assert normalize_mda_candle_resolution("1min") == "1"
    assert normalize_mda_candle_resolution("1h") == "H"
    assert normalize_mda_candle_resolution("D") == "D"
    assert normalize_mda_candle_resolution("daily") == "D"


def test_intraday_countback_scales_with_calendar_days_not_equal_to_raw_five():
    """``days_back=5`` must not mean five *bars* for 5-minute candles."""
    cb = intraday_countback_from_calendar_days("5", 5)
    assert cb >= 200, f"expected multi-session bar count, got {cb}"


def test_one_day_lookback_one_minute_has_more_bars_than_five_minute():
    cb5 = intraday_countback_from_calendar_days("5", 1)
    cb1 = intraday_countback_from_calendar_days("1", 1)
    assert cb1 > cb5
