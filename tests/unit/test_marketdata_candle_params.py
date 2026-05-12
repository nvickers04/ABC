"""MarketData.app candle URL + countback semantics (intraday vs daily)."""

from __future__ import annotations

import json

import httpx

from data.marketdata_client import (
    intraday_countback_from_calendar_days,
    normalize_mda_candle_resolution,
    _normalize_mda_api_key,
    _format_mda_denied_body,
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


def test_normalize_mda_api_key_strips_bearer_and_quotes():
    assert _normalize_mda_api_key("Bearer abc123") == "abc123"
    assert _normalize_mda_api_key("  bearer  tok  ") == "tok"
    assert _normalize_mda_api_key('"secret"') == "secret"
    assert _normalize_mda_api_key(None) is None


def test_format_mda_denied_body_includes_multi_ip_fields():
    req = httpx.Request("GET", "https://api.marketdata.app/v1/stocks/prices/AAPL/")
    body = {
        "s": "error",
        "errmsg": "Access denied test",
        "authorizedIP": "1.1.1.1",
        "blockedIP": "2.2.2.2",
        "troubleshootingGuide": "https://example.com/help",
    }
    resp = httpx.Response(403, request=req, content=json.dumps(body).encode())
    s = _format_mda_denied_body(resp)
    assert "Access denied test" in s
    assert "1.1.1.1" in s and "2.2.2.2" in s
    assert "example.com/help" in s
