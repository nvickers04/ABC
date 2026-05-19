"""Unit tests for execution.ibkr_connection (no live TWS)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_classify_tws_lost_codes():
    from execution.ibkr_connection import (
        ERROR_CONNECTIVITY_LOST,
        ERROR_TWS_SERVER_BROKEN,
        classify_error_code,
    )

    assert classify_error_code(ERROR_CONNECTIVITY_LOST) == "tws_lost"
    assert classify_error_code(ERROR_TWS_SERVER_BROKEN) == "tws_lost"


def test_classify_tws_restored_codes():
    from execution.ibkr_connection import (
        ERROR_CONNECTIVITY_RESTORED,
        ERROR_CONNECTIVITY_RESTORED_DATA_LOST,
        classify_error_code,
    )

    assert classify_error_code(ERROR_CONNECTIVITY_RESTORED) == "tws_restored"
    assert classify_error_code(ERROR_CONNECTIVITY_RESTORED_DATA_LOST) == "tws_restored"


def test_reconnect_backoff_caps():
    from execution.ibkr_connection import reconnect_backoff_seconds

    assert reconnect_backoff_seconds(0) == 5.0
    assert reconnect_backoff_seconds(1) == 10.0
    assert reconnect_backoff_seconds(10) == 60.0


def test_quote_source_on_connection_lost_queues_restore():
    from data.ibkr_quote_source import IBKRQuoteSource

    class _FakeConnector:
        def is_connected(self):
            return True

        class _Ib:
            def cancelMktData(self, _c):
                pass

        ib = _Ib()

    qs = IBKRQuoteSource(_FakeConnector(), line_budget=2)
    qs._streams["AAPL"] = object()
    qs._streams["MSFT"] = object()
    qs.on_connection_lost(cause="tws_restart")
    assert qs.lines_in_use == 0
    assert set(qs._restore_symbols) == {"AAPL", "MSFT"}
