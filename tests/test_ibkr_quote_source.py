"""
Tests for IBKRQuoteSource — line-budget enforcement, LRU eviction,
fail-honest semantics, and snapshot fallback.

These tests do NOT touch real IBKR.  They mock the connector and ib_insync
ticker objects.  Streaming behavior is observed via the in-memory _streams
OrderedDict; we never actually open sockets.
"""
from __future__ import annotations

import asyncio
import math
import sys
import types
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest


# ============================================================
# Fixtures: stub ib_insync.contract.Stock so we don't need real ib_insync
# ============================================================

@pytest.fixture(autouse=True)
def _stub_ib_insync(monkeypatch):
    """Provide a minimal ib_insync.contract.Stock if real lib isn't installed,
    or wrap the real one to be inert."""
    try:
        import ib_insync.contract  # noqa
        # Real lib present — leave it alone, our mock connector returns
        # mock contracts anyway.
        yield
        return
    except Exception:
        pass
    fake_contract_mod = types.ModuleType("ib_insync.contract")

    class _Stock:
        def __init__(self, sym, exchange, currency):
            self.symbol = sym
            self.exchange = exchange
            self.currency = currency

        def __repr__(self):
            return f"<Stock {self.symbol}>"

    fake_contract_mod.Stock = _Stock
    fake_ib_insync = types.ModuleType("ib_insync")
    fake_ib_insync.contract = fake_contract_mod
    monkeypatch.setitem(sys.modules, "ib_insync", fake_ib_insync)
    monkeypatch.setitem(sys.modules, "ib_insync.contract", fake_contract_mod)
    yield


# ============================================================
# Mock IBKR connector + ticker
# ============================================================

@dataclass
class _FakeTicker:
    """Mimics enough of ib_insync.Ticker for IBKRQuoteSource."""
    contract: Any
    last: float = float("nan")
    bid: float = float("nan")
    ask: float = float("nan")
    volume: float = float("nan")
    high: float = float("nan")
    low: float = float("nan")


class _FakeIB:
    """Mimics enough of ib_insync.IB for IBKRQuoteSource."""

    def __init__(self):
        self.subscribed: List[Any] = []  # list of contracts currently subscribed
        self.cancelled: List[Any] = []
        self.market_data_type_calls: List[int] = []
        self.qualify_results = True  # set False to simulate qualify failure
        # Map symbol -> tick fields the next subscription should produce
        self.tick_data: dict[str, dict] = {}
        # Optional: how long _await_tick should wait before tick_data takes effect
        self.tick_delay_secs: float = 0.0
        self._scheduled_tick_targets: list = []

    def reqMarketDataType(self, t: int):
        self.market_data_type_calls.append(t)

    async def qualifyContractsAsync(self, contract):
        return [contract] if self.qualify_results else []

    def reqMktData(self, contract, generic_ticks: str, snapshot: bool, regulatory: bool):
        ticker = _FakeTicker(contract=contract)
        # If tick_data has values for this symbol, populate immediately
        sym = getattr(contract, "symbol", None)
        if sym and sym in self.tick_data and self.tick_delay_secs <= 0:
            for k, v in self.tick_data[sym].items():
                setattr(ticker, k, v)
        elif sym and sym in self.tick_data:
            # Schedule deferred population
            self._scheduled_tick_targets.append((ticker, dict(self.tick_data[sym]), self.tick_delay_secs))
        if not snapshot:
            self.subscribed.append(contract)
        return ticker

    def cancelMktData(self, contract):
        self.cancelled.append(contract)
        if contract in self.subscribed:
            self.subscribed.remove(contract)

    async def _drive_scheduled_ticks(self):
        """Helper: in an async test, await this to apply scheduled deferred ticks."""
        import asyncio
        while self._scheduled_tick_targets:
            ticker, fields, delay = self._scheduled_tick_targets.pop(0)
            if delay > 0:
                await asyncio.sleep(delay)
            for k, v in fields.items():
                setattr(ticker, k, v)


class _FakeConnector:
    def __init__(self):
        self.ib = _FakeIB()
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected


@pytest.fixture
def connector():
    return _FakeConnector()


# ============================================================
# Helper: run async coroutine in a fresh loop
# ============================================================

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) \
        if not asyncio.get_event_loop().is_closed() \
        else asyncio.new_event_loop().run_until_complete(coro)


# ============================================================
# Tests
# ============================================================

@pytest.mark.asyncio
async def test_promote_subscribes_and_tracks_line(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    src = IBKRQuoteSource(connector, line_budget=5)

    ok = await src.promote("AAPL")
    assert ok is True
    assert src.is_streaming("AAPL")
    assert src.lines_in_use == 1
    assert len(connector.ib.subscribed) == 1


@pytest.mark.asyncio
async def test_promote_idempotent(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    src = IBKRQuoteSource(connector, line_budget=5)
    await src.promote("AAPL")
    await src.promote("AAPL")
    assert src.lines_in_use == 1
    assert len(connector.ib.subscribed) == 1


@pytest.mark.asyncio
async def test_line_budget_eviction_lru(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    src = IBKRQuoteSource(connector, line_budget=3)
    for sym in ["AAA", "BBB", "CCC"]:
        await src.promote(sym)
    assert src.lines_in_use == 3
    # Touch AAA so BBB becomes LRU
    src._touch_lru("AAA")
    await src.promote("DDD")
    assert src.lines_in_use == 3
    assert not src.is_streaming("BBB")  # evicted
    assert src.is_streaming("AAA")
    assert src.is_streaming("CCC")
    assert src.is_streaming("DDD")
    # cancelMktData was called for BBB
    assert any(getattr(c, "symbol", None) == "BBB" for c in connector.ib.cancelled)


@pytest.mark.asyncio
async def test_get_quote_returns_none_when_disconnected(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    src = IBKRQuoteSource(connector, line_budget=5)
    connector._connected = False
    q = await src.get_quote("AAPL")
    assert q is None


@pytest.mark.asyncio
async def test_get_quote_returns_none_when_qualify_fails(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    connector.ib.qualify_results = False
    src = IBKRQuoteSource(connector, line_budget=5)
    q = await src.get_quote("ZZZZ")
    assert q is None


@pytest.mark.asyncio
async def test_get_quote_reads_streaming_ticker(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    connector.ib.tick_data["AAPL"] = {"last": 200.0, "bid": 199.95, "ask": 200.05, "volume": 1234567}
    src = IBKRQuoteSource(connector, line_budget=5)
    q = await src.get_quote("AAPL")
    assert q is not None
    assert q.symbol == "AAPL"
    assert q.last == 200.0
    assert q.bid == 199.95
    assert q.ask == 200.05
    assert q.volume == 1234567


@pytest.mark.asyncio
async def test_get_quote_returns_none_when_ticker_has_no_data(connector):
    """Stream subscribes but never produces a usable tick within timeout."""
    from data.ibkr_quote_source import IBKRQuoteSource
    src = IBKRQuoteSource(connector, line_budget=5, snapshot_timeout_secs=0.3)
    # No tick_data for AAPL — _await_tick will time out
    q = await src.get_quote("AAPL")
    assert q is None


@pytest.mark.asyncio
async def test_get_quote_only_bid_or_ask_is_unusable(connector):
    """Need either last, OR both bid and ask — bid alone is not enough."""
    from data.ibkr_quote_source import IBKRQuoteSource
    connector.ib.tick_data["AAPL"] = {"bid": 199.95}  # no ask, no last
    src = IBKRQuoteSource(connector, line_budget=5, snapshot_timeout_secs=0.2)
    q = await src.get_quote("AAPL")
    assert q is None


@pytest.mark.asyncio
async def test_writer_called_on_successful_read(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    written = []
    connector.ib.tick_data["AAPL"] = {"last": 200.0, "bid": 199.9, "ask": 200.1}
    src = IBKRQuoteSource(connector, line_budget=5, latest_quote_writer=written.append)
    await src.get_quote("AAPL")
    assert len(written) == 1
    assert written[0].symbol == "AAPL"
    assert written[0].last == 200.0


@pytest.mark.asyncio
async def test_writer_failure_does_not_block_quote(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    def bad_writer(_):
        raise RuntimeError("db unavailable")
    connector.ib.tick_data["AAPL"] = {"last": 200.0, "bid": 199.9, "ask": 200.1}
    src = IBKRQuoteSource(connector, line_budget=5, latest_quote_writer=bad_writer)
    q = await src.get_quote("AAPL")
    assert q is not None  # Quote should still come back


@pytest.mark.asyncio
async def test_demote_releases_line(connector):
    from data.ibkr_quote_source import IBKRQuoteSource
    src = IBKRQuoteSource(connector, line_budget=5)
    await src.promote("AAPL")
    assert src.lines_in_use == 1
    removed = await src.demote("AAPL")
    assert removed is True
    assert src.lines_in_use == 0
    assert any(getattr(c, "symbol", None) == "AAPL" for c in connector.ib.cancelled)


@pytest.mark.asyncio
async def test_snapshot_does_not_consume_persistent_line(connector):
    """allow_promote=False uses snapshot mode, no long-lived line."""
    from data.ibkr_quote_source import IBKRQuoteSource
    connector.ib.tick_data["TSLA"] = {"last": 250.0, "bid": 249.9, "ask": 250.1}
    src = IBKRQuoteSource(connector, line_budget=5, snapshot_timeout_secs=0.3)
    q = await src.get_quote("TSLA", allow_promote=False)
    assert q is not None
    assert q.last == 250.0
    # Snapshot must not have left a persistent stream
    assert src.lines_in_use == 0
    assert not src.is_streaming("TSLA")


@pytest.mark.asyncio
async def test_imbalance_generic_tick_passed_to_reqmktdata(connector):
    """Verify genericTickList='588' is sent to IBKR for piggyback imbalances."""
    from data.ibkr_quote_source import IBKRQuoteSource

    captured = []
    orig = connector.ib.reqMktData

    def spy(contract, generic_ticks, snapshot, regulatory):
        captured.append((generic_ticks, snapshot, regulatory))
        return orig(contract, generic_ticks, snapshot, regulatory)

    connector.ib.reqMktData = spy
    src = IBKRQuoteSource(connector, line_budget=5)
    await src.promote("AAPL")
    assert captured, "reqMktData was not called"
    generic_ticks, snapshot, regulatory = captured[0]
    # '225' = auction values (open/close cross) — see DEFAULT_GENERIC_TICK_LIST.
    assert generic_ticks == "225"
    assert snapshot is False
    assert regulatory is False


def test_read_ticker_populates_auction_fields():
    """_read_ticker copies auctionImbalance/Volume/Price + regulatoryImbalance."""
    from data.ibkr_quote_source import _read_ticker

    class _T:
        last = 100.0
        bid = 99.95
        ask = 100.05
        volume = 1_000_000
        high = 101.0
        low = 99.0
        # Imbalance fields (negative = sell-side, must NOT be filtered out)
        auctionImbalance = -50_000.0
        auctionVolume = 250_000
        auctionPrice = 100.10
        regulatoryImbalance = -45_000.0

    q = _read_ticker("AAPL", _T())
    assert q is not None
    assert q.auction_imbalance == -50_000.0
    assert q.auction_volume == 250_000
    assert q.auction_price == 100.10
    assert q.regulatory_imbalance == -45_000.0


def test_read_ticker_auction_fields_default_to_none():
    """When ticker has no auction attrs, fields stay None (no error)."""
    from data.ibkr_quote_source import _read_ticker

    class _T:
        last = 100.0
        bid = 99.95
        ask = 100.05
        volume = 1_000_000
        high = 101.0
        low = 99.0
        # No auctionImbalance / auctionVolume / etc. attributes.

    q = _read_ticker("AAPL", _T())
    assert q is not None
    assert q.auction_imbalance is None
    assert q.auction_volume is None
    assert q.auction_price is None
    assert q.regulatory_imbalance is None


def test_read_ticker_auction_imbalance_nan_becomes_none():
    """ib_insync uses NaN for 'no value yet' — must normalize to None."""
    from data.ibkr_quote_source import _read_ticker

    class _T:
        last = 100.0
        bid = 99.95
        ask = 100.05
        volume = 1_000_000
        high = 101.0
        low = 99.0
        auctionImbalance = float("nan")
        auctionVolume = float("nan")
        auctionPrice = float("nan")
        regulatoryImbalance = float("nan")

    q = _read_ticker("AAPL", _T())
    assert q is not None
    assert q.auction_imbalance is None
    assert q.auction_volume is None
    assert q.auction_price is None
    assert q.regulatory_imbalance is None


def test_read_ticker_auction_imbalance_zero_preserved():
    """Zero imbalance is meaningful (book is balanced) — must not be filtered."""
    from data.ibkr_quote_source import _read_ticker

    class _T:
        last = 100.0
        bid = 99.95
        ask = 100.05
        volume = 1_000_000
        high = 101.0
        low = 99.0
        auctionImbalance = 0.0
        auctionVolume = 100_000
        auctionPrice = 100.0
        regulatoryImbalance = 0.0

    q = _read_ticker("AAPL", _T())
    assert q is not None
    assert q.auction_imbalance == 0.0
    assert q.regulatory_imbalance == 0.0


@pytest.mark.asyncio
async def test_singleton_returns_none_when_disabled(monkeypatch):
    """When IBKR_QUOTES_ENABLED is False, get_ibkr_quote_source returns None."""
    from data import ibkr_quote_source as mod
    mod.reset_ibkr_quote_source_for_tests()
    from core import config as cfg
    monkeypatch.setattr(cfg, "IBKR_QUOTES_ENABLED", False, raising=False)
    assert mod.get_ibkr_quote_source() is None
