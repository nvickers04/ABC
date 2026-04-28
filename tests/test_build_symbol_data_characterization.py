"""PR17 - Characterization tests for ``signals.scorer._build_symbol_data``.

Locks in the data-assembly contract that every signal depends on:

  * **Always-present keys**: ``data_provider``, ``quote``, ``candles``,
    ``spy_candles``, ``qqq_candles``, ``environment``.
  * **Optional dp-fetched keys** are each wrapped in ``try/except`` so a
    failure in one source does not break the whole bundle.
  * **Fundamentals override**: the basic ``get_fundamentals`` result is
    written to BOTH ``fundamentals`` and ``basic_fundamentals``; the
    extended result, when present, OVERWRITES ``fundamentals`` only --
    ``basic_fundamentals`` is preserved for ``size_factor`` (needs
    ``market_cap``).
  * **Tier gating**: ``iv_info`` / ``option_chain`` only present when
    ``tier >= 2``.
  * **Aliases**: ``candles_daily`` mirrors ``candles``;
    ``earnings_info`` mirrors ``earnings``.
"""

from __future__ import annotations

import pytest


class _StubDP:
    """Configurable data-provider stub.

    Each method either returns a fixed value or raises ``RuntimeError``
    based on the ``raises`` set passed at construction. This lets us
    assert that any one fetch failure leaves the rest of the bundle
    intact.
    """

    def __init__(self, *, raises: set[str] | None = None,
                 basic_fund=None, ext_fund=None, earnings=None,
                 earnings_history=None, analyst=None, institutional=None,
                 insider=None, peer=None, news=None):
        self._raises = raises or set()
        self._basic_fund = basic_fund
        self._ext_fund = ext_fund
        self._earnings = earnings
        self._earnings_history = earnings_history
        self._analyst = analyst
        self._institutional = institutional
        self._insider = insider
        self._peer = peer
        self._news = news

    def _maybe_raise(self, key: str):
        if key in self._raises:
            raise RuntimeError(f"stubbed failure for {key}")

    def get_fundamentals(self, symbol):
        self._maybe_raise("get_fundamentals")
        return self._basic_fund

    def get_extended_fundamentals(self, symbol):
        self._maybe_raise("get_extended_fundamentals")
        return self._ext_fund

    def get_earnings_info(self, symbol):
        self._maybe_raise("get_earnings_info")
        return self._earnings

    def get_earnings_history(self, symbol):
        self._maybe_raise("get_earnings_history")
        return self._earnings_history

    def get_analyst_data(self, symbol):
        self._maybe_raise("get_analyst_data")
        return self._analyst

    def get_institutional_data(self, symbol):
        self._maybe_raise("get_institutional_data")
        return self._institutional

    def get_insider_data(self, symbol):
        self._maybe_raise("get_insider_data")
        return self._insider

    def get_peer_comparison(self, symbol):
        self._maybe_raise("get_peer_comparison")
        return self._peer

    def get_news(self, symbol):
        self._maybe_raise("get_news")
        return self._news


# ── Tests ─────────────────────────────────────────────────────


class TestAlwaysPresentKeys:
    def test_core_keys_always_present(self):
        from signals.scorer import _build_symbol_data
        dp = _StubDP(raises={
            "get_fundamentals", "get_extended_fundamentals",
            "get_earnings_info", "get_earnings_history",
            "get_analyst_data", "get_institutional_data",
            "get_insider_data", "get_peer_comparison", "get_news",
        })
        quotes = {"AAPL": "Q"}
        candles_map = {"AAPL": "C"}
        d = _build_symbol_data(
            "AAPL", dp, quotes, candles_map,
            spy_candles="SPY", qqq_candles="QQQ", env={"vix": 14.0},
        )
        assert d["data_provider"] is dp
        assert d["quote"] == "Q"
        assert d["candles"] == "C"
        assert d["spy_candles"] == "SPY"
        assert d["qqq_candles"] == "QQQ"
        assert d["environment"] == {"vix": 14.0}

    def test_missing_quote_or_candles_is_none(self):
        from signals.scorer import _build_symbol_data
        dp = _StubDP(raises={
            "get_fundamentals", "get_extended_fundamentals",
            "get_earnings_info", "get_earnings_history",
            "get_analyst_data", "get_institutional_data",
            "get_insider_data", "get_peer_comparison", "get_news",
        })
        d = _build_symbol_data("AAPL", dp, {}, {})
        assert d["quote"] is None
        assert d["candles"] is None
        # spy/qqq/env default to None when caller omits them
        assert d["spy_candles"] is None
        assert d["qqq_candles"] is None
        assert d["environment"] is None


class TestFundamentalsOverride:
    def test_basic_then_extended_overrides_only_fundamentals(self):
        """``get_extended_fundamentals`` overwrites ``fundamentals`` but
        ``basic_fundamentals`` is preserved (size_factor needs market_cap
        from the basic source)."""
        from signals.scorer import _build_symbol_data
        basic = {"market_cap": 3_000_000_000}
        ext = {"fcf": 100, "roe": 0.25}
        dp = _StubDP(
            raises={
                "get_earnings_info", "get_earnings_history",
                "get_analyst_data", "get_institutional_data",
                "get_insider_data", "get_peer_comparison", "get_news",
            },
            basic_fund=basic, ext_fund=ext,
        )
        d = _build_symbol_data("AAPL", dp, {}, {})
        assert d["fundamentals"] is ext
        assert d["basic_fundamentals"] is basic

    def test_extended_falsy_does_not_override(self):
        """If ``get_extended_fundamentals`` returns a falsy value (None,
        {}), the basic result remains as ``fundamentals``."""
        from signals.scorer import _build_symbol_data
        basic = {"market_cap": 3_000_000_000}
        dp = _StubDP(
            raises={
                "get_earnings_info", "get_earnings_history",
                "get_analyst_data", "get_institutional_data",
                "get_insider_data", "get_peer_comparison", "get_news",
            },
            basic_fund=basic, ext_fund=None,
        )
        d = _build_symbol_data("AAPL", dp, {}, {})
        assert d["fundamentals"] is basic
        assert d["basic_fundamentals"] is basic

    def test_basic_failure_keeps_extended(self):
        """If the basic fetch raises but the extended one succeeds, the
        extended result should still populate ``fundamentals``. Current
        behavior: ``basic_fundamentals`` is absent in this case."""
        from signals.scorer import _build_symbol_data
        ext = {"fcf": 100}
        dp = _StubDP(
            raises={
                "get_fundamentals",
                "get_earnings_info", "get_earnings_history",
                "get_analyst_data", "get_institutional_data",
                "get_insider_data", "get_peer_comparison", "get_news",
            },
            ext_fund=ext,
        )
        d = _build_symbol_data("AAPL", dp, {}, {})
        assert d["fundamentals"] is ext
        assert "basic_fundamentals" not in d


class TestOptionalSourceFailuresIsolated:
    def test_one_failure_does_not_break_others(self):
        from signals.scorer import _build_symbol_data
        # Only news raises; everything else returns a value.
        dp = _StubDP(
            raises={"get_news"},
            basic_fund={"a": 1}, ext_fund=None,
            earnings={"e": 1}, earnings_history=[1, 2],
            analyst={"an": 1}, institutional={"in": 1},
            insider={"is": 1}, peer={"p": 1},
        )
        d = _build_symbol_data("AAPL", dp, {}, {})
        assert d["fundamentals"] == {"a": 1}
        assert d["earnings"] == {"e": 1}
        assert d["earnings_history"] == [1, 2]
        assert d["analyst"] == {"an": 1}
        assert d["institutional"] == {"in": 1}
        assert d["insider"] == {"is": 1}
        assert d["peer_comparison"] == {"p": 1}
        # The failed source is simply absent from the bundle.
        assert "news" not in d

    def test_all_optional_failures_yield_minimal_bundle(self):
        """When every optional dp call raises, the bundle still has
        the always-present core keys but no optional fields."""
        from signals.scorer import _build_symbol_data
        dp = _StubDP(raises={
            "get_fundamentals", "get_extended_fundamentals",
            "get_earnings_info", "get_earnings_history",
            "get_analyst_data", "get_institutional_data",
            "get_insider_data", "get_peer_comparison", "get_news",
        })
        d = _build_symbol_data("AAPL", dp, {}, {})
        for missing in ("fundamentals", "basic_fundamentals",
                        "earnings", "earnings_history", "analyst",
                        "institutional", "insider", "peer_comparison",
                        "news"):
            assert missing not in d, missing
        # earnings_info alias also absent because earnings was absent.
        # NOTE: current implementation always sets the alias key from
        # ``data.get("earnings")`` so it WILL exist as None even when
        # the optional fetch failed -- pinned below for clarity.
        assert d["earnings_info"] is None


class TestTierGating:
    def test_tier_1_omits_options_data(self):
        from signals.scorer import _build_symbol_data
        dp = _StubDP(raises={
            "get_fundamentals", "get_extended_fundamentals",
            "get_earnings_info", "get_earnings_history",
            "get_analyst_data", "get_institutional_data",
            "get_insider_data", "get_peer_comparison", "get_news",
        })
        d = _build_symbol_data("AAPL", dp, {}, {}, tier=1,
                               iv_info="IV", option_chain="OC")
        assert "iv_info" not in d
        assert "option_chain" not in d

    def test_tier_2_includes_options_data(self):
        from signals.scorer import _build_symbol_data
        dp = _StubDP(raises={
            "get_fundamentals", "get_extended_fundamentals",
            "get_earnings_info", "get_earnings_history",
            "get_analyst_data", "get_institutional_data",
            "get_insider_data", "get_peer_comparison", "get_news",
        })
        d = _build_symbol_data("AAPL", dp, {}, {}, tier=2,
                               iv_info="IV", option_chain="OC")
        assert d["iv_info"] == "IV"
        assert d["option_chain"] == "OC"


class TestAliases:
    def test_candles_daily_mirrors_candles(self):
        from signals.scorer import _build_symbol_data
        dp = _StubDP(raises={
            "get_fundamentals", "get_extended_fundamentals",
            "get_earnings_info", "get_earnings_history",
            "get_analyst_data", "get_institutional_data",
            "get_insider_data", "get_peer_comparison", "get_news",
        })
        candles_obj = object()
        d = _build_symbol_data("AAPL", dp, {}, {"AAPL": candles_obj})
        assert d["candles"] is candles_obj
        assert d["candles_daily"] is candles_obj

    def test_earnings_info_alias_present_even_when_earnings_missing(self):
        """Pinning current behavior: ``earnings_info`` is set
        unconditionally from ``data.get("earnings")`` so it will be
        ``None`` when the earnings fetch raised, not absent."""
        from signals.scorer import _build_symbol_data
        dp = _StubDP(raises={
            "get_fundamentals", "get_extended_fundamentals",
            "get_earnings_info",
            "get_earnings_history", "get_analyst_data",
            "get_institutional_data", "get_insider_data",
            "get_peer_comparison", "get_news",
        })
        d = _build_symbol_data("AAPL", dp, {}, {})
        assert "earnings" not in d
        assert "earnings_info" in d
        assert d["earnings_info"] is None
