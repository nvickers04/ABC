"""Simulation / simulate_backtest tests (mocked MDA, IBKR, and fast agent cycles)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from core.central_profit_config import simulate_backtest
from core.profit_profiles import PROFIT_PROFILE_ENV, VALID_PROFILES
from core.simulation.runner import run_backtest_async
from core.simulation.types import BacktestResult
from tests.profit_test_utils import load_real_profit_config, mock_archive_for_range, nyse_session_dates


@pytest.fixture(autouse=True)
def _isolated_db():
    from core.central_profit_config import clear_shared_replay_data

    clear_shared_replay_data()
    yield
    clear_shared_replay_data()


@pytest.fixture
def sim_env(monkeypatch, tmp_path):
    """Isolate archives, disable Postgres scorer lookup, fast agent cycles."""
    from core.simulation import archive as arch

    monkeypatch.setattr(arch, "ARCHIVE_ROOT", tmp_path / "archives")
    monkeypatch.setenv("ABC_SIMULATION", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(
        "core.simulation.runner._top_composite_for_date",
        lambda _d: ("SPY", 0.72),
    )

    async def _fast_cycle(self) -> int:
        return 30

    monkeypatch.setattr("core.agent.TradingAgent.run_cycle", _fast_cycle)
    yield tmp_path


def _async_archive_mock(archives: dict[str, dict]):
    async def _ensure(symbol: str, start: str, end: str):
        return archives.get(symbol.upper()) or archives.get("SPY")

    return _ensure


class TestSimulateBacktestUnit:
    """Unit-level backtest with mocked historical data."""

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_simulate_backtest_uses_real_profit_config(self, sim_env, profile, monkeypatch):
        start, end = "2024-06-03", "2024-06-07"
        archives = mock_archive_for_range(start, end)
        monkeypatch.setenv(PROFIT_PROFILE_ENV, profile)
        cfg_before = load_real_profit_config(profile)

        with patch(
            "core.simulation.archive.ensure_archive_async",
            side_effect=_async_archive_mock(archives),
        ):
            result = simulate_backtest(
                profile,
                start,
                end,
                initial_cash=50_000.0,
                cycles_per_day=1,
            )

        assert isinstance(result, BacktestResult)
        assert result.profile == profile
        assert result.trading_days == len(nyse_session_dates(start, end))
        assert result.cycles_run == result.trading_days * 1
        assert result.initial_equity == pytest.approx(50_000.0)
        assert any("QualityMatrix" in n for n in result.notes)

        cfg_after = load_real_profit_config(profile)
        assert cfg_after.risk.max_daily_llm_cost == cfg_before.risk.max_daily_llm_cost
        if profile == "conservative":
            assert cfg_after.risk.max_daily_llm_cost == pytest.approx(3.0)
        elif profile == "aggressive":
            assert cfg_after.risk.max_daily_llm_cost == pytest.approx(6.5)

    def test_simulate_backtest_no_trading_sessions(self, sim_env, monkeypatch):
        """Holiday-only window yields zero trading days without raising."""
        archives = mock_archive_for_range("2024-01-02", "2024-01-05")
        load_real_profit_config("balanced")
        with patch(
            "core.simulation.archive.ensure_archive_async",
            side_effect=_async_archive_mock(archives),
        ):
            result = simulate_backtest("balanced", "2024-01-01", "2024-01-01")
        assert result.trading_days == 0
        assert result.cycles_run == 0
        assert result.total_profit == pytest.approx(0.0)

    def test_simulate_backtest_empty_archive_still_runs(self, sim_env, monkeypatch):
        async def _empty(_symbol, _start, _end):
            return {"symbol": "SPY", "bars": []}

        load_real_profit_config("balanced")
        with patch("core.simulation.archive.ensure_archive_async", side_effect=_empty):
            result = simulate_backtest("balanced", "2024-06-03", "2024-06-05", cycles_per_day=1)
        assert result.trading_days >= 1
        assert result.cycles_run >= 1

    def test_shared_replay_loads_archives_once(self, sim_env, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        from core.central_profit_config import clear_shared_replay_data, get_shared_replay_data

        clear_shared_replay_data()
        start, end = "2024-06-03", "2024-06-05"
        archives = mock_archive_for_range(start, end)
        mock = AsyncMock(side_effect=lambda sym, s, e: archives.get(sym.upper(), archives["SPY"]))

        with patch("core.simulation.archive.ensure_archive_async", mock):
            shared = get_shared_replay_data(start, end)
            asyncio.run(shared.load())
            asyncio.run(shared.load())
            a = shared.spawn_session()
            b = shared.spawn_session()

        assert a._bars is b._bars
        assert mock.await_count == len(shared.symbols)

    def test_config_patches_applied_in_runner(self, sim_env, monkeypatch):
        """Patched LLM cap should appear in runner notes (via load + apply_config_patches)."""
        start, end = "2024-06-03", "2024-06-04"
        archives = mock_archive_for_range(start, end)
        patches = {"risk": {"max_daily_llm_cost": 2.25}}
        with patch(
            "core.simulation.archive.ensure_archive_async",
            side_effect=_async_archive_mock(archives),
        ):
            result = simulate_backtest(
                "balanced",
                start,
                end,
                config_patches=patches,
                candidate_id="test_patch",
                cycles_per_day=1,
            )
        assert result.profile == "test_patch"
        assert any("2.25" in n for n in result.notes)

    def test_invalid_profile_name_raises(self, sim_env):
        load_real_profit_config("balanced")
        with pytest.raises(ValueError, match="profit profile"):
            simulate_backtest("not_a_profile", "2024-06-03", "2024-06-05")

    def test_oversized_backtest_window_raises(self, sim_env, monkeypatch):
        monkeypatch.setenv("ABC_MAX_BACKTEST_CALENDAR_DAYS", "5")
        load_real_profit_config("balanced")
        with pytest.raises(ValueError, match="exceeds limit"):
            simulate_backtest("balanced", "2024-01-01", "2024-06-01")

    def test_invalid_date_format_raises(self, sim_env):
        load_real_profit_config("balanced")
        with pytest.raises(ValueError, match="Invalid backtest dates"):
            simulate_backtest("balanced", "not-a-date", "2024-06-05")

    def test_mda_rate_limit_falls_back_to_yfinance(self, sim_env, monkeypatch):
        start, end = "2024-06-03", "2024-06-04"
        archives = mock_archive_for_range(start, end)

        async def _rate_limit(*_a, **_k):
            return None

        yf_called = {"n": 0}

        def _yf(symbol, start, end):
            yf_called["n"] += 1
            return archives.get(symbol.upper(), archives["SPY"])

        load_real_profit_config("balanced")
        with (
            patch("core.simulation.archive.fetch_mda_daily_async", side_effect=_rate_limit),
            patch("core.simulation.archive._fetch_yfinance_daily", side_effect=_yf),
        ):
            simulate_backtest("balanced", start, end, cycles_per_day=1)
        assert yf_called["n"] >= 1


class TestRunBacktestAsyncIntegration:
    """Async runner path (no asyncio.run wrapper)."""

    @pytest.mark.asyncio
    async def test_run_backtest_async_mocked(self, sim_env, monkeypatch):
        start, end = "2024-06-03", "2024-06-05"
        archives = mock_archive_for_range(start, end)
        cfg = load_real_profit_config("conservative")
        assert cfg.loop.react_max_consecutive_tool_failures == 3

        with patch(
            "core.simulation.archive.ensure_archive_async",
            side_effect=_async_archive_mock(archives),
        ):
            result = await run_backtest_async(
                "conservative",
                start,
                end,
                cycles_per_day=2,
            )
        assert result.profile == "conservative"
        assert result.cycles_run == result.trading_days * 2


class TestSimulationMocksExternalIO:
    """Ensure MarketData / yfinance are not called when archives are on disk."""

    def test_mda_and_yfinance_not_called_when_archives_cached(self, sim_env):
        from core.simulation.archive import save_archive

        start, end = "2024-06-03", "2024-06-04"
        archives = mock_archive_for_range(start, end)
        for sym, payload in archives.items():
            save_archive(sym, start, end, payload)
        load_real_profit_config("balanced")

        mda_mock = AsyncMock()
        with (
            patch("core.simulation.archive.fetch_mda_daily_async", mda_mock),
            patch("core.simulation.archive._fetch_yfinance_daily") as yf,
        ):
            simulate_backtest("balanced", start, end, cycles_per_day=1)
        mda_mock.assert_not_called()
        yf.assert_not_called()
