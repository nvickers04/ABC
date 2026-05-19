"""Backtest simulation harness."""

from __future__ import annotations

import pytest

from core.simulation.archive import bars_by_date, save_archive
from core.simulation.csv_export import write_trade_log_csv
from core.simulation.report import format_backtest_comparison, format_backtest_report
from core.entry_cli import parse_simulate_profiles
from core.simulation.stats import build_backtest_result
from core.simulation.sim_broker import SimFill
from core.simulation.trade_log import SimTradeLog


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_build_backtest_result_metrics():
    trades = [
        SimFill("AAPL", "SELL", 1, 100.0, pnl=50.0),
        SimFill("AAPL", "SELL", 1, 100.0, pnl=-20.0),
    ]
    curve = [("2024-01-02", 100_000.0), ("2024-01-03", 100_030.0)]
    r = build_backtest_result(
        profile="balanced",
        start_date="2024-01-02",
        end_date="2024-01-03",
        trading_days=2,
        cycles_run=4,
        initial_equity=100_000.0,
        equity_curve=curve,
        closed_trades=trades,
        llm_cost_usd=0.05,
    )
    assert r.total_profit == pytest.approx(30.0)
    assert r.trade_count == 2
    assert r.win_rate == 50.0
    assert "Backtest" in format_backtest_report(r)


def test_parse_simulate_profiles_dedupes():
    assert parse_simulate_profiles("conservative,balanced,aggressive") == [
        "conservative",
        "balanced",
        "aggressive",
    ]
    assert parse_simulate_profiles("balanced,balanced") == ["balanced"]


def test_format_backtest_comparison_table():
    def _mk(profile: str, profit: float) -> build_backtest_result:
        return build_backtest_result(
            profile=profile,
            start_date="2024-01-02",
            end_date="2024-01-05",
            trading_days=3,
            cycles_run=3,
            initial_equity=100_000.0,
            equity_curve=[("2024-01-02", 100_000.0), ("2024-01-05", 100_000.0 + profit)],
            closed_trades=[],
            llm_cost_usd=0.01,
        )

    text = format_backtest_comparison(
        [
            _mk("conservative", 100.0),
            _mk("balanced", 250.0),
            _mk("aggressive", -50.0),
        ]
    )
    assert "Backtest profile comparison" in text
    assert "`conservative`" in text and "`aggressive`" in text
    assert "$250.00" in text
    assert "Highest total profit" in text
    assert "`balanced`" in text


def test_trade_log_csv_export(tmp_path):
    trade = SimTradeLog(
        profit_profile="aggressive",
        symbol="NVDA",
        qty=2,
        entry_time_utc="2024-06-03T13:35:00+00:00",
        exit_time_utc="2024-06-03T20:00:00+00:00",
        entry_price=120.0,
        exit_price=125.0,
        realized_pnl=10.0,
        realized_rr=2.08,
        session_date="2024-06-03",
        exit_reason="sell",
    )
    result = build_backtest_result(
        profile="aggressive",
        start_date="2024-06-03",
        end_date="2024-06-05",
        trading_days=1,
        cycles_run=1,
        initial_equity=100_000.0,
        equity_curve=[],
        closed_trades=[],
        trade_log=[trade],
        llm_cost_usd=0.0,
    )
    out = tmp_path / "trades.csv"
    write_trade_log_csv(out, [result])
    text = out.read_text(encoding="utf-8")
    assert "profit_profile" in text
    assert "NVDA" in text
    assert "entry_time_utc" in text
    assert "realized_rr" in text
    assert "aggressive" in text


def test_archive_roundtrip(tmp_path, monkeypatch):
    from core.simulation import archive as arch

    monkeypatch.setattr(arch, "ARCHIVE_ROOT", tmp_path / "archives")
    payload = {
        "symbol": "SPY",
        "bars": [
            {
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "timestamp": 1,
                "date": "2024-06-03",
            }
        ],
    }
    save_archive("SPY", "2024-06-01", "2024-06-10", payload)
    loaded = arch.load_archive("SPY", "2024-06-01", "2024-06-10")
    assert loaded is not None
    assert bars_by_date(loaded)["2024-06-03"]["close"] == 1.5
