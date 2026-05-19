"""optimize_profiles.py grid / genetic / live-optimize tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from core.profile_optimization import composite_score, lookback_dates
from core.profit_profiles import PROFIT_PROFILE_ENV, VALID_PROFILES
from core.simulation.stats import build_backtest_result
from core.simulation.sim_broker import SimFill
from tests.profit_test_utils import load_real_profit_config


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def _mock_backtest_result(profile: str, *, profit: float, win_rate: float = 60.0) -> build_backtest_result:
    trades = [
        SimFill("SPY", "SELL", 1, 100.0, pnl=profit / 2),
        SimFill("SPY", "SELL", 1, 100.0, pnl=profit / 2),
    ]
    return build_backtest_result(
        profile=profile,
        start_date="2024-06-03",
        end_date="2024-06-07",
        trading_days=3,
        cycles_run=6,
        initial_equity=100_000.0,
        equity_curve=[
            ("2024-06-03", 100_000.0),
            ("2024-06-07", 100_000.0 + profit),
        ],
        closed_trades=trades,
        llm_cost_usd=0.05,
    )


def _profile_scores() -> dict[str, float]:
    """Deterministic composite ordering: aggressive > balanced > conservative."""
    return {
        "conservative": composite_score(0.3, 1.0, 45.0),
        "balanced": composite_score(0.8, 1.2, 55.0),
        "aggressive": composite_score(1.4, 1.8, 65.0),
    }


def _simulate_side_effect(profile: str, start: str, end: str, **kwargs):
    scores = _profile_scores()
    base = str(profile)
    profit_map = {"conservative": 50.0, "balanced": 200.0, "aggressive": 400.0}
    profit = profit_map.get(base, 100.0)
    cid = kwargs.get("candidate_id") or base
    return _mock_backtest_result(cid, profit=profit)


def _grid_args(**overrides) -> argparse.Namespace:
    defaults = {
        "days": 7,
        "end": "2024-06-10",
        "output": None,
        "initial_cash": 100_000.0,
        "cycles_per_day": 2,
        "quick": True,
        "baseline": "balanced",
        "top": 5,
        "genetic": False,
        "generations": 3,
        "population": 6,
        "elite": 1,
        "mutation_rate": 0.25,
        "seed": 42,
        "save_profile": None,
        "emit_snippet": None,
        "parallel": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestOptimizeProfilesGrid:
    def test_run_grid_returns_valid_best_profile(self):
        from scripts import optimize_profiles as op

        load_real_profit_config("balanced")
        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_simulate_side_effect):
            payload = op.run_grid_optimization(_grid_args(quick=True))

        best = payload["best"]
        assert best["base_profile"] in VALID_PROFILES
        assert best["candidate_id"] in VALID_PROFILES
        assert payload["mode"] == "grid"
        assert payload["baseline_profile"] == "balanced"
        assert "recommended_config_changes" in payload

    def test_grid_best_improves_composite_vs_conservative(self):
        from scripts import optimize_profiles as op

        load_real_profit_config("balanced")
        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_simulate_side_effect):
            payload = op.run_grid_optimization(_grid_args(quick=True))

        by_id = {r["candidate_id"]: r["metrics"]["composite_score"] for r in payload["rankings"]}
        assert by_id["aggressive"] > by_id["conservative"]
        assert payload["best"]["metrics"]["composite_score"] == pytest.approx(by_id["aggressive"])
        assert payload["best"]["metrics"]["composite_score"] >= by_id["balanced"]

    def test_grid_rankings_sorted_descending(self):
        from scripts import optimize_profiles as op

        load_real_profit_config("balanced")
        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_simulate_side_effect):
            payload = op.run_grid_optimization(_grid_args(quick=True, top=3))

        scores = [r["metrics"]["composite_score"] for r in payload["rankings"]]
        assert scores == sorted(scores, reverse=True)

    def test_grid_all_failures_exits(self):
        from scripts import optimize_profiles as op

        load_real_profit_config("balanced")

        def _boom(*_a, **_k):
            raise RuntimeError("sim failed")

        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_boom):
            with pytest.raises(SystemExit, match="All candidates failed"):
                op.run_grid_optimization(_grid_args(quick=True))

    def test_main_writes_json(self, tmp_path, monkeypatch):
        repo = Path(__file__).resolve().parents[1]
        monkeypatch.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        out = tmp_path / "opt_out.json"
        argv = [
            "--quick",
            "--days",
            "7",
            "--end",
            "2024-06-10",
            "-o",
            str(out),
        ]
        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_simulate_side_effect):
            from scripts.optimize_profiles import main

            assert main(argv) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["best"]["base_profile"] in VALID_PROFILES


class TestOptimizeProfilesGenetic:
    def test_genetic_mode_uses_mocked_simulation(self):
        from scripts import optimize_profiles as op

        load_real_profit_config("balanced")
        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_simulate_side_effect):
            payload = op.run_genetic_optimization(
                _grid_args(genetic=True, generations=2, population=6, quick=True)
            )
        assert payload["mode"] == "genetic"
        assert payload["best"]["base_profile"] in VALID_PROFILES
        assert payload["best"]["metrics"]["composite_score"] > 0
        assert len(payload["generation_history"]) == 2

    def test_genetic_save_profile_writes_registry(self, tmp_path, monkeypatch):
        from core import profit_profiles as pp
        from scripts import optimize_profiles as op

        reg = tmp_path / "evolved_profiles.json"
        monkeypatch.setattr(pp, "evolved_profiles_path", lambda: reg)
        load_real_profit_config("balanced")

        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_simulate_side_effect):
            payload = op.run_genetic_optimization(
                _grid_args(
                    genetic=True,
                    generations=2,
                    population=6,
                    save_profile="evolved_test_opt",
                )
            )
        assert payload["saved_profile"]["name"] == "evolved_test_opt"
        assert reg.is_file()
        saved = json.loads(reg.read_text(encoding="utf-8"))
        assert "evolved_test_opt" in saved["profiles"]


class TestOptimizeProfilesEdgeCases:
    def test_lookback_dates_invalid_end_still_parses(self):
        with pytest.raises(ValueError):
            lookback_dates(7, end_date="not-a-date")

    def test_lookback_dates_single_day(self):
        start, end = lookback_dates(1, end_date="2024-06-10")
        assert start == end == "2024-06-10"

    def test_quick_grid_loads_real_profit_config_per_candidate(self):
        """Each candidate uses composed config (not singleton) with correct profile caps."""
        from scripts import optimize_profiles as op

        seen_llm_caps: list[float] = []

        def _capture(profile, start, end, **kwargs):
            composed = kwargs.get("composed")
            assert composed is not None, "optimizer should pass composed= for thread safety"
            seen_llm_caps.append(composed.risk.max_daily_llm_cost)
            cid = kwargs.get("candidate_id") or profile
            return _mock_backtest_result(cid, profit=10.0)

        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_capture):
            op.run_grid_optimization(_grid_args(quick=True))

        assert 3.0 in seen_llm_caps  # conservative
        assert 6.5 in seen_llm_caps  # aggressive

    def test_grid_parallel_uses_thread_safe_composed(self):
        from scripts import optimize_profiles as op

        composed_calls = []

        def _capture(profile, start, end, **kwargs):
            assert kwargs.get("composed") is not None
            composed_calls.append(kwargs["composed"].risk.max_daily_llm_cost)
            return _mock_backtest_result(profile, profit=50.0)

        with patch("core.optimizer_backtest.simulate_backtest", side_effect=_capture):
            op.run_grid_optimization(_grid_args(quick=True, parallel=2))

        assert len(composed_calls) == 3


class TestLiveOptimizeCLI:
    def test_live_optimize_empty_logs_defaults_balanced(self, tmp_path, monkeypatch):
        from core import profit_cycle_logger as pcl
        from core.entry_cli import run_live_optimize_cli

        monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
        monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)
        load_real_profit_config("aggressive")

        args = argparse.Namespace(
            live_optimize_days=7,
            live_optimize_output=str(tmp_path / "live.json"),
        )
        assert run_live_optimize_cli(args) == 0
        data = json.loads((tmp_path / "live.json").read_text(encoding="utf-8"))
        assert data["suggested_profile"] == "balanced"
        assert data["confidence"] == "none"

    def test_live_optimize_picks_best_from_logs(self, tmp_path, monkeypatch):
        from core import profit_cycle_logger as pcl
        from core.entry_cli import run_live_optimize_cli
        from datetime import datetime, timedelta, timezone

        monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
        monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)
        load_real_profit_config("balanced")

        now = datetime.now(timezone.utc)
        entries = []
        for i in range(5):
            entries.append(
                {
                    "ts": (now - timedelta(hours=i + 1)).isoformat(),
                    "session_date": now.strftime("%Y-%m-%d"),
                    "cycle_id": i + 1,
                    "profit_profile": "aggressive",
                    "outcome": "done",
                    "pnl": {"cycle_realized_pnl_usd": 25.0, "llm_cost_usd": 0.1},
                    "trade_outcome": {"action": "done"},
                }
            )
        for i in range(5):
            entries.append(
                {
                    "ts": (now - timedelta(hours=i + 10)).isoformat(),
                    "session_date": now.strftime("%Y-%m-%d"),
                    "cycle_id": i + 10,
                    "profit_profile": "conservative",
                    "outcome": "done",
                    "pnl": {"cycle_realized_pnl_usd": -8.0, "llm_cost_usd": 0.1},
                    "trade_outcome": {"action": "done"},
                }
            )
        path = tmp_path / f"profit_cycles_{now.strftime('%Y-%m-%d')}.json"
        path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

        args = argparse.Namespace(
            live_optimize_days=7,
            live_optimize_output=str(tmp_path / "live.json"),
        )
        run_live_optimize_cli(args)
        data = json.loads((tmp_path / "live.json").read_text(encoding="utf-8"))
        assert data["suggested_profile"] == "aggressive"
        assert data["confidence"] == "high"
