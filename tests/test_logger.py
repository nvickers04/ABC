"""Profit cycle logger and ProfitConfig.log_cycle integration tests."""

from __future__ import annotations

import json
import os
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.central_profit_config import ProfitConfig, reload_profit_config
from core.profit_cycle_logger import (
    append_profit_cycle_log,
    collect_cycle_metrics,
    load_daily_entries,
    load_entries_since,
    snapshot_profit_config,
)
from core.profit_profiles import PROFIT_PROFILE_ENV, VALID_PROFILES
from tests.profit_test_utils import load_real_profit_config


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override conftest Postgres fixture — logger tests use JSON files only."""
    yield


@pytest.fixture(autouse=True)
def _isolated_log_env(tmp_path, monkeypatch):
    """JSON-only logs; no Postgres profit_cycle_logs writes."""
    from core import profit_cycle_logger as pcl

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for k in ("PGHOST", "PGDATABASE", "PGUSER"):
        monkeypatch.delenv(k, raising=False)
    yield tmp_path


@pytest.fixture
def mock_agent():
    """Minimal agent surface for collect_cycle_metrics."""
    gw = SimpleNamespace(net_liquidation=105_000.0)
    agent = SimpleNamespace(
        gateway=gw,
        _session_high_water=110_000.0,
        _profit_log_prev_realized_pnl=100.0,
        _profit_last_tool={"action": "done", "symbol": "SPY", "success": True},
    )
    return agent


class TestProfitConfigSnapshot:
    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_snapshot_reflects_real_profit_config(self, profile):
        os.environ[PROFIT_PROFILE_ENV] = profile
        cfg = load_real_profit_config(profile)
        snap = snapshot_profit_config(cfg)
        assert snap["profit_profile"] == profile
        assert snap["trading_mode"] == cfg.trading_mode
        assert snap["risk"]["max_daily_llm_cost"] == cfg.risk.max_daily_llm_cost
        assert snap["loop"]["react_max_turns_per_cycle"] == cfg.loop.react_max_turns_per_cycle
        assert snap["memory"]["cycle_wm_max_chars"] == cfg.memory.cycle_wm_max_chars
        assert snap["prompt"]["llm_max_tokens"] == cfg.prompt.llm_max_tokens

    def test_profiles_differ_on_key_levers(self):
        snaps = {}
        for profile in VALID_PROFILES:
            snaps[profile] = snapshot_profit_config(load_real_profit_config(profile))
        assert (
            snaps["conservative"]["risk"]["max_daily_llm_cost"]
            < snaps["aggressive"]["risk"]["max_daily_llm_cost"]
        )
        assert (
            snaps["conservative"]["memory"]["cycle_wm_max_chars"]
            < snaps["aggressive"]["memory"]["cycle_wm_max_chars"]
        )


class TestLogCycleIntegration:
    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_log_cycle_persists_profile_config(self, profile, mock_agent):
        cfg = load_real_profit_config(profile)
        cfg.log_cycle(
            cycle_id=10,
            outcome="done",
            cooldown_seconds=45,
            session="regular",
            cycle_summary=f"cycle under {profile}",
            cycle_actions=["quality_status", "done"],
            agent=mock_agent,
        )
        entries = load_daily_entries(date.today().isoformat())
        assert len(entries) == 1
        assert entries[0]["profit_profile"] == profile
        assert entries[0]["config"]["risk"]["max_daily_llm_cost"] == cfg.risk.max_daily_llm_cost
        assert entries[0]["quality"]["overall_quality"]  # populated or default

    def test_append_profit_cycle_log_accepts_explicit_config(self, mock_agent):
        cfg = load_real_profit_config("balanced")
        rec = append_profit_cycle_log(
            cfg,
            cycle_id=1,
            outcome="market_hours",
            cooldown_seconds=60,
            session="premarket",
            agent=mock_agent,
        )
        assert rec.outcome == "market_hours"
        assert rec.config["prompt"]["min_rr_paper"] == cfg.prompt.min_rr_paper

    def test_multiple_cycles_same_day(self, mock_agent):
        cfg = load_real_profit_config("balanced")
        for cid in (1, 2, 3):
            cfg.log_cycle(
                cycle_id=cid,
                outcome="done",
                cooldown_seconds=30,
                agent=mock_agent,
            )
        entries = load_daily_entries(date.today().isoformat())
        assert len(entries) == 3
        assert [e["cycle_id"] for e in entries] == [1, 2, 3]


class TestCollectCycleMetrics:
    def test_collect_without_agent_returns_defaults(self):
        pnl, quality, trade = collect_cycle_metrics(None)
        assert pnl.cycle_realized_pnl_usd == 0.0
        assert trade.action == ""

    def test_collect_with_agent_and_mock_quality(self, mock_agent):
        with patch("data.cost_tracker.get_cost_tracker") as gct:
            gct.return_value.get_budget_summary.return_value = SimpleNamespace(
                today_llm_cost=1.5,
                today_realized_pnl=150.0,
            )
            with patch("core.quality.quality_matrix.get_quality_matrix_service") as gqm:
                m = SimpleNamespace(
                    overall_quality="full",
                    risk_multiplier=0.85,
                    execution_quality=0.9,
                )
                gqm.return_value.get_matrix.return_value = m
                mock_agent._check_daily_loss = lambda: None
                pnl, quality, trade = collect_cycle_metrics(mock_agent)
        assert quality.overall_quality == "full"
        assert trade.action == "done"


class TestLoggerEdgeCases:
    def test_load_daily_entries_missing_date_returns_empty(self):
        assert load_daily_entries("1999-01-01") == []

    def test_load_entries_since_filters_by_ts(self, mock_agent):
        cfg = load_real_profit_config("balanced")
        cfg.log_cycle(cycle_id=1, outcome="done", cooldown_seconds=30, agent=mock_agent)
        from datetime import datetime, timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        rows = load_entries_since(since)
        assert len(rows) >= 1

    def test_corrupt_json_file_returns_empty(self, tmp_path):
        from core import profit_cycle_logger as pcl

        bad = tmp_path / f"profit_cycles_{date.today().isoformat()}.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_daily_entries(date.today().isoformat()) == []

    def test_postgres_disabled_without_database_url(self, monkeypatch, mock_agent):
        from core import profit_cycle_logger as pcl

        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("PGHOST", raising=False)
        assert pcl._postgres_enabled() is False
        cfg = load_real_profit_config("balanced")
        with patch.object(pcl, "_append_postgres") as pg:
            cfg.log_cycle(cycle_id=99, outcome="done", cooldown_seconds=1, agent=mock_agent)
        pg.assert_not_called()

    def test_postgres_append_failure_does_not_raise(self, monkeypatch, mock_agent):
        from core import profit_cycle_logger as pcl

        monkeypatch.setattr(pcl, "_postgres_enabled", lambda: True)
        cfg = load_real_profit_config("balanced")
        with patch.object(pcl, "_ensure_pg_table", side_effect=RuntimeError("no db")):
            rec = append_profit_cycle_log(
                cfg,
                cycle_id=5,
                outcome="done",
                cooldown_seconds=10,
                agent=mock_agent,
            )
        assert rec.cycle_id == 5
        assert load_daily_entries(date.today().isoformat())

    def test_postgres_failure_logs_profit_config(self, monkeypatch, mock_agent, caplog):
        import logging

        from core import profit_cycle_logger as pcl

        monkeypatch.setattr(pcl, "_postgres_enabled", lambda: True)
        cfg = load_real_profit_config("balanced")
        with (
            patch.object(pcl, "_ensure_pg_table"),
            patch("memory.get_db", side_effect=RuntimeError("connection refused")),
            caplog.at_level(logging.WARNING),
        ):
            append_profit_cycle_log(
                cfg,
                cycle_id=6,
                outcome="done",
                cooldown_seconds=10,
                agent=mock_agent,
            )
        assert any("postgres append failed" in r.message for r in caplog.records)

    def test_profile_label_override_in_snapshot(self):
        cfg = load_real_profit_config("balanced")
        snap = snapshot_profit_config(cfg, profile_label="ga_candidate_1")
        assert snap["profit_profile"] == "ga_candidate_1"

    def test_agent_profit_log_and_return_uses_profit_config(self, mock_agent):
        from core.agent import TradingAgent

        cfg = load_real_profit_config("aggressive")
        agent = MagicMock(spec=TradingAgent)
        agent._cycle_id = 7
        agent._current_session = "regular"
        agent._last_cycle_summary = "mock"
        agent._profit_log_prev_realized_pnl = None
        agent._profit_last_tool = {}
        agent.gateway = mock_agent.gateway
        agent._check_daily_loss = lambda: None
        agent._session_high_water = 110_000.0

        with patch("core.central_profit_config.get_profit_config", return_value=cfg):
            TradingAgent._profit_log_and_return(agent, 30, "done", cycle_actions=["done"])

        entries = load_daily_entries(date.today().isoformat())
        assert any(e["cycle_id"] == 7 for e in entries)
