"""Profitability cycle logger."""

from __future__ import annotations

import json
from datetime import date

import pytest

from core.central_profit_config import ProfitConfig, reload_profit_config
from core.profit_cycle_logger import (
    append_profit_cycle_log,
    load_daily_entries,
    snapshot_profit_config,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_snapshot_profit_config_keys():
    cfg = reload_profit_config()
    snap = snapshot_profit_config(cfg)
    assert "risk" in snap and "loop" in snap and "memory" in snap
    assert "profit_profile" in snap


def test_append_and_load_json(tmp_path, monkeypatch):
    from core import profit_cycle_logger as pcl

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)
    cfg = reload_profit_config()
    rec = append_profit_cycle_log(
        cfg,
        cycle_id=1,
        outcome="done",
        cooldown_seconds=30,
        session="regular",
        cycle_summary="test",
        cycle_actions=["quality_status", "done"],
    )
    assert rec.cycle_id == 1
    entries = load_daily_entries(date.today().isoformat())
    assert len(entries) == 1
    assert entries[0]["outcome"] == "done"
    assert entries[0]["config"]["risk"]["max_daily_llm_cost"] > 0


def test_profit_config_log_cycle_delegates(tmp_path, monkeypatch):
    from core import profit_cycle_logger as pcl

    monkeypatch.setattr(pcl, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pcl, "_postgres_enabled", lambda: False)
    cfg = reload_profit_config()
    cfg.log_cycle(
        cycle_id=2,
        outcome="done",
        cooldown_seconds=60,
        session="regular",
    )
    path = tmp_path / f"profit_cycles_{date.today().isoformat()}.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["entries"]) == 1
