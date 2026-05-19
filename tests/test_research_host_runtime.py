"""Research host runtime — decoupling and token cap (no live TWS)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("ABC_RESEARCH_HOST", raising=False)
    yield
    monkeypatch.delenv("ABC_RESEARCH_HOST", raising=False)


def test_mark_research_host_process(clean_env):
    from core.runtime.research_host_runtime import (
        is_research_host_process,
        mark_research_host_process,
    )

    assert is_research_host_process() is False
    mark_research_host_process()
    assert is_research_host_process() is True


def test_request_shutdown_is_idempotent(clean_env, monkeypatch):
    from core.runtime import research_host_runtime as rh
    from core.runtime.research_host_runtime import request_shutdown, shutdown_reason

    stop_scorer = MagicMock()
    stop_evo = MagicMock()
    monkeypatch.setattr("signals.scorer.stop_scorer", stop_scorer)
    monkeypatch.setattr("signals.template_evolution.stop_evolution", stop_evo)

    request_shutdown("test")
    request_shutdown("test2")
    assert shutdown_reason() == "test"
    stop_scorer.assert_called_once()
    stop_evo.assert_called_once()
    rh._shutdown_reason = None


def test_publish_research_host_heartbeat_writes_status_keys(monkeypatch):
    store: dict[str, float] = {}

    def _set(key: str, value: float, reason: str = "", *, log: bool = True) -> None:
        store[key] = value

    def _get(key: str, default: float) -> float:
        return float(store.get(key, default))

    monkeypatch.setattr("memory.set_research_config", _set)
    monkeypatch.setattr("memory.get_research_config", _get)
    monkeypatch.setattr("memory.repos.config_repo.set_research_config", _set)
    monkeypatch.setattr("memory.repos.config_repo.get_research_config", _get)

    from core.runtime.heartbeat import (
        RESEARCH_HOST_ROUND_KEY,
        RESEARCH_HOST_STATUS_KEY,
        RESEARCH_HOST_USAGE_PCT_KEY,
        ResearchHostStatus,
        publish_research_host_heartbeat,
        read_heartbeat,
    )
    ts = publish_research_host_heartbeat(
        now=1_700_000_100.0,
        status=ResearchHostStatus.SCORING,
        round_num=7,
        usage_pct=42.5,
    )
    assert ts == pytest.approx(1_700_000_100.0)
    assert read_heartbeat() == pytest.approx(ts)
    assert store[RESEARCH_HOST_STATUS_KEY] == ResearchHostStatus.SCORING
    assert store[RESEARCH_HOST_ROUND_KEY] == 7.0
    assert store[RESEARCH_HOST_USAGE_PCT_KEY] == pytest.approx(42.5)


def test_is_research_host_operational_rejects_cap_stopped(monkeypatch):
    import time

    store: dict[str, float] = {
        "research_host_heartbeat_ts": time.time(),
        "research_host_status": 2.0,
    }

    def _get(key: str, default: float) -> float:
        return float(store.get(key, default))

    monkeypatch.setattr("memory.get_research_config", _get)
    monkeypatch.setattr("memory.repos.config_repo.get_research_config", _get)
    monkeypatch.setattr(
        "core.runtime.heartbeat.read_heartbeat",
        lambda: store["research_host_heartbeat_ts"],
    )

    from core.runtime.heartbeat import (
        RESEARCH_HOST_STATUS_KEY,
        ResearchHostStatus,
        is_research_host_operational,
    )

    assert is_research_host_operational(stale_after_s=120.0) is True
    store[RESEARCH_HOST_STATUS_KEY] = ResearchHostStatus.CAP_STOPPED
    assert is_research_host_operational(stale_after_s=120.0) is False
