"""Two-machine split: research host heartbeat vs trader in-process scorer.

Uses real ``research_config`` via isolated Postgres (conftest) for heartbeat
round-trip; mocks ``run_research_threaded`` and Grok/IBKR (never invoked).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from quality_test_support import reset_quality_runtime_state


def _trader_args(**overrides):
    base = dict(
        no_research=False,
        force_in_process=False,
        require_research_host=False,
        verbose=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


def test_research_writes_trader_reads_heartbeat():
    """Research host round → trader sees fresh heartbeat (split-host contract)."""
    from core.runtime.heartbeat import (
        HEARTBEAT_KEY,
        heartbeat_age_s,
        is_research_host_alive,
        write_heartbeat,
    )
    from memory import get_research_config

    now = time.time()
    write_heartbeat(now=now)
    assert get_research_config(HEARTBEAT_KEY, 0.0) == pytest.approx(now)
    assert is_research_host_alive(stale_after_s=120.0, now=now) is True
    assert heartbeat_age_s(now=now) < 1.0


def test_trader_skips_inprocess_scorer_when_heartbeat_fresh(monkeypatch):
    """Production path: remote research host → no ``run_research_threaded`` on trader."""
    import __main__ as trader_main

    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_alive", lambda *a, **k: True
    )
    monkeypatch.setattr("core.runtime.heartbeat.heartbeat_age_s", lambda *a, **k: 5.0)
    monkeypatch.setattr("core.config.TRADER_IN_PROCESS_SCORER_NEVER", False, raising=False)

    threaded: list[str] = []

    def _no_threaded(**_kw):
        threaded.append("called")

    monkeypatch.setattr("signals.scorer.run_research_threaded", _no_threaded)

    trader_main._configure_scorer(_trader_args())
    assert threaded == []


def test_trader_starts_inprocess_when_heartbeat_stale_and_not_required(monkeypatch):
    """Dev single-box: stale heartbeat → trader may run scorer in-process."""
    import __main__ as trader_main

    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_alive", lambda *a, **k: False
    )
    monkeypatch.setattr(
        "core.runtime.heartbeat.heartbeat_age_s", lambda *a, **k: float("inf")
    )
    monkeypatch.setattr("core.config.TRADER_IN_PROCESS_SCORER_NEVER", False, raising=False)

    threaded: list[str] = []

    def _threaded(**_kw):
        threaded.append("called")

    monkeypatch.setattr("signals.scorer.run_research_threaded", _threaded)

    trader_main._configure_scorer(_trader_args())
    assert threaded == ["called"]


def test_require_research_host_exits_when_heartbeat_stale(monkeypatch):
    """``--require-research-host`` / TRADER_IN_PROCESS_SCORER=never hard-fail."""
    import __main__ as trader_main

    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_alive", lambda *a, **k: False
    )
    monkeypatch.setattr(
        "core.runtime.heartbeat.heartbeat_age_s", lambda *a, **k: float("inf")
    )
    monkeypatch.setattr("core.config.TRADER_IN_PROCESS_SCORER_NEVER", False, raising=False)
    monkeypatch.setattr("signals.scorer.run_research_threaded", MagicMock())

    with pytest.raises(SystemExit) as exc:
        trader_main._configure_scorer(_trader_args(require_research_host=True))
    assert exc.value.code == 2


def test_trader_in_process_scorer_never_env_exits_without_heartbeat(monkeypatch):
    import __main__ as trader_main

    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_alive", lambda *a, **k: False
    )
    monkeypatch.setattr("core.runtime.heartbeat.heartbeat_age_s", lambda *a, **k: 999.0)
    monkeypatch.setattr("core.config.TRADER_IN_PROCESS_SCORER_NEVER", True, raising=False)
    monkeypatch.setattr("signals.scorer.run_research_threaded", MagicMock())

    with pytest.raises(SystemExit) as exc:
        trader_main._configure_scorer(_trader_args())
    assert exc.value.code == 2


def test_force_in_process_starts_scorer_even_with_fresh_heartbeat(monkeypatch):
    """Dev override: ``--force-in-process`` may double-write with research host."""
    import __main__ as trader_main

    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_alive", lambda *a, **k: True
    )
    threaded: list[str] = []

    def _threaded(**_kw):
        threaded.append("called")

    monkeypatch.setattr("signals.scorer.run_research_threaded", _threaded)

    trader_main._configure_scorer(_trader_args(force_in_process=True))
    assert threaded == ["called"]


def test_operating_context_sync_follows_heartbeat(monkeypatch):
    from core.runtime.operating_context import get_operating_context
    from core.runtime.heartbeat import write_heartbeat

    ctx = get_operating_context()
    write_heartbeat(now=time.time())
    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_alive",
        lambda *a, **k: True,
    )
    assert ctx.sync_researcher_from_heartbeat() is True
    assert ctx.quality.researcher_available is True
    assert ctx.is_independent_mode is False

    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_alive",
        lambda *a, **k: False,
    )
    assert ctx.sync_researcher_from_heartbeat() is False
    assert ctx.is_independent_mode is True
