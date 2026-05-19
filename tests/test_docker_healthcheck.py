"""Unit tests for scripts/docker_healthcheck.py (mocked deps)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_usage_message_on_bad_role():
    from scripts import docker_healthcheck as dh

    with pytest.raises(SystemExit) as exc:
        dh.main()
    assert exc.value.code == 1


def test_check_research_ok(monkeypatch):
    from scripts import docker_healthcheck as dh

    monkeypatch.setattr(dh, "_load_env", lambda: None)
    monkeypatch.setattr(
        "memory.get_db",
        lambda: MagicMock(execute=MagicMock(return_value=MagicMock(fetchone=lambda: (1,)))),
    )
    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_operational",
        lambda *a, **k: True,
    )
    assert dh.check_research() == 0


def test_check_trader_requires_research_when_env_set(monkeypatch):
    from scripts import docker_healthcheck as dh

    monkeypatch.setenv("DOCKER_HEALTHCHECK_REQUIRE_RESEARCH", "1")
    monkeypatch.setattr(dh, "_load_env", lambda: None)
    with (
        patch("core.config.validate_config", return_value=[]),
        patch(
            "memory.get_db",
            return_value=MagicMock(
                execute=MagicMock(return_value=MagicMock(fetchone=lambda: (1,)))
            ),
        ),
        patch(
            "core.runtime.heartbeat.is_research_host_operational",
            return_value=False,
        ),
    ):
        assert dh.check_trader() == 1
