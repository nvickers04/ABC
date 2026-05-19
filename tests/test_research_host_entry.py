"""Research host entry point (``python -m research``) — mocked external IO."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quality_test_support import reset_quality_runtime_state


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


@pytest.fixture(autouse=True)
def _clean_state():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.asyncio
async def test_run_invokes_scorer_not_grok(monkeypatch):
    """``_run`` wires scorer only (no Grok trader, no IBKR orders)."""
    from research import host

    init_db = MagicMock()
    monkeypatch.setattr("memory.init_db", init_db)
    run_research = AsyncMock()
    monkeypatch.setattr("signals.scorer.run_research", run_research)
    monkeypatch.setattr(
        "signals.template_evolution.run_template_evolution_threaded",
        MagicMock(),
    )

    await host._run(verbose=False, run_evolution=True)

    init_db.assert_called_once()
    run_research.assert_awaited_once_with(verbose=False, use_cadence=True)


def test_main_mda_health_failure_exits(monkeypatch):
    from research import host

    monkeypatch.setattr(
        "core.entry_cli.parse_research_args",
        lambda: SimpleNamespace(verbose=False, no_evolution=True),
    )
    monkeypatch.setattr("core.log_setup.configure_root_logging", MagicMock())
    monkeypatch.setattr(host, "ensure_utf8_stdio", MagicMock())
    monkeypatch.setattr(host, "bind_research_host_context", MagicMock())
    monkeypatch.setattr(host, "log_banner", MagicMock())

    dp = MagicMock()
    dp.get_quote.return_value = None
    monkeypatch.setattr("data.data_provider.get_data_provider", lambda: dp)
    monkeypatch.setattr("core.config.RESEARCHER_MDA_HEALTH_CHECK_ENABLED", True)

    with pytest.raises(SystemExit) as exc:
        host.main()
    assert exc.value.code == 3


def test_main_token_cap_exceeded_exits(monkeypatch):
    from research import host

    monkeypatch.setattr(
        "core.entry_cli.parse_research_args",
        lambda: SimpleNamespace(verbose=False, no_evolution=True),
    )
    monkeypatch.setattr("core.log_setup.configure_root_logging", MagicMock())
    monkeypatch.setattr(host, "ensure_utf8_stdio", MagicMock())
    monkeypatch.setattr(host, "bind_research_host_context", MagicMock())
    monkeypatch.setattr(host, "log_banner", MagicMock())

    dp = MagicMock()
    dp.get_quote.return_value = {"last": 500.0}
    monkeypatch.setattr("data.data_provider.get_data_provider", lambda: dp)
    monkeypatch.setattr("core.config.RESEARCHER_MDA_HEALTH_CHECK_ENABLED", True)
    monkeypatch.setattr("core.config.RESEARCHER_DAILY_TOKEN_CAP", 100)

    def _usage(_key, default=0.0):
        return 150.0

    monkeypatch.setattr("memory.get_research_config", _usage)

    with pytest.raises(SystemExit) as exc:
        host.main()
    assert exc.value.code == 4


def test_main_passes_no_evolution_to_run(monkeypatch):
    from research import host

    monkeypatch.setattr(
        "core.entry_cli.parse_research_args",
        lambda: SimpleNamespace(verbose=True, no_evolution=True),
    )
    monkeypatch.setattr("core.log_setup.configure_root_logging", MagicMock())
    monkeypatch.setattr(host, "ensure_utf8_stdio", MagicMock())
    monkeypatch.setattr(host, "bind_research_host_context", MagicMock())
    monkeypatch.setattr(host, "log_banner", MagicMock())
    monkeypatch.setattr("core.config.RESEARCHER_MDA_HEALTH_CHECK_ENABLED", False)

    run_mock = AsyncMock()
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(host.asyncio, "new_event_loop", lambda: loop)

    with patch.object(host, "_run", run_mock):
        host.main()

    run_mock.assert_called_once_with(verbose=True, run_evolution=False)


def test_research_module_exposes_host_main():
    from research.host import main as host_main
    from research import __main__ as research_entry

    assert research_entry.main is host_main
