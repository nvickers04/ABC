"""ToolRegistry metadata and experiment toggles."""

from __future__ import annotations

import os

import pytest

from core.tool_registry import get_tool_registry, reset_tool_registry_for_tests


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_tool_registry_for_tests()
    yield
    reset_tool_registry_for_tests()


def test_registry_has_agent_tools():
    reg = get_tool_registry()
    names = reg.agent_action_names()
    assert "quote" in names
    assert "market_order" in names
    assert len(names) >= 90


def test_disable_and_weight_env(monkeypatch):
    monkeypatch.setenv("TOOL_REGISTRY_DISABLE", "research,flatten_limits")
    monkeypatch.setenv("TOOL_REGISTRY_WEIGHTS", '{"quote": 3.5}')
    reset_tool_registry_for_tests()
    reg = get_tool_registry()
    assert not reg.is_enabled("research")
    assert reg.effective_weight("quote") == 3.5
    assert reg.effective_weight("research") == 0.0


def test_provider_capabilities_not_agent_callable():
    reg = get_tool_registry()
    spec = reg.get_spec("data.get_quote")
    assert spec is not None
    assert spec.agent_callable is False
