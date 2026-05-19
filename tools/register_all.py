"""Attach all tool module handlers to :class:`core.tool_registry.ToolRegistry`.

This is the only place handler maps are merged into the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.tool_registry import ToolRegistry


def register_handlers(registry: ToolRegistry, handlers: dict) -> None:
    """Bind a module's handler dict into the registry."""
    registry.bind_handlers(handlers)


def attach_all_handlers(registry: ToolRegistry) -> None:
    """Register every agent tool handler from tools/* submodules."""
    from tools.tools_account import register_handlers as reg_account
    from tools.tools_instruments import register_handlers as reg_instruments
    from tools.tools_multiagent import register_handlers as reg_multiagent
    from tools.tools_options import register_handlers as reg_options
    from tools.tools_orders import register_handlers as reg_orders
    from tools.tools_research import register_handlers as reg_research
    from tools.tools_signal_breakdown import register_handlers as reg_signal
    from tools.tools_sizing import register_handlers as reg_sizing
    from tools.tools_stats import register_handlers as reg_stats
    from tools.tools_working_memory import register_handlers as reg_wm

    reg_research(registry)
    reg_account(registry)
    reg_orders(registry)
    reg_options(registry)
    reg_stats(registry)
    reg_sizing(registry)
    reg_instruments(registry)
    reg_multiagent(registry)
    reg_wm(registry)
    reg_signal(registry)

    # Executor-hosted planning tools (need ToolExecutor instance methods)
    from tools.tools_executor import _handle_enter_option, _handle_plan_order

    registry.bind_handler("plan_order", _handle_plan_order)
    registry.bind_handler("enter_option", _handle_enter_option)
