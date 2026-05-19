"""Thread-local ProfitConfig overrides for parallel optimizer backtests.

Parallel workers push a :class:`~core.central_profit_config.ComposedProfitConfig` for the
duration of one simulation so ``get_*_config()`` reads per-thread values without mutating
the process singleton or other threads' active config.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.central_profit_config import ComposedProfitConfig
    from core.loop_config import LoopConfig
    from core.memory_config import MemoryConfig
    from core.prompt_config import PromptConfig
    from core.risk_execution_config import RiskExecutionConfig
    from core.tool_registry import ToolRegistry

_tls = threading.local()


def _stacks() -> dict[str, list[Any]]:
    stacks = getattr(_tls, "stacks", None)
    if stacks is None:
        stacks = {}
        _tls.stacks = stacks
    return stacks


def push_composed_config(composed: ComposedProfitConfig) -> None:
    """Install ``composed`` for the current thread only (stacked)."""
    stacks = _stacks()
    stacks.setdefault("risk", []).append(composed.risk)
    stacks.setdefault("loop", []).append(composed.loop)
    stacks.setdefault("memory", []).append(composed.memory)
    stacks.setdefault("prompt", []).append(composed.prompt)
    stacks.setdefault("tools", []).append(composed.tools)


def pop_composed_config() -> None:
    """Restore the previous thread-local config frame."""
    stacks = _stacks()
    for key in ("risk", "loop", "memory", "prompt", "tools"):
        st = stacks.get(key)
        if not st:
            raise RuntimeError(f"profit_config_context: empty stack for {key!r}")
        st.pop()
        if not st:
            stacks.pop(key, None)


def get_thread_risk_config() -> RiskExecutionConfig | None:
    st = _stacks().get("risk")
    return st[-1] if st else None


def get_thread_loop_config() -> LoopConfig | None:
    st = _stacks().get("loop")
    return st[-1] if st else None


def get_thread_memory_config() -> MemoryConfig | None:
    st = _stacks().get("memory")
    return st[-1] if st else None


def get_thread_prompt_config() -> PromptConfig | None:
    st = _stacks().get("prompt")
    return st[-1] if st else None


def get_thread_tool_registry() -> ToolRegistry | None:
    st = _stacks().get("tools")
    return st[-1] if st else None


def clear_thread_config() -> None:
    """Drop thread-local stacks (tests)."""
    if hasattr(_tls, "stacks"):
        del _tls.stacks


__all__ = [
    "clear_thread_config",
    "get_thread_loop_config",
    "get_thread_memory_config",
    "get_thread_prompt_config",
    "get_thread_risk_config",
    "get_thread_tool_registry",
    "pop_composed_config",
    "push_composed_config",
]
