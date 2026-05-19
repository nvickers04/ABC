"""Master composition of all profitability-related configuration.

Load via :func:`load_profit_config` / :func:`get_profit_config` at process startup
(``__main__.py``, ``research/host.py``). Sub-configs remain the source of truth;
this module only composes them and syncs :mod:`core.config` module exports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from core.loop_config import LoopConfig, install_loop_config, reload_loop_config
from core.memory_config import MemoryConfig, install_memory_config, reload_memory_config
from core.profit_profiles import (
    PROFIT_PROFILE_ENV,
    ProfitProfile,
    apply_loop_profile,
    apply_memory_profile,
    apply_prompt_profile,
    apply_risk_profile,
    apply_tool_profile,
    normalize_profit_profile,
    profile_change_lines,
    profile_note,
)
from core.prompt_config import PromptConfig, install_prompt_config, reload_prompt_config
from core.risk_execution_config import (
    RiskExecutionConfig,
    install_risk_execution_config,
    reload_risk_execution_config,
)
from core.tool_registry import (
    ToolRegistry,
    get_tool_registry,
    install_tool_registry,
    reset_tool_registry_for_tests,
)


def _profitability_note(field_info: FieldInfo | None) -> str:
    if field_info is None:
        return ""
    raw = field_info.description or ""
    for prefix in ("Profitability: ", "profitability: "):
        if raw.startswith(prefix):
            return raw[len(prefix) :].strip()
    return raw.strip()


def _format_value(val: Any, *, max_len: int = 120) -> str:
    if isinstance(val, dict):
        if len(val) <= 6:
            return repr(val)
        return f"<dict len={len(val)}>"
    if isinstance(val, (list, tuple)):
        if len(val) <= 8:
            return repr(val)
        return f"<{type(val).__name__} len={len(val)}>"
    s = repr(val)
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _print_model_section(title: str, model: BaseModel) -> None:
    print(f"## {title}")
    for name, field_info in model.model_fields.items():
        val = getattr(model, name)
        note = _profitability_note(field_info)
        print(f"  {name} = {_format_value(val)}")
        if note:
            print(f"    -> P&L: {note}")
    print()


def _print_tool_registry_section(registry: ToolRegistry) -> None:
    print("## Tool registry")
    tools = list(registry.tools.values())
    enabled = [t for t in tools if t.enabled]
    mutating = [t for t in tools if t.mutates_broker]
    print(f"  tools_registered = {len(tools)}")
    print(f"  tools_enabled = {len(enabled)}")
    print(f"  tools_mutating_broker = {len(mutating)}")
    print("    -> P&L: catalog of agent tools; weights and blocks steer spend vs edge.")
    print()
    for spec in sorted(tools, key=lambda t: (-t.profitability_weight, t.name)):
        note_parts = [
            f"weight={spec.profitability_weight:.1f}",
            f"tier={spec.cost_tier}",
            f"cat={spec.category}",
        ]
        if spec.mutates_broker:
            note_parts.append("mutates")
        if not spec.enabled:
            note_parts.append("DISABLED")
        line = " ".join(note_parts)
        desc = (spec.schema_description or "").split("\n")[0][:80]
        print(f"  {spec.name}: {line}")
        if desc:
            print(f"    -> P&L: {desc}")
    print()


@dataclass(frozen=True)
class ProfitConfig:
    """Composed view of prompt, tools, memory, loop, and risk/execution settings."""

    prompt: PromptConfig
    tools: ToolRegistry
    memory: MemoryConfig
    loop: LoopConfig
    risk: RiskExecutionConfig

    @property
    def trading_mode(self) -> str:
        return str(self.risk.trading_mode)

    def summary(self) -> None:
        """Print every configured lever with value and one-line profitability note."""
        print("=" * 72)
        print("ProfitConfig — all profitability levers")
        print(f"  trading_mode = {self.trading_mode!r}")
        print(f"  ibkr_account_type = {self.risk.ibkr_account_type!r}")
        print(f"  tools_registered = {len(self.tools.tools)}")
        print("=" * 72)
        print()
        _print_model_section("Risk & execution", self.risk)
        _print_model_section("Agent loop (ReAct / scheduler / gates)", self.loop)
        _print_model_section("Memory & prompt budget", self.memory)
        _print_model_section("LLM prompts & quality sampling", self.prompt)
        _print_tool_registry_section(self.tools)
        print(
            "Env tips: TRADING_MODE, RISK_PER_TRADE, MAX_DAILY_LLM_COST, CASH_ONLY, "
            "CYCLE_* caps, TOOL_REGISTRY_DISABLE / TOOL_REGISTRY_WEIGHTS, PROFIT_PROFILE."
        )

    def optimize_for_profit(
        self,
        profile: str = "balanced",
        *,
        verbose: bool = True,
    ) -> ProfitConfig:
        """Apply a validated profitability profile across all five sub-configs.

        Profiles: ``conservative``, ``balanced`` (env defaults only), ``aggressive``.
        Persists ``PROFIT_PROFILE`` in the environment and reloads singletons.
        """
        norm = normalize_profit_profile(profile)
        os.environ[PROFIT_PROFILE_ENV] = norm
        refreshed = reload_profit_config()
        if verbose:
            print("=" * 72)
            print(f"ProfitConfig.optimize_for_profit({norm!r})")
            print(f"  {profile_note(norm)}")
            print("=" * 72)
            for line in profile_change_lines(
                norm,
                risk=refreshed.risk,
                loop=refreshed.loop,
                memory=refreshed.memory,
                prompt=refreshed.prompt,
                tools=refreshed.tools,
            ):
                print(line)
            print()
        return refreshed


def _active_profit_profile() -> ProfitProfile | None:
    raw = os.getenv(PROFIT_PROFILE_ENV, "").strip().lower()
    if not raw or raw == "balanced":
        return None
    return normalize_profit_profile(raw)


def _apply_profile_to_profit_config(cfg: ProfitConfig, profile: ProfitProfile) -> ProfitConfig:
    risk = apply_risk_profile(cfg.risk, profile)
    memory = apply_memory_profile(cfg.memory, profile)
    loop = apply_loop_profile(cfg.loop, profile)
    prompt = apply_prompt_profile(cfg.prompt, profile)
    tools = cfg.tools
    apply_tool_profile(tools, profile)
    install_risk_execution_config(risk)
    install_memory_config(memory)
    install_loop_config(loop)
    install_prompt_config(prompt)
    install_tool_registry(tools)
    return replace(cfg, risk=risk, memory=memory, loop=loop, prompt=prompt, tools=tools)


def _clear_profile_installs() -> None:
    install_risk_execution_config(None)
    install_memory_config(None)
    install_loop_config(None)
    install_prompt_config(None)
    install_tool_registry(None)


def _sync_core_config_exports(risk: RiskExecutionConfig) -> None:
    """Refresh :mod:`core.config` module-level aliases after sub-config reload."""
    import core.config as cfg

    cfg._sync_risk_module_exports(risk)
    cfg.rebuild_prompt_exports()


def build_profit_config() -> ProfitConfig:
    """Load all sub-config singletons (call after env / CLI overrides are set)."""
    _clear_profile_installs()
    risk = reload_risk_execution_config()
    memory = reload_memory_config()
    loop = reload_loop_config()
    prompt = reload_prompt_config()
    tools = get_tool_registry(reload=True)
    cfg = ProfitConfig(
        prompt=prompt,
        tools=tools,
        memory=memory,
        loop=loop,
        risk=risk,
    )
    profile = _active_profit_profile()
    if profile is not None:
        cfg = _apply_profile_to_profit_config(cfg, profile)
        risk = cfg.risk
    _sync_core_config_exports(risk)
    return cfg


@lru_cache(maxsize=1)
def get_profit_config() -> ProfitConfig:
    """Return the cached :class:`ProfitConfig` (built on first access)."""
    return build_profit_config()


def reload_profit_config() -> ProfitConfig:
    """Clear all config caches and rebuild the master config."""
    get_profit_config.cache_clear()
    reset_tool_registry_for_tests()
    return get_profit_config()


def load_profit_config(*, dotenv: bool = True) -> ProfitConfig:
    """Load ``.env``, apply fresh settings, sync ``core.config``, return master config."""
    if dotenv:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            pass
    return reload_profit_config()


__all__ = [
    "PROFIT_PROFILE_ENV",
    "ProfitConfig",
    "build_profit_config",
    "get_profit_config",
    "load_profit_config",
    "reload_profit_config",
]
