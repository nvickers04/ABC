"""Central registry for agent tools and underlying data/execution capabilities.

All agent-callable tools register handlers via :func:`tools.register_all.attach_all_handlers`.
Metadata (schema, profitability weight, validation) lives here so experiments can
enable/disable or re-weight tools without editing handler modules.

Environment overrides (optional):
  ``TOOL_REGISTRY_DISABLE`` — comma-separated tool names to disable
  ``TOOL_REGISTRY_WEIGHTS`` — JSON object ``{"research": 0.5, "market_order": 3}``
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator

ToolPackage = Literal["tools", "execution", "data"]
ToolCategory = Literal[
    "research",
    "knowledge",
    "planning",
    "account",
    "order_control",
    "stock_orders",
    "option_orders",
    "observability",
    "working_memory",
    "provider",
]
CostTier = Literal["free", "low", "medium", "high", "llm"]

# ReAct loop actions handled inside ToolExecutor (not separate handlers).
_EXECUTOR_PASSTHROUGH_ACTIONS: frozenset[str] = frozenset(
    {"buy", "sell", "wait", "think", "feedback", "done"}
)


class ValidationRuleKind(str, Enum):
    """Host-side checks applied before dispatch."""

    REQUIRES_BROKER = "requires_broker"
    REQUIRES_SYMBOL = "requires_symbol"
    EMERGENCY_ONLY = "emergency_only"
    UNIVERSE_ENTRY_GUARD = "universe_entry_guard"
    QUALITY_MATRIX_GATE = "quality_matrix_gate"
    CASH_ONLY_NO_SHORT_STOCK = "cash_only_no_short_stock"


class ToolValidationRule(BaseModel):
    """Single validation rule attached to a tool spec."""

    kind: ValidationRuleKind
    description: str = Field(
        default="",
        description="Profitability: documents why this rule exists (risk/cost/safety).",
    )


class ToolSpec(BaseModel):
    """Metadata for one registered tool or provider capability."""

    name: str = Field(description="Canonical action name (agent JSON ``action`` field).")
    schema_description: str = Field(
        description="Profitability: parameter schema shown to the model; bad schemas cause bad trades.",
    )
    profitability_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Profitability: relative priority in playbook sorting and experiment overlays; "
            "higher weights surface execution tools during prompt budget trims."
        ),
    )
    enabled: bool = Field(
        default=True,
        description="Profitability: when False the agent cannot call this tool (zero risk/cost).",
    )
    validation_rules: list[ToolValidationRule] = Field(default_factory=list)
    package: ToolPackage = Field(
        default="tools",
        description="Owning package: tools (agent), data (MDA/IBKR reads), execution (broker).",
    )
    category: ToolCategory = Field(default="research")
    agent_callable: bool = Field(
        default=True,
        description="Profitability: False for provider stubs the agent never calls directly.",
    )
    cost_tier: CostTier = Field(
        default="low",
        description="Profitability: llm/high tools burn budget; free tools should be preferred first.",
    )
    mutates_broker: bool = Field(
        default=False,
        description="Profitability: True when tool can change positions/orders (real P&L impact).",
    )
    underlying_provider: str | None = Field(
        default=None,
        description="Profitability: data/execution method backing this tool (trace slippage/cost).",
    )

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        return str(v).strip().lower()


class ToolRegistry(BaseModel):
    """Process-wide tool catalog with optional handler bindings."""

    model_config = {"arbitrary_types_allowed": True}

    tools: dict[str, ToolSpec] = Field(default_factory=dict)
    handlers: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # ── Registration ─────────────────────────────────────────────────────

    def register_spec(self, spec: ToolSpec) -> None:
        """Add or replace tool metadata (no handler)."""
        self.tools[spec.name] = spec

    def bind_handler(self, name: str, handler: Callable[..., Any]) -> None:
        """Attach an async handler; auto-create a minimal spec if missing."""
        key = name.strip().lower()
        if key not in self.tools:
            cat = _CATEGORY_MAP.get(key, "research")
            mutates = key in _ORDER_ACTION_NAMES and key not in ("plan_order", "enter_option")
            self.register_spec(
                ToolSpec(
                    name=key,
                    schema_description=f"{{}} -> {key} (auto-registered)",
                    profitability_weight=_default_weight(key, cat, mutates),
                    package="tools",
                    category=cat,
                    agent_callable=True,
                    cost_tier=_cost_tier(key),
                    mutates_broker=mutates,
                    underlying_provider=_underlying_provider(key),
                )
            )
        self.handlers[key] = handler

    def bind_handlers(self, mapping: dict[str, Callable[..., Any]]) -> None:
        """Bind many handlers at once."""
        for name, handler in mapping.items():
            self.bind_handler(name, handler)

    # ── Experiment toggles ───────────────────────────────────────────────

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a tool by name."""
        key = name.strip().lower()
        if key not in self.tools:
            raise KeyError(key)
        self.tools[key].enabled = enabled

    def set_weight(self, name: str, weight: float) -> None:
        """Set profitability weight (0–10)."""
        key = name.strip().lower()
        if key not in self.tools:
            raise KeyError(key)
        self.tools[key].profitability_weight = float(weight)

    def apply_env_overrides(self) -> None:
        """Apply ``TOOL_REGISTRY_DISABLE`` and ``TOOL_REGISTRY_WEIGHTS`` from the environment."""
        raw_disable = os.getenv("TOOL_REGISTRY_DISABLE", "").strip()
        if raw_disable:
            for part in raw_disable.split(","):
                name = part.strip().lower()
                if name and name in self.tools:
                    self.set_enabled(name, False)

        raw_weights = os.getenv("TOOL_REGISTRY_WEIGHTS", "").strip()
        if raw_weights:
            try:
                weights = json.loads(raw_weights)
                if isinstance(weights, dict):
                    for name, w in weights.items():
                        if isinstance(name, str) and name.lower() in self.tools:
                            self.set_weight(name, float(w))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # ── Queries ──────────────────────────────────────────────────────────

    def get_spec(self, name: str) -> ToolSpec | None:
        return self.tools.get(name.strip().lower())

    def get_handler(self, name: str) -> Callable[..., Any] | None:
        return self.handlers.get(name.strip().lower())

    def is_enabled(self, name: str) -> bool:
        spec = self.get_spec(name)
        return bool(spec and spec.enabled)

    def effective_weight(self, name: str) -> float:
        spec = self.get_spec(name)
        if not spec or not spec.enabled:
            return 0.0
        return float(spec.profitability_weight)

    def agent_action_names(self) -> list[str]:
        """Sorted agent-callable tools that are enabled and have handlers."""
        return sorted(
            n
            for n, spec in self.tools.items()
            if spec.agent_callable and spec.enabled and n in self.handlers
        )

    def valid_action_names(self) -> list[str]:
        """All agent tools with handlers (enabled or not)."""
        return sorted(
            n
            for n, spec in self.tools.items()
            if spec.agent_callable and n in self.handlers
        )

    def enabled_agent_tools(self) -> dict[str, ToolSpec]:
        return {
            n: s
            for n, s in self.tools.items()
            if s.agent_callable and s.enabled
        }

    def validate_dispatch(
        self,
        name: str,
        params: dict[str, Any] | None,
        *,
        gateway_connected: bool,
    ) -> str | None:
        """Return an error string if validation fails, else None."""
        key = name.strip().lower()
        if key in _EXECUTOR_PASSTHROUGH_ACTIONS:
            return None
        spec = self.get_spec(name)
        if spec is None:
            return f"Unknown tool: {name}"
        if not spec.enabled:
            return f"Tool '{name}' is disabled by ToolRegistry configuration"
        if not spec.agent_callable:
            return f"Tool '{name}' is not agent-callable (provider capability only)"

        params = params or {}
        for rule in spec.validation_rules:
            if rule.kind == ValidationRuleKind.REQUIRES_BROKER and not gateway_connected:
                return f"Tool '{name}' requires broker connection"
        return None

    def playbook_lines(self, *, max_tools: int | None = None) -> dict[str, str]:
        """Compact intent lines sorted by descending profitability weight."""
        items = [
            (n, s)
            for n, s in self.tools.items()
            if s.agent_callable and s.enabled and n in self.handlers
        ]
        items.sort(key=lambda x: (-x[1].profitability_weight, x[0]))
        if max_tools is not None:
            items = items[:max_tools]
        return {n: s.schema_description.split(" -> ", 1)[-1][:120] for n, s in items}

    @classmethod
    def build_default(cls) -> ToolRegistry:
        """Construct registry metadata from executor docstring + provider stubs."""
        reg = cls()
        schemas = _parse_executor_docstring()
        categories = _category_for_names(set(schemas))
        order_actions = _ORDER_ACTION_NAMES
        emergency = frozenset({"flatten_limits", "cancel_all_orphans"})

        for name, schema in sorted(schemas.items()):
            cat = categories.get(name, "research")
            mutates = name in order_actions and name not in ("plan_order", "enter_option")
            rules: list[ToolValidationRule] = [
                ToolValidationRule(
                    kind=ValidationRuleKind.QUALITY_MATRIX_GATE,
                    description="Host QualityMatrix may block or scale (profitability guard).",
                ),
            ]
            if mutates:
                rules.append(
                    ToolValidationRule(
                        kind=ValidationRuleKind.REQUIRES_BROKER,
                        description="Profitability: needs live IBKR session to affect P&L.",
                    )
                )
                if name not in _EXIT_ORDER_ACTIONS:
                    rules.append(
                        ToolValidationRule(
                            kind=ValidationRuleKind.UNIVERSE_ENTRY_GUARD,
                            description="Profitability: entries confined to research universe.",
                        )
                    )
            if name in emergency:
                rules.append(
                    ToolValidationRule(
                        kind=ValidationRuleKind.EMERGENCY_ONLY,
                        description="Profitability: misuse causes accidental flatten/cancel damage.",
                    )
                )
            if "symbol" in schema.lower() and name not in ("market_hours", "budget", "stats"):
                rules.append(
                    ToolValidationRule(
                        kind=ValidationRuleKind.REQUIRES_SYMBOL,
                        description="Profitability: symbol required to avoid accidental wrong-name orders.",
                    )
                )

            reg.register_spec(
                ToolSpec(
                    name=name,
                    schema_description=schema,
                    profitability_weight=_default_weight(name, cat, mutates),
                    enabled=True,
                    validation_rules=rules,
                    package="tools",
                    category=cat,
                    agent_callable=True,
                    cost_tier=_cost_tier(name),
                    mutates_broker=mutates,
                    underlying_provider=_underlying_provider(name),
                )
            )

        _register_provider_capabilities(reg)
        return reg


# ── Defaults helpers ───────────────────────────────────────────────────────

_ORDER_ACTION_NAMES: frozenset[str] = frozenset(
    {
        "market_order", "limit_order", "stop_order", "stop_limit",
        "trailing_stop", "bracket_order", "modify_stop", "oca_order",
        "flatten_limits", "moc_order", "loc_order", "moo_order", "loo_order",
        "trailing_stop_limit", "adaptive_order", "midprice_order", "relative_order",
        "gtd_order", "fok_order", "ioc_order", "vwap_order", "twap_order",
        "iceberg_order", "snap_mid_order", "close_position",
        "buy_option", "covered_call", "cash_secured_put", "protective_put",
        "vertical_spread", "iron_condor", "iron_butterfly", "straddle",
        "strangle", "collar", "calendar_spread", "diagonal_spread",
        "butterfly", "ratio_spread", "jade_lizard", "close_option", "close_spread",
        "roll_option", "plan_order", "enter_option", "multi_leg",
        "cancel_order", "cancel_stops", "cancel_all_orphans",
    }
)

_EXIT_ORDER_ACTIONS: frozenset[str] = frozenset(
    {
        "close_position", "close_option", "close_spread", "roll_option",
        "modify_stop", "flatten_limits", "cancel_order", "cancel_stops",
    }
)

_CATEGORY_MAP: dict[str, ToolCategory] = {}
for _n in (
    "quote", "candles", "fundamentals", "earnings", "atr", "iv_info", "news",
    "analysts", "extended_fundamentals", "institutional_data", "insider_data",
    "peer_comparison", "chart_intraday", "chart_swing", "chart_full", "chart_quick",
    "option_chain", "option_quote", "option_greeks", "position_greeks",
    "research", "research_engine", "prior_research", "briefing", "context_quality",
):
    _CATEGORY_MAP[_n] = "research"
for _n in ("market_hours", "budget", "economic_calendar"):
    _CATEGORY_MAP[_n] = "knowledge"
for _n in ("plan_order", "enter_option", "calculate_size", "instrument_selector"):
    _CATEGORY_MAP[_n] = "planning"
for _n in ("account", "positions", "open_orders", "get_position", "refresh_state"):
    _CATEGORY_MAP[_n] = "account"
for _n in ("cancel_order", "cancel_stops", "cancel_all_orphans", "flatten_limits", "modify_stop"):
    _CATEGORY_MAP[_n] = "order_control"
for _n in _ORDER_ACTION_NAMES:
    if _n not in _CATEGORY_MAP:
        if _n in (
            "buy_option", "vertical_spread", "iron_condor", "iron_butterfly",
            "straddle", "strangle", "collar", "calendar_spread", "diagonal_spread",
            "butterfly", "ratio_spread", "jade_lizard", "close_option", "close_spread",
            "roll_option", "multi_leg", "covered_call", "cash_secured_put", "protective_put",
            "enter_option", "option_chain", "option_greeks", "option_quote", "position_greeks",
        ):
            _CATEGORY_MAP[_n] = "option_orders"
        elif _n not in ("plan_order", "enter_option"):
            _CATEGORY_MAP[_n] = "stock_orders"
for _n in (
    "stats", "daily_summary", "review_trades", "execution_status", "trader_rules",
    "open_hypotheses", "signal_breakdown", "quality_status", "quality_for_symbol",
    "provenance_audit", "current_constraints",
):
    _CATEGORY_MAP[_n] = "observability"
for _n in ("update_working_memory", "clear_working_memory_entry"):
    _CATEGORY_MAP[_n] = "working_memory"


def _category_for_names(names: set[str]) -> dict[str, ToolCategory]:
    out: dict[str, ToolCategory] = {}
    for n in names:
        out[n] = _CATEGORY_MAP.get(n, "research")
    return out


def _default_weight(name: str, category: ToolCategory, mutates: bool) -> float:
    if name == "research":
        return 0.4
    if category == "observability":
        return 0.6
    if category == "working_memory":
        return 0.7
    if category == "knowledge":
        return 0.8
    if category == "planning":
        return 1.8
    if mutates:
        return 2.5 if category == "stock_orders" else 2.2
    if category == "research":
        return 1.0
    return 1.0


def _cost_tier(name: str) -> CostTier:
    if name in ("research",):
        return "llm"
    if name in ("chart_full", "extended_fundamentals", "institutional_data", "option_chain"):
        return "high"
    if name in ("briefing", "candles", "fundamentals", "peer_comparison"):
        return "medium"
    if name in ("quote", "atr", "market_hours", "budget", "stats"):
        return "low"
    return "low"


def _underlying_provider(name: str) -> str | None:
    _MAP = {
        "quote": "data.DataProvider.get_quote",
        "candles": "data.DataProvider.get_candles",
        "atr": "data.DataProvider.get_atr",
        "fundamentals": "data.DataProvider.get_fundamentals",
        "earnings": "data.DataProvider.get_earnings_info",
        "iv_info": "data.DataProvider.get_iv_info",
        "news": "data.DataProvider.get_news",
        "analysts": "data.DataProvider.get_analyst_data",
        "option_chain": "data.DataProvider.get_option_chain",
        "market_order": "execution.IBKRGateway.place_order",
        "limit_order": "execution.IBKRGateway.place_order",
        "positions": "execution.IBKRGateway.get_positions",
        "account": "execution.IBKRGateway.get_account_summary",
    }
    return _MAP.get(name)


def _parse_executor_docstring() -> dict[str, str]:
    """Parse ``tools/tools_executor.py`` module docstring for tool schemas."""
    path = Path(__file__).resolve().parents[1] / "tools" / "tools_executor.py"
    content = path.read_text(encoding="utf-8")
    m = re.match(r'^.*?"""\n(.*?)"""\n\nimport ', content, re.DOTALL)
    if not m:
        return {}
    body = m.group(1)
    schemas: dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        hit = re.match(r"^([a-z_][a-z0-9_]*)\s*:\s*(\{[^}]*\}|{})\s*->\s*(.+)$", line, re.I)
        if hit:
            name = hit.group(1).lower()
            schemas[name] = f"{hit.group(2)} -> {hit.group(3)}"
    return schemas


def _register_provider_capabilities(reg: ToolRegistry) -> None:
    """Document data/execution methods not directly agent-callable."""
    providers: list[tuple[str, str, str, ToolPackage]] = [
        ("data.get_quote", "{symbol}", "data.DataProvider.get_quote", "data"),
        ("data.get_candles", "{symbol, days?, resolution?}", "data.DataProvider.get_candles", "data"),
        ("data.get_atr", "{symbol, period?}", "data.DataProvider.get_atr", "data"),
        ("data.get_option_chain", "{symbol, expiration?, ...}", "data.DataProvider.get_option_chain", "data"),
        ("data.get_iv_info", "{symbol, dte_min?, dte_max?}", "data.DataProvider.get_iv_info", "data"),
        ("data.get_fundamentals", "{symbol}", "data.DataProvider.get_fundamentals", "data"),
        ("data.get_mda_usage", "{}", "data.DataProvider.get_mda_usage", "data"),
        ("execution.connect", "{}", "execution.IBKRGateway.connect", "execution"),
        ("execution.disconnect", "{}", "execution.IBKRGateway.disconnect", "execution"),
        ("execution.get_positions", "{}", "execution.IBKRGateway.get_positions", "execution"),
        ("execution.place_order", "{contract, order}", "execution.IBKRGateway.place_order", "execution"),
    ]
    for name, schema, provider, package in providers:
        reg.register_spec(
            ToolSpec(
                name=name,
                schema_description=schema,
                profitability_weight=0.5,
                enabled=True,
                package=package,
                category="provider",
                agent_callable=False,
                cost_tier="low",
                underlying_provider=provider,
            )
        )


_registry: ToolRegistry | None = None


def get_tool_registry(*, reload: bool = False) -> ToolRegistry:
    """Return the process-wide :class:`ToolRegistry` with handlers attached."""
    global _registry
    if _registry is None or reload:
        reg = ToolRegistry.build_default()
        from tools.register_all import attach_all_handlers

        attach_all_handlers(reg)
        reg.apply_env_overrides()
        _registry = reg
    return _registry


def reset_tool_registry_for_tests() -> None:
    """Clear singleton (tests)."""
    global _registry
    _registry = None


def install_tool_registry(registry: ToolRegistry | None) -> None:
    """Pin a profile-adjusted registry (``None`` clears for next reload)."""
    global _registry
    _registry = registry


__all__ = [
    "CostTier",
    "ToolCategory",
    "ToolPackage",
    "ToolRegistry",
    "ToolSpec",
    "ToolValidationRule",
    "ValidationRuleKind",
    "get_tool_registry",
    "install_tool_registry",
    "reset_tool_registry_for_tests",
]
