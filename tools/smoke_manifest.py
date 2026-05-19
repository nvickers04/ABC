"""Partition of :data:`tools.tools_executor._REGISTRY` for trader tool smoke runs.

Safe default smoke (paper, read-only brokerage): executes **safe_names()** —
research data, charts, sizing plans, broker read-only snapshots, ``plan_order`` /
``enter_option`` (selection / planning only, no placements).

Skipped by default (**broker_mutating_names()**): stock/option placement, cancels,
stops mods, orphans cleanup (except emergencies never auto-tested).

Never auto-tested (**NEVER_AUTOTEST**): ``flatten_limits``, ``cancel_all_orphans``.
Run those only deliberately with operator approval.
"""

from __future__ import annotations

from core.tool_registry import get_tool_registry
from tools.tools_executor import _ORDER_ACTIONS


def _registry_keys() -> frozenset[str]:
    return frozenset(get_tool_registry().handlers.keys())

NEVER_AUTOTEST: frozenset[str] = frozenset({"flatten_limits", "cancel_all_orphans"})


def broker_mutating_names() -> frozenset[str]:
    """Tools that submit orders or cancel/modify broker working orders."""
    all_reg = _registry_keys()
    raw = (
        _ORDER_ACTIONS
        - {"plan_order", "enter_option", "flatten_limits"}
        | {"cancel_order", "cancel_stops"}
    )
    return frozenset(raw & all_reg)


def safe_names() -> frozenset[str]:
    """Tools safe to invoke in default paper smoke (no placements / cancels)."""
    return _registry_keys() - broker_mutating_names() - NEVER_AUTOTEST


def registry_partition_ok() -> bool:
    """True if SAFE | MUTATING | NEVER == registry (_disjoint)."""
    b = broker_mutating_names()
    s = safe_names()
    n = NEVER_AUTOTEST
    if b & n:
        return False
    if s & b or s & n:
        return False
    return s | b | n == _registry_keys()
