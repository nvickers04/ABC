"""Registry partition for ``tools.smoke_manifest`` — no broker, no Postgres."""

from tools.smoke_manifest import NEVER_AUTOTEST, broker_mutating_names, safe_names
from tools.tools_executor import _REGISTRY


def test_registry_partition_is_complete_and_disjoint():
    b = broker_mutating_names()
    s = safe_names()
    n = NEVER_AUTOTEST
    all_reg = set(_REGISTRY.keys())
    assert b & n == set()
    assert s & b == set()
    assert s & n == set()
    assert s | b | n == all_reg


def test_planning_tools_are_not_broker_mutating():
    assert "plan_order" in safe_names()
    assert "enter_option" in safe_names()
