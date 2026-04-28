"""Logging convention enforcement.

Locks the project-wide convention that every module declaring a
module-level ``logger`` uses ``logging.getLogger(__name__)`` (so log
records carry the dotted package path, which the rotating file handler
in ``__main__.setup_logging`` relies on).

Also enforces that no business-logic module contains stray top-level
``print(...)`` calls. Acceptable exceptions: ``__main__.py`` (CLI),
``core.agent`` script-mode block, ``execution.ibkr_core`` connection
smoke test, and modules whose prints live exclusively inside
``if __name__ == "__main__":`` blocks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Top-level packages we want to enforce the convention on.
PACKAGE_DIRS = ["core", "data", "execution", "memory", "research", "signals", "tools"]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for pkg in PACKAGE_DIRS:
        pkg_dir = ROOT / pkg
        if not pkg_dir.exists():
            continue
        files.extend(pkg_dir.rglob("*.py"))
    return files


# ── Convention 1: module-level `logger = logging.getLogger(__name__)` ──


def _module_logger_assignments(tree: ast.Module) -> list[ast.Assign]:
    """Return module-level `logger = ...` assignments."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "logger":
                    out.append(node)
    return out


def _is_get_logger_dunder_name(call: ast.Call) -> bool:
    """True if call is ``logging.getLogger(__name__)``."""
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "getLogger":
        return False
    if not (
        isinstance(call.func.value, ast.Name) and call.func.value.id == "logging"
    ):
        return False
    if len(call.args) != 1:
        return False
    arg = call.args[0]
    return isinstance(arg, ast.Name) and arg.id == "__name__"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_module_logger_uses_dunder_name(path: Path) -> None:
    """Every top-level `logger = ...` must call ``logging.getLogger(__name__)``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assigns = _module_logger_assignments(tree)
    for assign in assigns:
        value = assign.value
        if not (isinstance(value, ast.Call) and _is_get_logger_dunder_name(value)):
            pytest.fail(
                f"{path.relative_to(ROOT)}:{assign.lineno} — module-level "
                f"`logger` must be `logging.getLogger(__name__)`"
            )


# ── Convention 2: no stray top-level print() in business logic ──────


def _toplevel_print_calls(tree: ast.Module) -> list[ast.Call]:
    """Return print(...) calls that live outside ``if __name__ == '__main__':``."""

    def _is_main_guard(node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        test = node.test
        # Match: __name__ == "__main__"
        if not isinstance(test, ast.Compare):
            return False
        if not (isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            return False
        if not test.ops or not isinstance(test.ops[0], ast.Eq):
            return False
        if not test.comparators:
            return False
        cmp0 = test.comparators[0]
        return isinstance(cmp0, ast.Constant) and cmp0.value == "__main__"

    out: list[ast.Call] = []

    def _walk(nodes):
        for node in nodes:
            if _is_main_guard(node):
                continue  # entire __main__ block is allowed
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "print"
                ):
                    out.append(child)

    _walk(tree.body)
    return out


# Modules with documented print usage that lives outside a __main__ guard
# (e.g. inside a top-level demo function). Add sparingly.
PRINT_ALLOWLIST: set[str] = {
    # Top-level `test_connection()` smoke test — invoked from this file's
    # own `if __name__ == "__main__":` block. Refactoring to logger calls
    # would defeat the point (it's an interactive CLI check).
    "execution/ibkr_core.py",
}


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_stray_top_level_prints(path: Path) -> None:
    rel = path.relative_to(ROOT).as_posix()
    if rel in PRINT_ALLOWLIST:
        pytest.skip(f"{rel} explicitly allowlisted")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = _toplevel_print_calls(tree)
    if calls:
        lines = ", ".join(str(c.lineno) for c in calls)
        pytest.fail(
            f"{rel} contains {len(calls)} stray print() call(s) at line(s) "
            f"{lines} — prefer `logger.<level>(...)` for business logic. "
            f"If intentional CLI output, move into an `if __name__ == "
            f"\"__main__\":` block or add to PRINT_ALLOWLIST."
        )
