#!/usr/bin/env python3
"""One-off: switch logging.getLogger -> core.log_context.get_logger in target trees."""

from __future__ import annotations

from pathlib import Path

IMPORT_LINE = "from core.log_context import get_logger\n"

TARGETS = [
    Path("core/runtime"),
    Path("tools"),
    Path("execution"),
    Path("research/host.py"),
]


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "logging.getLogger" not in text:
        return False
    new = text.replace("logger = logging.getLogger(__name__)", "logger = get_logger(__name__)")
    new = new.replace(
        'logger = logging.getLogger("research.host")',
        'logger = get_logger("research.host")',
    )
    if new == text and "from core.log_context import get_logger" in text:
        return False
    if "from core.log_context import get_logger" not in new:
        if "from __future__ import annotations" in new:
            new = new.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\n" + IMPORT_LINE,
                1,
            )
        else:
            new = IMPORT_LINE + new
    path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    changed: list[str] = []
    for target in TARGETS:
        paths = [target] if target.suffix == ".py" else sorted(target.glob("*.py"))
        for path in paths:
            if patch(path):
                changed.append(str(path))
    print(f"patched {len(changed)} files")
    for name in changed:
        print(f"  {name}")


if __name__ == "__main__":
    main()
