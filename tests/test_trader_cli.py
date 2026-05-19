"""Trader CLI flag aliases (no Postgres)."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override tests/conftest autouse — no database required."""
    yield


def _parse_trader_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--require-daemon",
        "--require-research-host",
        action="store_true",
        dest="require_research_host",
    )
    group.add_argument("--force-in-process", action="store_true")
    return parser.parse_args(argv)


def test_require_research_host_flag():
    args = _parse_trader_args(["--require-research-host"])
    assert args.require_research_host is True


def test_require_daemon_legacy_alias():
    args = _parse_trader_args(["--require-daemon"])
    assert args.require_research_host is True


def test_main_help_lists_both_flags():
    proc = subprocess.run(
        [sys.executable, "__main__.py", "--help"],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--require-research-host" in proc.stdout
    assert "--require-daemon" in proc.stdout
