"""Trader and research CLI flags (no Postgres)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override tests/conftest autouse — no database required."""
    yield


def test_require_research_host_flag():
    from core.entry_cli import parse_trader_args

    args = parse_trader_args(["--require-research-host"])
    assert args.require_research_host is True
    assert args.force_in_process is False


def test_require_daemon_legacy_alias():
    from core.entry_cli import parse_trader_args

    args = parse_trader_args(["--require-daemon"])
    assert args.require_research_host is True


def test_verbose_short_flag():
    from core.entry_cli import parse_trader_args

    args = parse_trader_args(["-v"])
    assert args.verbose is True


def test_force_in_process_exclusive_with_require_host():
    from core.entry_cli import build_trader_parser

    parser = build_trader_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--require-research-host", "--force-in-process"])


def test_trader_help_lists_flags():
    proc = subprocess.run(
        [sys.executable, "__main__.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--require-research-host" in proc.stdout
    assert "--require-daemon" in proc.stdout
    assert "--force-in-process" in proc.stdout
    assert "--no-research" in proc.stdout
    assert "examples:" in proc.stdout


def test_research_help_lists_flags():
    proc = subprocess.run(
        [sys.executable, "-m", "research", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--verbose" in proc.stdout
    assert "--no-evolution" in proc.stdout
    assert "-v" in proc.stdout
