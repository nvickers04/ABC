"""Unit tests for scripts/health_common.py (no Postgres/IBKR)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    yield

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from health_common import (  # noqa: E402
    EXIT_FAIL,
    EXIT_OK,
    EXIT_WARN,
    Reporter,
    Status,
)


def test_reporter_exit_codes():
    rep = Reporter("Test", use_color=False)
    rep.ok("a", "fine")
    assert rep.exit_code() == EXIT_OK

    rep.warn("b", "hmm")
    assert rep.exit_code() == EXIT_WARN

    rep.fail("c", "bad")
    assert rep.exit_code() == EXIT_FAIL


def test_status_label_no_color():
    from health_common import status_label

    assert "OK" in status_label(Status.OK, use_color=False)
