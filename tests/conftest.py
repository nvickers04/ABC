"""Shared pytest fixtures for the ABC test suite (PR27).

This conftest hoists the ``_isolated_db`` autouse fixture that was
previously copy-pasted at the top of every test file. The fixture
swaps ``memory._DB_PATH`` to a tmp path, calls ``memory.reset_state``
to clear all module-level state (pending dicts, calibration version,
cached connection), then ``init_db()`` for a fresh schema.

Tests that need a different setup can override ``_isolated_db``
locally — pytest's nearest-fixture rule means a same-named fixture in
a test module wins over this one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Redirect memory module to a fresh temp DB for every test."""
    import memory
    db_path = tmp_path / "test.db"
    memory.reset_state(db_path=db_path)
    memory.init_db()
    yield
    memory.reset_state()


@pytest.fixture
def db():
    """Return the test DB connection."""
    from memory import get_db
    return get_db()
