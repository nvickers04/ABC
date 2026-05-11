"""Unit tests under ``tests/unit/`` — no shared Postgres ``memory`` fixture."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override root ``tests/conftest.py`` autouse (no ``memory.init_db()``)."""
    yield
