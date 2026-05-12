"""TRADER_IN_PROCESS_SCORER env is parsed at core.config import time."""

import importlib
import os

import pytest


@pytest.fixture()
def _reload_config(monkeypatch):
    import core.config as cc

    yield cc
    monkeypatch.delenv("TRADER_IN_PROCESS_SCORER", raising=False)
    importlib.reload(cc)


def test_trader_ips_never_aliases(_reload_config, monkeypatch):
    monkeypatch.setenv("TRADER_IN_PROCESS_SCORER", "remote_only")
    importlib.reload(_reload_config)
    assert _reload_config.TRADER_IN_PROCESS_SCORER_NEVER is True


def test_trader_ips_auto_default(_reload_config, monkeypatch):
    # load_dotenv() on reload repopulates from .env if the var is unset; pin auto so
    # operator .env (e.g. never) does not override this assertion.
    monkeypatch.setenv("TRADER_IN_PROCESS_SCORER", "auto")
    importlib.reload(_reload_config)
    assert _reload_config.TRADER_IN_PROCESS_SCORER_NEVER is False
