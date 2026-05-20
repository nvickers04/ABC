"""Tests for infra/runtime/env_profile.py (deploy env merge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from infra.runtime.env_profile import (
    VALID_ENVS,
    merge_env_files,
    parse_env_file,
    resolve_profile_path,
    write_env_file,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    """No Postgres required for env profile parsing."""
    yield


def test_valid_envs():
    assert "paper" in VALID_ENVS
    assert "live" in VALID_ENVS
    assert "dev" in VALID_ENVS


def test_parse_env_file_ignores_comments(tmp_path: Path):
    p = tmp_path / "t.env"
    p.write_text(
        "# comment\nTRADING_MODE=paper\n\nIBKR_PORT=7497\n",
        encoding="utf-8",
    )
    assert parse_env_file(p) == {"TRADING_MODE": "paper", "IBKR_PORT": "7497"}


def test_merge_env_files_later_wins(tmp_path: Path):
    a = tmp_path / "a.env"
    b = tmp_path / "b.env"
    a.write_text("PROFIT_PROFILE=balanced\nIBKR_PORT=7497\n", encoding="utf-8")
    b.write_text("PROFIT_PROFILE=aggressive\n", encoding="utf-8")
    merged = merge_env_files(a, b)
    assert merged["PROFIT_PROFILE"] == "aggressive"
    assert merged["IBKR_PORT"] == "7497"


def test_resolve_profile_path_example():
    path = resolve_profile_path("paper")
    assert path.name.endswith(".env.example") or path.name == "paper.env"
    assert path.parent.name == "env"


def test_write_env_file_roundtrip(tmp_path: Path):
    out = tmp_path / "out.env"
    write_env_file(out, {"A": "1", "B": "has space"})
    assert "A=1" in out.read_text(encoding="utf-8")
    assert 'B="has space"' in out.read_text(encoding="utf-8")


def test_resolve_unknown_env():
    with pytest.raises(ValueError, match="Unknown env"):
        resolve_profile_path("staging")
