"""Prompt budget helpers (no Postgres)."""

from __future__ import annotations

import pytest

from core.quality.quality_matrix import QualityMatrix
from core.runtime.prompt_budget import (
    build_continuity_block,
    estimate_tokens,
    truncate_text,
)


@pytest.fixture(autouse=True)
def _isolated_db():
    yield


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_quality_matrix_compact_shorter_but_keeps_posture():
    m = QualityMatrix(
        overall_quality="minimal",
        risk_multiplier=0.4,
        force_conservative_reasoning=True,
        blocked_tool_categories=["research"],
    )
    full = m.to_prompt_block(0.4, compact=False)
    compact = m.to_prompt_block(0.4, compact=True)
    assert len(compact) < len(full)
    assert "MINIMAL" in compact
    assert "risk ×0.40" in compact
    assert "Blocked: research" in compact
    assert "CONSERVATIVE" in compact


def test_continuity_block_compacts_snapshots():
    text = build_continuity_block(
        last_cycle_summary="x" * 300,
        market_snapshots=[f"snap{i}" for i in range(10)],
        max_snapshots=2,
        max_snapshot_chars=20,
        max_summary_chars=50,
    )
    assert "LAST:" in text
    assert "SNAPS:" in text
    assert "snap8" in text or "snap9" in text
    assert len(text) < 400


def test_truncate_text_marker():
    out = truncate_text("hello world", 8)
    assert "…" in out or len(out) <= 8
