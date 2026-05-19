"""Integration: research-host outputs and simulated trader-cycle QualityMatrix posture."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.quality_test_support import reset_quality_runtime_state


@pytest.fixture(autouse=True)
def _isolated_db():
    """Override root conftest — these tests use in-memory fakes only."""
    yield


@pytest.fixture(autouse=True)
def _clean_quality():
    reset_quality_runtime_state()
    yield
    reset_quality_runtime_state()


@pytest.fixture
def research_config_store() -> dict[str, float]:
    return {
        "quality_matrix_enabled": 1.0,
        "estimated_ir": 0.0,
        "ir_gate_open": 0.0,
    }


class _ResearchIntegrationDB:
    """Minimal Postgres stub for QualityMatrix populate + sim composite lookup."""

    def __init__(
        self,
        *,
        feedback_rows: list[dict] | None = None,
        composite_rows: list[dict] | None = None,
    ) -> None:
        self.feedback_rows = list(feedback_rows or [])
        self.composite_rows = list(composite_rows or [])

    def execute(self, sql: str, params: tuple = ()):
        s = " ".join(sql.lower().split())
        if "from trade_feedback" in s and "group by symbol" in s:
            return _Rows(self.feedback_rows)
        if "from composite_scores" in s:
            rows = sorted(
                self.composite_rows,
                key=lambda r: float(r.get("composite_score", 0)),
                reverse=True,
            )
            if "order by composite_score desc" in s:
                return _CompositeRows(rows[:1])
            return _Rows(rows)
        if "from trade_feedback" in s and "order by ts desc" in s:
            return _Rows([])
        if "from tool_usage_log" in s or "from decision_provenance" in s:
            return _Rows([])
        return _Rows([])

    def commit(self) -> None:
        pass


class _Rows:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = [dict(r) for r in rows]

    def fetchall(self) -> list[dict]:
        return list(self._rows)

    def fetchone(self) -> dict | None:
        return dict(self._rows[0]) if self._rows else None


class _CompositeRows:
    """Rows compatible with simulation runner index access (row[0], row[1])."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = [
            (str(r["symbol"]), float(r.get("composite_score", 0))) for r in rows
        ]

    def fetchone(self) -> tuple[str, float] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[str, float]]:
        return list(self._rows)


@dataclass(frozen=True)
class _QualitySnapshot:
    overall_quality: str
    risk_multiplier: float
    global_execution_quality: float
    can_initiate_new_risk: bool
    blocked_categories: tuple[str, ...]
    top_composite: tuple[str | None, float]


def _seed_research_host_outputs(
    db: _ResearchIntegrationDB,
    store: dict[str, float],
    *,
    strong: bool,
) -> None:
    """Simulate research-host DB writes (scorer + combiner snapshot)."""
    now = time.time()
    if strong:
        db.composite_rows = [
            {
                "symbol": "NVDA",
                "composite_score": 0.82,
                "ts": now,
            }
        ]
        store["estimated_ir"] = 0.18
        store["ir_gate_open"] = 1.0
    else:
        db.composite_rows = []
        store["estimated_ir"] = 0.02
        store["ir_gate_open"] = 0.0


async def _quality_after_simulated_cycle(
    monkeypatch: pytest.MonkeyPatch,
    db: _ResearchIntegrationDB,
    store: dict[str, float],
    *,
    researcher_alive: bool,
    strong_research_signals: bool,
) -> _QualitySnapshot:
    """Mirror trader cycle quality steps: heartbeat sync → populate → cycle context refresh."""
    reset_quality_runtime_state()
    from core.runtime.cycle_context import build_cycle_user_context
    from core.runtime.operating_context import get_operating_context
    from core.quality.quality_matrix import get_quality_matrix_service
    from core.simulation.runner import _top_composite_for_date

    monkeypatch.setenv("ABC_SIMULATION", "1")
    monkeypatch.setattr("memory.get_db", lambda: db)
    monkeypatch.setattr(
        "memory.get_research_config",
        lambda key, default=0.0: float(store.get(key, default)),
    )
    monkeypatch.setattr(
        "memory.set_research_config",
        lambda key, val, **kwargs: store.__setitem__(key, float(val)),
    )
    monkeypatch.setattr(
        "core.runtime.heartbeat.is_research_host_operational",
        lambda *args, **kwargs: researcher_alive,
    )
    monkeypatch.setattr(
        "core.runtime.cycle_context.load_attention_block",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "core.runtime.cycle_context.load_intuition_block",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "core.runtime.working_memory_access.get_active_working_memory",
        lambda: _NoopWM(),
    )

    if strong_research_signals:
        _seed_research_host_outputs(db, store, strong=True)
    elif researcher_alive:
        _seed_research_host_outputs(db, store, strong=False)

    ctx = get_operating_context()
    if researcher_alive:
        ctx.set_researcher_available()
    else:
        ctx.set_researcher_unavailable()
    ctx.sync_researcher_from_heartbeat()
    if researcher_alive:
        ctx.quality.working_memory_completeness = 1.0
        ctx.quality.hypotheses_available = True
        ctx._recalculate_overall_quality()

    svc = get_quality_matrix_service()
    svc.populate(db)
    matrix = svc.get_matrix()

    await build_cycle_user_context(
        operating_context=ctx,
        state_text="═══ MARKET: REGULAR ═══\n",
        cost_line="",
        continuity_text="",
        pre_scan_prompt="",
        gap_guard_prompt="",
        et_now=datetime.now(timezone.utc),
    )
    matrix = svc.get_matrix()
    sym, score = _top_composite_for_date(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    return _QualitySnapshot(
        overall_quality=matrix.overall_quality,
        risk_multiplier=float(matrix.risk_multiplier),
        global_execution_quality=float(matrix.global_execution_quality),
        can_initiate_new_risk=bool(matrix.can_initiate_new_risk()),
        blocked_categories=tuple(matrix.blocked_tool_categories),
        top_composite=(sym, float(score)),
    )


class _NoopWM:
    def curate(self) -> None:
        return None

    def render(self, max_entries_per_section: int = 3) -> str:
        return ""


@pytest.mark.asyncio
async def test_research_host_signals_improve_quality_matrix_in_sim_cycle(
    monkeypatch: pytest.MonkeyPatch,
    research_config_store: dict[str, float],
) -> None:
    """Research-host heartbeat + signal pipeline raise QualityMatrix vs offline baseline."""
    from core.loop_config import get_loop_config

    lc = get_loop_config()
    good_feedback = [
        {
            "symbol": "NVDA",
            "n": 24,
            "avg_gap": 0.001,
            "winrate": 0.62,
        }
    ]
    poor_feedback = [
        {
            "symbol": "NVDA",
            "n": 8,
            "avg_gap": 22.0,
            "winrate": 0.38,
        }
    ]

    offline = await _quality_after_simulated_cycle(
        monkeypatch,
        _ResearchIntegrationDB(feedback_rows=poor_feedback),
        dict(research_config_store),
        researcher_alive=False,
        strong_research_signals=False,
    )
    online_weak = await _quality_after_simulated_cycle(
        monkeypatch,
        _ResearchIntegrationDB(feedback_rows=poor_feedback),
        dict(research_config_store),
        researcher_alive=True,
        strong_research_signals=False,
    )
    online_strong = await _quality_after_simulated_cycle(
        monkeypatch,
        _ResearchIntegrationDB(feedback_rows=good_feedback),
        dict(research_config_store),
        researcher_alive=True,
        strong_research_signals=True,
    )

    assert offline.overall_quality == "minimal"
    assert "research" in offline.blocked_categories
    assert offline.can_initiate_new_risk is False
    assert offline.risk_multiplier <= lc.minimal_posture_rm_cap + 1e-6

    assert online_weak.overall_quality == "limited"
    assert online_weak.risk_multiplier > offline.risk_multiplier
    assert online_weak.blocked_categories == ()

    assert online_strong.overall_quality == "full"
    assert online_strong.risk_multiplier > online_weak.risk_multiplier
    assert online_strong.global_execution_quality > online_weak.global_execution_quality
    assert online_strong.can_initiate_new_risk is True
    assert online_strong.top_composite[0] == "NVDA"
    assert online_strong.top_composite[1] >= 0.8


async def _quality_status_after_simulated_cycle(
    monkeypatch: pytest.MonkeyPatch,
    db: _ResearchIntegrationDB,
    store: dict[str, float],
    *,
    researcher_alive: bool,
    strong_research_signals: bool,
) -> dict[str, Any]:
    await _quality_after_simulated_cycle(
        monkeypatch,
        db,
        store,
        researcher_alive=researcher_alive,
        strong_research_signals=strong_research_signals,
    )
    from tools.tools_research import handle_quality_status

    return await handle_quality_status(MagicMock(), {})


@pytest.mark.asyncio
async def test_sim_cycle_quality_status_reflects_research_host(
    monkeypatch: pytest.MonkeyPatch,
    research_config_store: dict[str, float],
) -> None:
    """quality_status (first BacktestLLM action) reports higher matrix RM with research signals."""
    good_feedback = [{"symbol": "NVDA", "n": 20, "avg_gap": 0.001, "winrate": 0.6}]
    poor_feedback = [{"symbol": "NVDA", "n": 8, "avg_gap": 22.0, "winrate": 0.38}]

    offline_status = await _quality_status_after_simulated_cycle(
        monkeypatch,
        _ResearchIntegrationDB(feedback_rows=poor_feedback),
        dict(research_config_store),
        researcher_alive=False,
        strong_research_signals=False,
    )
    online_status = await _quality_status_after_simulated_cycle(
        monkeypatch,
        _ResearchIntegrationDB(feedback_rows=good_feedback),
        dict(research_config_store),
        researcher_alive=True,
        strong_research_signals=True,
    )

    assert offline_status["overall_quality"] == "minimal"
    assert float(online_status["risk_multiplier"]) > float(offline_status["risk_multiplier"])
    assert online_status["overall_quality"] == "full"
    assert online_status["matrix"]["matrix_can_new_risk"] is True
