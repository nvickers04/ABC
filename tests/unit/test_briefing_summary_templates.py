"""briefing_summary template OOS enrichment (track_record + template_leaderboard)."""

from __future__ import annotations

from unittest.mock import patch

from signals.briefing import briefing_summary


def _minimal_data(
    *,
    template_perf: list[dict],
    recommendations: list[dict],
    env_vol: str | None = "high",
    n_eff: float | None = None,
) -> dict:
    d = {
        "env": (
            {
                "volatility_regime": env_vol,
                "trend_regime": "up",
            }
            if env_vol
            else None
        ),
        "weights": [],
        "composites": [],
        "recommendations": recommendations,
        "feedback": [],
        "template_perf": template_perf,
        "estimated_ir": 0.1,
        "ir_gate_open": True,
        "ir_gate_min": 0.05,
    }
    if n_eff is not None:
        d["n_eff"] = n_eff
    return d


def test_track_record_attached_when_trades_above_min():
    data = _minimal_data(
        template_perf=[
            {
                "template_name": "momentum_long",
                "regime_key": "all",
                "trades": 20,
                "wins": 12,
                "avg_return_pct": 0.15,
                "sharpe": 1.2,
            }
        ],
        recommendations=[
            {
                "symbol": "NVDA",
                "template_name": "momentum_long",
                "direction": "long",
                "composite_score": 0.5,
                "order_type": "limit",
                "entry_price": 100.0,
                "target_price": 105.0,
                "stop_price": 98.0,
                "legs_json": None,
            }
        ],
    )
    with patch("core.config.BRIEFING_MIN_TEMPLATE_TRADES", 5), patch(
        "core.config.BRIEFING_TEMPLATE_LEADERBOARD_K", 8
    ):
        out = briefing_summary(data)
    ar = out["ACTION_REQUIRED"]
    assert len(ar) == 1
    assert "track_record" in ar[0]
    assert ar[0]["track_record"]["trades"] == 20
    assert ar[0]["track_record"]["win_pct"] == 60.0
    assert ar[0]["track_record"]["sharpe"] == 1.2
    lb = out["template_leaderboard"]
    assert len(lb) == 1
    assert lb[0]["template"] == "momentum_long"


def test_no_track_record_when_below_min_trades():
    data = _minimal_data(
        template_perf=[
            {
                "template_name": "momentum_long",
                "regime_key": "all",
                "trades": 3,
                "wins": 2,
                "avg_return_pct": 0.1,
                "sharpe": 0.5,
            }
        ],
        recommendations=[
            {
                "symbol": "NVDA",
                "template_name": "momentum_long",
                "direction": "long",
                "composite_score": 0.5,
                "order_type": None,
                "entry_price": None,
                "target_price": None,
                "stop_price": None,
                "legs_json": None,
            }
        ],
    )
    with patch("core.config.BRIEFING_MIN_TEMPLATE_TRADES", 5), patch(
        "core.config.BRIEFING_TEMPLATE_LEADERBOARD_K", 8
    ):
        out = briefing_summary(data)
    assert "track_record" not in out["ACTION_REQUIRED"][0]
    assert out["template_leaderboard"] == []


def test_prefers_regime_specific_row():
    data = _minimal_data(
        env_vol="high",
        template_perf=[
            {
                "template_name": "t1",
                "regime_key": "high",
                "trades": 10,
                "wins": 6,
                "avg_return_pct": 0.2,
                "sharpe": 2.0,
            },
            {
                "template_name": "t1",
                "regime_key": "all",
                "trades": 100,
                "wins": 50,
                "avg_return_pct": 0.05,
                "sharpe": 0.1,
            },
        ],
        recommendations=[
            {
                "symbol": "X",
                "template_name": "t1",
                "direction": "long",
                "composite_score": 0.4,
                "order_type": None,
                "entry_price": None,
                "target_price": None,
                "stop_price": None,
                "legs_json": None,
            }
        ],
    )
    with patch("core.config.BRIEFING_MIN_TEMPLATE_TRADES", 5), patch(
        "core.config.BRIEFING_TEMPLATE_LEADERBOARD_K", 8
    ):
        out = briefing_summary(data)
    tr = out["ACTION_REQUIRED"][0]["track_record"]
    assert tr["trades"] == 10
    assert tr["sharpe"] == 2.0


def test_leaderboard_sorted_by_sharpe():
    data = _minimal_data(
        template_perf=[
            {
                "template_name": "weak",
                "regime_key": "all",
                "trades": 10,
                "wins": 5,
                "avg_return_pct": 0.01,
                "sharpe": 0.2,
            },
            {
                "template_name": "strong",
                "regime_key": "all",
                "trades": 10,
                "wins": 7,
                "avg_return_pct": 0.3,
                "sharpe": 1.5,
            },
        ],
        recommendations=[],
    )
    with patch("core.config.BRIEFING_MIN_TEMPLATE_TRADES", 5), patch(
        "core.config.BRIEFING_TEMPLATE_LEADERBOARD_K", 8
    ):
        out = briefing_summary(data)
    lb = out["template_leaderboard"]
    assert [x["template"] for x in lb] == ["strong", "weak"]


def test_persisted_track_record_wins_over_template_perf():
    """Point-in-time snapshot on the rec row beats rolling template_perf."""
    data = _minimal_data(
        template_perf=[
            {
                "template_name": "momentum_long",
                "regime_key": "all",
                "trades": 100,
                "wins": 50,
                "avg_return_pct": 0.01,
                "sharpe": 0.2,
            }
        ],
        recommendations=[
            {
                "symbol": "NVDA",
                "template_name": "momentum_long",
                "direction": "long",
                "composite_score": 0.5,
                "order_type": "limit",
                "entry_price": 100.0,
                "target_price": 105.0,
                "stop_price": 98.0,
                "legs_json": None,
                "track_record": {
                    "trades": 20,
                    "win_pct": 75.0,
                    "avg_return_pct": 0.3,
                    "sharpe": 2.5,
                    "regime_key": "high",
                },
            }
        ],
    )
    with patch("core.config.BRIEFING_MIN_TEMPLATE_TRADES", 5), patch(
        "core.config.BRIEFING_TEMPLATE_LEADERBOARD_K", 8
    ):
        out = briefing_summary(data)
    tr = out["ACTION_REQUIRED"][0]["track_record"]
    assert tr["win_pct"] == 75.0
    assert tr["trades"] == 20
    assert tr["sharpe"] == 2.5


def test_edge_includes_n_eff_and_numeric_first_note():
    data = _minimal_data(
        template_perf=[],
        recommendations=[],
        n_eff=14.25,
    )
    with patch("core.config.BRIEFING_MIN_TEMPLATE_TRADES", 5), patch(
        "core.config.BRIEFING_TEMPLATE_LEADERBOARD_K", 8
    ):
        out = briefing_summary(data)
    edge = out["edge"]
    assert edge["n_eff"] == 14.25
    assert "estimated_ir" in edge
    assert "numbers" in edge["note"].lower()
    assert "veto" in edge["note"].lower()
