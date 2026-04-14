"""
Signal-based agent briefing — composite scores, template recommendations,
signal quality metrics, and template performance.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from signals.base import SIGNAL_REGISTRY

logger = logging.getLogger(__name__)


def query_briefing_data() -> dict:
    """Query all briefing data from signal combination DB tables."""
    try:
        from memory import get_db
        db = get_db()
    except Exception:
        return {}

    result: dict = {}

    # ── Environment ─────────────────────────────────────────
    try:
        env = db.execute(
            """SELECT volatility_regime, trend_regime, breadth_regime,
                      momentum_regime, volume_regime, avg_atr_pct,
                      dispersion, strategy_fit_json
               FROM environment_snapshots
               ORDER BY id DESC LIMIT 1"""
        ).fetchone()
        if env:
            result["env"] = dict(env)
    except Exception:
        pass

    # ── Signal weights + N_eff ──────────────────────────────
    try:
        rows = db.execute(
            "SELECT signal_name, weight, n_eff, category FROM signal_weights ORDER BY ABS(weight) DESC"
        ).fetchall()
        if rows:
            result["weights"] = [dict(r) for r in rows]
            result["n_eff"] = float(rows[0]["n_eff"]) if rows[0]["n_eff"] else None
    except Exception:
        pass

    # ── Composite scores (latest round) ─────────────────────
    try:
        latest_ts = db.execute(
            "SELECT MAX(ts) FROM composite_scores"
        ).fetchone()[0]
        if latest_ts:
            composites = db.execute(
                """SELECT symbol, composite_score, signal_breakdown_json
                   FROM composite_scores
                   WHERE ts = ?
                   ORDER BY ABS(composite_score) DESC""",
                (latest_ts,),
            ).fetchall()
            if composites:
                result["composites"] = [dict(r) for r in composites]
    except Exception:
        pass

    # ── Template recommendations (latest round) ─────────────
    try:
        latest_ts = db.execute(
            "SELECT MAX(ts) FROM template_recommendations"
        ).fetchone()[0]
        if latest_ts:
            recs = db.execute(
                """SELECT symbol, template_name, direction, composite_score,
                          order_type, entry_price, target_price, stop_price, legs_json
                   FROM template_recommendations
                   WHERE ts = ?
                   ORDER BY ABS(composite_score) DESC""",
                (latest_ts,),
            ).fetchall()
            if recs:
                result["recommendations"] = [dict(r) for r in recs]
    except Exception:
        pass

    # ── Template performance (rolling) ──────────────────────
    try:
        perf = db.execute(
            """SELECT template_name, regime_key, trades, wins,
                      avg_return_pct, sharpe
               FROM template_performance
               WHERE trades > 0
               ORDER BY trades DESC"""
        ).fetchall()
        if perf:
            result["template_perf"] = [dict(r) for r in perf]
    except Exception:
        pass

    # ── Trade feedback (unchanged from old system) ──────────
    try:
        feedback_rows = db.execute(
            """SELECT simulated_return, actual_pnl, execution_gap
               FROM trade_feedback
               WHERE ts > datetime('now', '-7 days')
               ORDER BY ts DESC"""
        ).fetchall()
        if feedback_rows:
            result["feedback"] = [dict(r) for r in feedback_rows]
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Layer 1: Signal quality
# ---------------------------------------------------------------------------

def briefing_summary(data: dict) -> dict:
    """Layers 1-3: signal quality, top composites, and recommended trades."""
    env = data.get("env")
    weights = data.get("weights", [])
    composites = data.get("composites", [])
    recs = data.get("recommendations", [])
    feedback = data.get("feedback", [])

    # Regime
    regime = "unknown"
    if env:
        regime = f"{env.get('volatility_regime', '?')}/{env.get('trend_regime', '?')}"

    # Signal quality summary
    n_eff = data.get("n_eff")
    total_signals = len(SIGNAL_REGISTRY)
    weight_leaders = []
    for w in weights[:5]:
        weight_leaders.append(f"{w['signal_name']} ({w['weight']:.3f})")

    # Top composites with category breakdown
    top_composites = []
    for comp in composites[:8]:
        entry = {
            "symbol": comp["symbol"],
            "composite": round(comp["composite_score"], 3),
        }
        try:
            breakdown = json.loads(comp["signal_breakdown_json"]) if comp.get("signal_breakdown_json") else {}
            entry["breakdown"] = {k: round(v, 2) for k, v in breakdown.items()}
        except Exception:
            pass
        top_composites.append(entry)

    # Recommended trades
    actionable = []
    for rec in recs:
        entry_price = rec.get("entry_price")
        target_price = rec.get("target_price")
        stop_price = rec.get("stop_price")
        rr = None
        if entry_price and target_price and stop_price:
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            rr = round(reward / risk, 1) if risk > 0 else None

        legs = None
        if rec.get("legs_json"):
            try:
                ld = json.loads(rec["legs_json"]) if isinstance(rec["legs_json"], str) else rec["legs_json"]
                legs = ld.get("strategy")
            except Exception:
                pass

        item = {
            "symbol": rec["symbol"],
            "direction": rec["direction"],
            "template": rec["template_name"],
            "order_type": rec.get("order_type"),
            "entry": round(entry_price, 2) if entry_price else None,
            "target": round(target_price, 2) if target_price else None,
            "stop": round(stop_price, 2) if stop_price else None,
            "rr": rr,
            "composite": round(rec["composite_score"], 3),
        }
        if legs:
            item["legs"] = legs
        actionable.append(item)

    fb_line = None
    if feedback:
        avg_pnl = sum(r.get("actual_pnl") or 0 for r in feedback) / len(feedback)
        fb_line = f"{len(feedback)} trades, avg P&L ${avg_pnl:.2f}"

    result = {
        "regime": regime,
        "signal_quality": {
            "n_eff": round(n_eff, 1) if n_eff else None,
            "total_signals": total_signals,
            "independence_ratio": round(n_eff / max(total_signals, 1) * 100, 1) if n_eff else None,
            "weight_leaders": weight_leaders,
        },
        "top_composites": top_composites,
        "ACTION_REQUIRED": actionable,
        "instruction": (
            "ADAPT these signals to CURRENT prices — do NOT skip because entry_price is stale. "
            "The composite gives you direction + conviction. The template gives you structure. "
            "Build the trade at today's price using the suggested order type. "
            "Call option_chain(symbol, side='put') or side='call', then EXECUTE."
        ),
        "feedback": fb_line,
    }
    return result


# ---------------------------------------------------------------------------
# Layer 2: Recommended trades (signals detail view)
# ---------------------------------------------------------------------------

def briefing_signals(data: dict) -> dict:
    """Full recommended trade list with composite + template details."""
    recs = data.get("recommendations", [])
    composites = data.get("composites", [])

    # Build composite lookup
    comp_lookup = {}
    for c in composites:
        comp_lookup[c["symbol"]] = c

    out = []
    for rec in recs:
        entry_price = rec.get("entry_price")
        target_price = rec.get("target_price")
        stop_price = rec.get("stop_price")
        rr = None
        if entry_price and target_price and stop_price:
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            rr = round(reward / risk, 1) if risk > 0 else None

        legs = None
        if rec.get("legs_json"):
            try:
                ld = json.loads(rec["legs_json"]) if isinstance(rec["legs_json"], str) else rec["legs_json"]
                legs = ld.get("strategy")
            except Exception:
                pass

        item = {
            "symbol": rec["symbol"],
            "dir": rec["direction"],
            "template": rec["template_name"],
            "type": rec.get("order_type"),
            "entry": entry_price,
            "target": target_price,
            "stop": stop_price,
            "rr": rr,
            "composite": round(rec["composite_score"], 3),
        }
        if legs:
            item["legs"] = legs

        # Add category breakdown if available
        comp = comp_lookup.get(rec["symbol"])
        if comp and comp.get("signal_breakdown_json"):
            try:
                item["breakdown"] = json.loads(comp["signal_breakdown_json"])
            except Exception:
                pass

        out.append(item)
    return {"signals": out, "count": len(recs)}


# ---------------------------------------------------------------------------
# Layer 3: Template performance
# ---------------------------------------------------------------------------

def briefing_strategies(data: dict) -> dict:
    """Template performance table (replaces old strategy slots)."""
    perf = data.get("template_perf", [])
    out = []
    for row in perf:
        trades = row.get("trades", 0)
        wins = row.get("wins", 0)
        win_pct = round(wins / max(trades, 1) * 100, 1)
        entry = {
            "template": row.get("template_name"),
            "regime": row.get("regime_key"),
            "win_pct": win_pct,
            "trades": trades,
            "avg_ret": round(row.get("avg_return_pct", 0), 3),
            "sharpe": round(row["sharpe"], 2) if row.get("sharpe") else None,
        }
        out.append(entry)
    return {"templates": out, "count": len(perf)}


# ---------------------------------------------------------------------------
# Layer 4: Environment (updated with N_eff trend + signal health)
# ---------------------------------------------------------------------------

def briefing_environment(data: dict) -> dict:
    """Full environment regimes + signal health + N_eff trend."""
    env = data.get("env")
    if not env:
        return {"environment": "No environment data"}

    result: dict[str, Any] = {
        "volatility": env.get("volatility_regime"),
        "trend": env.get("trend_regime"),
        "breadth": env.get("breadth_regime"),
        "momentum": env.get("momentum_regime"),
        "volume": env.get("volume_regime"),
        "atr_pct": env.get("avg_atr_pct"),
        "dispersion": env.get("dispersion"),
    }

    # N_eff current + trend
    n_eff = data.get("n_eff")
    if n_eff:
        result["n_eff"] = round(n_eff, 1)
        result["n_eff_status"] = (
            "healthy" if n_eff >= 15
            else "warning" if n_eff >= 10
            else "alert"
        )

    # Signal health: check which signals have recent scores
    try:
        from memory import get_db
        import time
        db = get_db()
        stale_threshold = time.time() - 3600  # 1 hour
        stale = db.execute(
            """SELECT signal_name, MAX(ts) as last_ts
               FROM signal_scores
               GROUP BY signal_name
               HAVING MAX(ts) < ?""",
            (stale_threshold,),
        ).fetchall()
        if stale:
            result["stale_signals"] = [
                {"signal": row["signal_name"], "hours_ago": round((time.time() - row["last_ts"]) / 3600, 1)}
                for row in stale
            ]
        total_active = db.execute(
            "SELECT COUNT(DISTINCT signal_name) FROM signal_scores WHERE ts > ?",
            (stale_threshold,),
        ).fetchone()[0]
        result["active_signals"] = total_active
    except Exception:
        pass

    # Environment history from template performance + environment snapshots
    try:
        from memory import get_db
        db = get_db()
        hist_rows = db.execute(
            """SELECT
                   tp.template_name,
                   CASE
                       WHEN e.avg_atr_pct < 1.5 THEN 'low'
                       WHEN e.avg_atr_pct < 2.5 THEN 'normal'
                       WHEN e.avg_atr_pct < 3.5 THEN 'high'
                       ELSE 'extreme'
                   END as vol_band,
                   AVG(tp.avg_return) as avg_ret,
                   tp.win_pct,
                   tp.total_trades as n
               FROM template_performance tp
               JOIN environment_snapshots e
                 ON ABS(julianday(tp.ts) - julianday(e.session_date)) < 1
               WHERE tp.total_trades >= 2 AND e.avg_atr_pct IS NOT NULL
               GROUP BY tp.template_name, vol_band
               ORDER BY avg_ret DESC
               LIMIT 15"""
        ).fetchall()
        if hist_rows:
            lines = []
            for r in hist_rows:
                lines.append(
                    f"  {r['template_name']:<25s} vol={r['vol_band']:<7s} "
                    f"win={r['win_pct'] or 0:.0f}% avg_ret={r['avg_ret'] or 0:.4f} (n={r['n']})"
                )
            result["history"] = "\n".join(lines)[:600]
    except Exception:
        pass

    return result
