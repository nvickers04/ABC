"""
Execution templates — 10 template definitions + selection logic.

Each template defines an order type, use-case, and decision-boundary parameters
that are learned from historical performance and evolved continuously.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from research.config import COMPOSITE_TRADE_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

@dataclass
class TemplateDef:
    """Static definition of an execution template."""
    name: str
    order_type: str
    use_case: str
    default_boundaries: dict[str, tuple[float, float]]  # param -> (min, max)


TEMPLATE_DEFS: dict[str, TemplateDef] = {}


def _register(*args, **kwargs):
    t = TemplateDef(*args, **kwargs)
    TEMPLATE_DEFS[t.name] = t
    return t


_register(
    "stock_market", "market",
    "High conviction, immediate entry, low spread",
    {
        "composite_min": (0.55, 1.0), "composite_max": (0.55, 1.0),
        "iv_rank_min": (0.0, 100.0), "iv_rank_max": (0.0, 100.0),
        "atr_pct_min": (0.0, 10.0), "atr_pct_max": (0.0, 10.0),
    },
)
_register(
    "stock_bracket", "bracket",
    "High conviction, structured risk with stop/target OCAs",
    {
        "composite_min": (0.45, 1.0), "composite_max": (0.45, 1.0),
        "iv_rank_min": (0.0, 100.0), "iv_rank_max": (0.0, 100.0),
        "atr_pct_min": (0.5, 10.0), "atr_pct_max": (0.5, 10.0),
    },
)
_register(
    "stock_limit", "limit",
    "Moderate conviction, patient entry at limit",
    {
        "composite_min": (0.25, 0.60), "composite_max": (0.25, 0.60),
        "iv_rank_min": (0.0, 100.0), "iv_rank_max": (0.0, 100.0),
        "atr_pct_min": (0.5, 10.0), "atr_pct_max": (0.5, 10.0),
    },
)
_register(
    "stock_vwap", "vwap",
    "Large size, minimize market impact via VWAP",
    {
        "composite_min": (0.35, 1.0), "composite_max": (0.35, 1.0),
        "iv_rank_min": (0.0, 100.0), "iv_rank_max": (0.0, 100.0),
        "atr_pct_min": (0.3, 3.0), "atr_pct_max": (0.3, 3.0),
    },
)
_register(
    "stock_trailing", "trailing_stop_exit",
    "Trend following, let winners run with trailing stop",
    {
        "composite_min": (0.40, 1.0), "composite_max": (0.40, 1.0),
        "iv_rank_min": (0.0, 60.0), "iv_rank_max": (0.0, 60.0),
        "atr_pct_min": (0.5, 10.0), "atr_pct_max": (0.5, 10.0),
    },
)
_register(
    "vertical_spread", "vertical_spread",
    "Directional, high IV, defined risk",
    {
        "composite_min": (0.35, 1.0), "composite_max": (0.35, 1.0),
        "iv_rank_min": (40.0, 100.0), "iv_rank_max": (40.0, 100.0),
        "atr_pct_min": (0.5, 10.0), "atr_pct_max": (0.5, 10.0),
    },
)
_register(
    "calendar_diagonal", "calendar_spread",
    "Time decay + directional lean via calendar/diagonal",
    {
        "composite_min": (0.25, 0.55), "composite_max": (0.25, 0.55),
        "iv_rank_min": (30.0, 80.0), "iv_rank_max": (30.0, 80.0),
        "atr_pct_min": (0.3, 3.0), "atr_pct_max": (0.3, 3.0),
    },
)
_register(
    "premium_iron_condor", "iron_condor",
    "Neutral-to-low conviction, high IV, collect premium",
    {
        "composite_min": (0.0, 0.25), "composite_max": (0.0, 0.25),
        "iv_rank_min": (50.0, 100.0), "iv_rank_max": (50.0, 100.0),
        "atr_pct_min": (0.3, 8.0), "atr_pct_max": (0.3, 8.0),
    },
)
_register(
    "premium_butterfly", "butterfly",
    "Pinning thesis, high IV, defined-risk premium",
    {
        "composite_min": (0.0, 0.30), "composite_max": (0.0, 0.30),
        "iv_rank_min": (45.0, 100.0), "iv_rank_max": (45.0, 100.0),
        "atr_pct_min": (0.3, 8.0), "atr_pct_max": (0.3, 8.0),
    },
)
_register(
    "straddle_strangle", "straddle",
    "Vol expansion thesis, pre-event, buy volatility",
    {
        "composite_min": (0.20, 1.0), "composite_max": (0.20, 1.0),
        "iv_rank_min": (0.0, 40.0), "iv_rank_max": (0.0, 40.0),
        "atr_pct_min": (0.3, 10.0), "atr_pct_max": (0.3, 10.0),
    },
)


# ---------------------------------------------------------------------------
# Boundary loading from DB
# ---------------------------------------------------------------------------

def load_boundaries(db_conn) -> dict[str, dict[str, float]]:
    """Load current boundary parameters from template_boundaries table."""
    boundaries: dict[str, dict[str, float]] = {}
    cur = db_conn.execute(
        "SELECT template_name, param_name, param_value FROM template_boundaries"
    )
    for row in cur.fetchall():
        tname, pname, pval = row
        boundaries.setdefault(tname, {})[pname] = float(pval)
    return boundaries


def _normalize_boundary_params(params: dict[str, float]) -> dict[str, float]:
    """Normalize boundary params so *_min/*_max pairs stay ordered."""
    out = dict(params)
    pairs = [
        ("composite_min", "composite_max"),
        ("iv_rank_min", "iv_rank_max"),
        ("atr_pct_min", "atr_pct_max"),
    ]
    for min_k, max_k in pairs:
        if min_k in out and max_k in out:
            lo = float(out[min_k])
            hi = float(out[max_k])
            if lo > hi:
                lo, hi = hi, lo
            out[min_k] = lo
            out[max_k] = hi
    return out


def save_boundaries(
    db_conn,
    template_name: str,
    params: dict[str, float],
    generation: int = 0,
    fitness: float | None = None,
) -> None:
    """Persist boundary parameters for a template."""
    now = time.time()
    clean = {
        k: float(v)
        for k, v in params.items()
        if not str(k).startswith("_")
    }
    clean = _normalize_boundary_params(clean)
    rows = [
        (template_name, pname, pval, generation, fitness, now)
        for pname, pval in clean.items()
    ]
    db_conn.executemany(
        "INSERT OR REPLACE INTO template_boundaries "
        "(template_name, param_name, param_value, generation, fitness, updated_ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    db_conn.commit()


def init_default_boundaries(db_conn) -> None:
    """Seed boundary table with defaults if empty."""
    cur = db_conn.execute("SELECT COUNT(*) FROM template_boundaries")
    if cur.fetchone()[0] > 0:
        return  # Already seeded

    for tdef in TEMPLATE_DEFS.values():
        params = {}
        for pname, (lo, hi) in tdef.default_boundaries.items():
            # Use lo for *_min params, hi for *_max params so
            # the range [composite_min, composite_max] is wide enough to match.
            if pname.endswith("_min"):
                params[pname] = lo
            elif pname.endswith("_max"):
                params[pname] = hi
            else:
                params[pname] = (lo + hi) / 2.0
        save_boundaries(db_conn, tdef.name, params, generation=0, fitness=None)


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------

def select_template(
    db_conn,
    symbol: str,
    composite_score: float,
    iv_rank: float | None,
    atr_pct: float | None,
    vol_regime: str = "normal",
    quote: Any | None = None,
) -> dict | None:
    """
    Select the best execution template for a symbol.

    Returns trade dict or None if no template matches.
    """
    if abs(composite_score) < COMPOSITE_TRADE_THRESHOLD:
        return None

    direction = "long" if composite_score > 0 else "short"
    abs_composite = abs(composite_score)

    boundaries = load_boundaries(db_conn)
    candidates: list[tuple[str, float]] = []

    for tname, tdef in TEMPLATE_DEFS.items():
        b = _normalize_boundary_params(boundaries.get(tname, {}))
        comp_min = b.get("composite_min", 0.25)
        comp_max = b.get("composite_max", 1.0)

        if not (comp_min <= abs_composite <= comp_max):
            continue

        # IV rank check (skip if no IV data and template requires high IV)
        iv_min = b.get("iv_rank_min", 0.0)
        iv_max = b.get("iv_rank_max", 100.0)
        if iv_rank is not None and not (iv_min <= iv_rank <= iv_max):
            continue

        # ATR check
        atr_min = b.get("atr_pct_min", 0.0)
        atr_max = b.get("atr_pct_max", 5.0)
        if atr_pct is not None and not (atr_min <= atr_pct <= atr_max):
            continue

        # Template passes all boundary checks — get track record
        track = _get_track_record(db_conn, tname, vol_regime)
        score = track.get("win_rate", 0.5) * 100 + track.get("trades", 0) * 0.1
        candidates.append((tname, score))

    if not candidates:
        return None

    # Pick template with highest score (win_rate × trades)
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_name = candidates[0][0]
    best_def = TEMPLATE_DEFS[best_name]
    track = _get_track_record(db_conn, best_name, vol_regime)

    # Build entry/target/stop from quote + ATR
    entry_price = None
    target_price = None
    stop_price = None
    if quote:
        last = getattr(quote, "last", None) or getattr(quote, "mid", None)
        if last and last > 0 and atr_pct:
            entry_price = float(last)
            atr_dollars = entry_price * (atr_pct / 100.0)
            if direction == "long":
                target_price = round(entry_price + atr_dollars * 1.5, 2)
                stop_price = round(entry_price - atr_dollars, 2)
            else:
                target_price = round(entry_price - atr_dollars * 1.5, 2)
                stop_price = round(entry_price + atr_dollars, 2)

    return {
        "symbol": symbol,
        "direction": direction,
        "order_type": best_def.order_type,
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "max_hold_bars": 60,
        "setup_type": best_name,
        "legs_json": None,  # Populated by options leg builder if options template
        "composite_score": composite_score,
        "template_track_record": track,
        "vol_regime": vol_regime,
    }


def _get_track_record(
    db_conn,
    template_name: str,
    regime_key: str,
) -> dict:
    """Get historical performance for template in current regime."""
    cur = db_conn.execute(
        "SELECT trades, wins, avg_return_pct, sharpe "
        "FROM template_performance "
        "WHERE template_name = ? AND regime_key = ? "
        "ORDER BY updated_ts DESC LIMIT 1",
        (template_name, regime_key),
    )
    row = cur.fetchone()
    if row is None:
        # Fallback to aggregate record if regime-specific row not yet present.
        cur = db_conn.execute(
            "SELECT trades, wins, avg_return_pct, sharpe "
            "FROM template_performance "
            "WHERE template_name = ? AND regime_key = ? "
            "ORDER BY updated_ts DESC LIMIT 1",
            (template_name, "all"),
        )
        row = cur.fetchone()
    if row:
        trades, wins, avg_ret, sharpe = row
        return {
            "win_rate": wins / max(trades, 1),
            "trades": trades,
            "avg_return_pct": avg_ret,
            "sharpe": sharpe,
        }
    return {"win_rate": 0.5, "trades": 0, "avg_return_pct": 0.0, "sharpe": None}


def _track_record_snapshot_json(rec: dict) -> str | None:
    """Serialize template OOS snapshot for ``template_recommendations.track_record_json``."""
    track = rec.get("template_track_record")
    vol = str(rec.get("vol_regime") or "normal").strip() or "normal"
    if not isinstance(track, dict):
        return None
    trades = int(track.get("trades") or 0)
    if trades <= 0:
        return None
    win_rate = float(track.get("win_rate") or 0.0)
    sh = track.get("sharpe")
    payload = {
        "trades": trades,
        "win_pct": round(win_rate * 100.0, 1),
        "avg_return_pct": round(float(track.get("avg_return_pct") or 0.0), 4),
        "sharpe": round(float(sh), 2) if sh is not None else None,
        "regime_key": vol,
    }
    return json.dumps(payload)


def write_recommendations(
    db_conn,
    recommendations: list[dict],
) -> None:
    """Persist template recommendations to DB."""
    now = time.time()
    rows = []
    for rec in recommendations:
        tr_json = _track_record_snapshot_json(rec)
        rows.append((
            rec["symbol"],
            now,
            rec.get("setup_type", ""),
            rec.get("direction", ""),
            rec.get("composite_score", 0.0),
            rec.get("order_type", ""),
            rec.get("entry_price"),
            rec.get("target_price"),
            rec.get("stop_price"),
            json.dumps(rec.get("legs_json")) if rec.get("legs_json") else None,
            tr_json,
        ))
    if rows:
        db_conn.executemany(
            "INSERT OR REPLACE INTO template_recommendations "
            "(symbol, ts, template_name, direction, composite_score, order_type, "
            "entry_price, target_price, stop_price, legs_json, track_record_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        db_conn.commit()
