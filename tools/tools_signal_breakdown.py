"""Signal-breakdown tool — explain *why* a symbol's composite is what it is.

The agent already sees ranked composites in INTUITION; when it wants
the per-signal attribution for a given name, it calls
``signal_breakdown(symbol)`` and gets back the exact rows the
combiner used to produce that composite, sorted by the magnitude of
their contribution.

Output schema::

    {
      "symbol": "UNH",
      "composite": 0.42,
      "composite_ts": 1714324200.0,
      "n_contributing": 14,
      "components": [
        {
          "signal": "momentum",
          "score": 0.81,        # raw signal score (signed, in [-1, 1])
          "weight": 0.07,       # current signal_weights row
          "ic": 0.045,          # global per-signal IC (rolling window)
          "category": "momentum",
          "contribution": 0.057, # signed weight * score
          "trust": 0.0026,       # |score| * |IC| * weight (matches intuition)
        },
        ...
      ],
      "missing_data": ["gap", "iv_rank"],   # registered signals with no fresh score
    }
"""

from __future__ import annotations

from typing import Any

from core.log_context import get_logger

logger = get_logger(__name__)


HANDLERS: dict[str, Any] = {}


def _validate_symbol(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    sym = raw.strip().upper()
    if not sym or len(sym) > 10:
        return None
    if not all(c.isalnum() or c in ".-" for c in sym):
        return None
    return sym


async def handle_signal_breakdown(executor, params: dict) -> Any:
    """Per-signal attribution for one symbol's most-recent composite."""
    sym = _validate_symbol(params.get("symbol"))
    if sym is None:
        return {"error": "symbol required (1-10 alphanumeric chars)"}

    try:
        from memory import get_db
        conn = get_db()
    except Exception as e:
        return {"error": f"db unavailable: {e}"}

    # Latest composite for the symbol.
    try:
        cur = conn.execute(
            "SELECT composite_score, ts FROM composite_scores "
            "WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            (sym,),
        )
        row = cur.fetchone()
    except Exception as e:
        return {"error": f"composite read failed: {e}"}
    if row is None:
        return {"error": f"no composite_score row for {sym} yet"}
    composite, composite_ts = float(row[0]), float(row[1])

    # Latest per-signal score for the symbol.
    try:
        cur = conn.execute(
            """
            SELECT signal_name, score, confidence
              FROM signal_scores
             WHERE symbol = ? AND ts = (
                SELECT MAX(ts) FROM signal_scores s2
                 WHERE s2.signal_name = signal_scores.signal_name
                   AND s2.symbol = ?
             )
            """,
            (sym, sym),
        )
        score_rows = cur.fetchall()
    except Exception as e:
        return {"error": f"signal_scores read failed: {e}"}
    score_by_sig: dict[str, float] = {}
    for sig, sc, _conf in score_rows:
        score_by_sig[str(sig)] = float(sc)

    # Weights.
    try:
        cur = conn.execute("SELECT signal_name, weight, category FROM signal_weights")
        weight_rows = cur.fetchall()
    except Exception as e:
        return {"error": f"signal_weights read failed: {e}"}
    weight_by_sig: dict[str, float] = {}
    cat_by_sig: dict[str, str] = {}
    for sig, w, cat in weight_rows:
        weight_by_sig[str(sig)] = float(w)
        if cat is not None:
            cat_by_sig[str(sig)] = str(cat)

    # IC (re-use combiner's global IC).
    ic_by_sig: dict[str, float] = {}
    try:
        from signals.combiner import SIGNAL_REGISTRY, _compute_signal_ic
        names = sorted(SIGNAL_REGISTRY.keys())
        if names:
            stats = _compute_signal_ic(conn, names)
            for sig, d in stats.items():
                ic_by_sig[str(sig)] = float(d.get("ic", 0.0))
    except Exception as e:
        logger.debug("signal_breakdown: ic compute failed: %s", e)

    # Build components table.
    components: list[dict[str, Any]] = []
    missing_data: list[str] = []
    registered: set[str] = set()
    try:
        from signals.combiner import SIGNAL_REGISTRY
        registered = set(SIGNAL_REGISTRY.keys())
    except Exception:
        pass

    for sig_name in sorted(set(score_by_sig) | registered):
        if sig_name not in score_by_sig:
            if sig_name in registered:
                missing_data.append(sig_name)
            continue
        score = score_by_sig[sig_name]
        weight = weight_by_sig.get(sig_name, 0.0)
        ic = ic_by_sig.get(sig_name, 0.0)
        contribution = weight * score
        trust = abs(score) * abs(ic) * weight
        components.append({
            "signal": sig_name,
            "score": float(score),
            "weight": float(weight),
            "ic": float(ic),
            "category": cat_by_sig.get(sig_name, "unknown"),
            "contribution": float(contribution),
            "trust": float(trust),
        })
    components.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "symbol": sym,
        "composite": composite,
        "composite_ts": composite_ts,
        "n_contributing": len(components),
        "components": components,
        "missing_data": sorted(missing_data),
    }


HANDLERS["signal_breakdown"] = handle_signal_breakdown


def register_handlers(registry) -> None:
    """Register this module's handlers on the central :class:`core.tool_registry.ToolRegistry`."""
    registry.bind_handlers(HANDLERS)
