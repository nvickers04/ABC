"""
Naive intuition layer — ranks symbols by an "attention score" so the
agent sees the most-interesting names *now*, instead of round-robining
the whole universe.

This is the **global-IC version** described in
docs/PLAN_COGNITIVE_ARCHITECTURE.md §5.  A future PR may swap the
global IC for the per-symbol IC produced by ``signals.per_symbol_ic``.

Score formula (per symbol)::

    attention_score = abs(composite)
                    * sum_signals( abs(signal_score) * abs(IC) * weight )
                    + abs(novelty)

where:
  * ``composite``      = latest composite score for the symbol (in [-1, 1])
  * ``signal_score``   = latest per-signal score for that symbol
  * ``IC``             = global per-signal IC over the rolling window
  * ``weight``         = current signal_weights row
  * ``novelty``        = composite_now − composite_prev (last persisted round)

The renderer produces the INTUITION block, intended to sit just below
the ATTENTION block in the cycle prompt.

All public helpers are fail-soft: if anything goes wrong (cold start,
table missing, etc.) we return an empty dict / empty string and log
at DEBUG.  The agent prompt must never crash because of intuition.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────

TOP_N: int = 5            # rows shown in the INTUITION block
TOP_DRIVERS: int = 3      # signals listed per row (top by |contribution|)
NOVELTY_LOOKBACK_S: float = 12 * 3600.0  # only count "prev" within 12h


# ── Latest-data helpers ─────────────────────────────────────────


def _latest_signal_scores(conn) -> dict[str, dict[str, float]]:
    """Return ``{symbol: {signal_name: score}}`` using each (signal,symbol)'s
    most-recent ``signal_scores`` row."""
    out: dict[str, dict[str, float]] = {}
    try:
        cur = conn.execute(
            """
            SELECT signal_name, symbol, score
              FROM signal_scores
             WHERE ts = (
                SELECT MAX(ts) FROM signal_scores s2
                 WHERE s2.signal_name = signal_scores.signal_name
                   AND s2.symbol = signal_scores.symbol
             )
            """
        )
    except Exception as e:
        logger.debug("intuition: signal_scores read failed: %s", e)
        return out
    for sig, sym, sc in cur.fetchall():
        out.setdefault(str(sym), {})[str(sig)] = float(sc)
    return out


def _latest_weights(conn) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        cur = conn.execute("SELECT signal_name, weight FROM signal_weights")
    except Exception as e:
        logger.debug("intuition: signal_weights read failed: %s", e)
        return out
    for name, w in cur.fetchall():
        out[str(name)] = float(w)
    return out


def _latest_composites(conn) -> dict[str, tuple[float, float]]:
    """Return ``{symbol: (composite, ts)}`` for each symbol's most-recent row."""
    out: dict[str, tuple[float, float]] = {}
    try:
        cur = conn.execute(
            """
            SELECT symbol, composite_score, ts
              FROM composite_scores
             WHERE ts = (
                SELECT MAX(ts) FROM composite_scores c2 WHERE c2.symbol = composite_scores.symbol
             )
            """
        )
    except Exception as e:
        logger.debug("intuition: composite_scores read failed: %s", e)
        return out
    for sym, comp, ts in cur.fetchall():
        out[str(sym)] = (float(comp), float(ts))
    return out


def _previous_composites(
    conn,
    *,
    latest_ts_by_symbol: dict[str, float],
    lookback_s: float = NOVELTY_LOOKBACK_S,
) -> dict[str, float]:
    """Per-symbol, fetch the most recent ``composite_score`` strictly older
    than the latest row and within ``lookback_s`` seconds.  Used for
    novelty.  Symbols with no prior return nothing."""
    if not latest_ts_by_symbol:
        return {}
    out: dict[str, float] = {}
    for sym, latest_ts in latest_ts_by_symbol.items():
        cutoff = latest_ts - lookback_s
        try:
            cur = conn.execute(
                """
                SELECT composite_score FROM composite_scores
                 WHERE symbol = ? AND ts < ? AND ts >= ?
                 ORDER BY ts DESC LIMIT 1
                """,
                (sym, latest_ts, cutoff),
            )
            row = cur.fetchone()
        except Exception:
            row = None
        if row is not None:
            out[sym] = float(row[0])
    return out


def _global_ic(conn) -> dict[str, float]:
    """Global per-signal IC over the combiner's rolling window.

    Re-uses ``signals.combiner._compute_signal_ic`` (same data the
    weights are derived from) so the score is consistent with what the
    combiner already trusts."""
    try:
        from signals.combiner import _compute_signal_ic, SIGNAL_REGISTRY
    except Exception as e:
        logger.debug("intuition: combiner import failed: %s", e)
        return {}
    try:
        names = sorted(SIGNAL_REGISTRY.keys())
        if not names:
            return {}
        stats = _compute_signal_ic(conn, names)
    except Exception as e:
        logger.debug("intuition: ic compute failed: %s", e)
        return {}
    return {sig: float(d.get("ic", 0.0)) for sig, d in stats.items()}


# ── Score computation ───────────────────────────────────────────


def compute_attention_scores(
    conn,
    *,
    universe: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Compute the attention score for each symbol with current data.

    Returns ``{symbol: {"score", "composite", "novelty", "trust",
    "drivers"}}``.  ``drivers`` is a list of ``(signal_name,
    contribution_magnitude, signed_score)`` sorted by ``|contribution|``
    descending; contribution = ``|score| * |IC| * weight``.

    Returns ``{}`` on any read failure.
    """
    composites = _latest_composites(conn)
    if not composites:
        return {}
    if universe is not None:
        wanted = {s.upper() for s in universe}
        composites = {k: v for k, v in composites.items() if k.upper() in wanted}
        if not composites:
            return {}

    sigs_by_sym = _latest_signal_scores(conn)
    weights = _latest_weights(conn)
    ic = _global_ic(conn)
    prevs = _previous_composites(
        conn,
        latest_ts_by_symbol={sym: ts for sym, (_, ts) in composites.items()},
    )

    out: dict[str, dict[str, Any]] = {}
    for sym, (comp, _ts) in composites.items():
        sym_scores = sigs_by_sym.get(sym, {})
        # Per-signal contribution: |score| * |IC| * weight
        contributions: list[tuple[str, float, float]] = []
        trust = 0.0
        for sig_name, score in sym_scores.items():
            w = weights.get(sig_name)
            i = ic.get(sig_name)
            if w is None or i is None:
                continue
            mag = abs(float(score)) * abs(float(i)) * float(w)
            if mag <= 0.0:
                continue
            trust += mag
            contributions.append((sig_name, float(mag), float(score)))
        contributions.sort(key=lambda x: x[1], reverse=True)

        prev = prevs.get(sym)
        novelty = 0.0 if prev is None else float(comp) - float(prev)
        score = abs(float(comp)) * trust + abs(novelty)
        out[sym] = {
            "score": float(score),
            "composite": float(comp),
            "novelty": float(novelty),
            "trust": float(trust),
            "drivers": contributions,
        }
    return out


# ── Render ───────────────────────────────────────────────────────


def _sign(x: float) -> str:
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"


def render_intuition_block(
    conn,
    *,
    top_n: int = TOP_N,
    top_drivers: int = TOP_DRIVERS,
    universe: Optional[list[str]] = None,
) -> str:
    """Render the INTUITION block for the cycle prompt.

    Header is ``INTUITION (top {N} by attention score)``; each row::

        1. UNH  composite=+1.42  novelty=+0.31  drivers: momentum(+), gap(+), iv_rank(-)

    Returns ``""`` if nothing to show (no composites yet, no signals,
    or the read failed).  Always fail-soft.
    """
    try:
        scored = compute_attention_scores(conn, universe=universe)
    except Exception as e:
        logger.debug("intuition.render: compute failed: %s", e)
        return ""
    if not scored:
        return ""
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_n]
    if not ranked:
        return ""
    lines = [f"INTUITION (top {len(ranked)} by attention score)"]
    for i, (sym, info) in enumerate(ranked, start=1):
        comp = info["composite"]
        nov = info["novelty"]
        drivers = info["drivers"][:top_drivers]
        if drivers:
            drv_text = ", ".join(
                f"{name}({_sign(signed)})" for name, _mag, signed in drivers
            )
        else:
            drv_text = "no contributing signals"
        lines.append(
            f"{i}. {sym}  composite={comp:+.2f}  novelty={nov:+.2f}  drivers: {drv_text}"
        )
    return "\n".join(lines)
