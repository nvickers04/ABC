"""
Memory Layer — PostgreSQL persistence for research + trading.

The signal-combination engine owns
the live tables (signal_scores, signal_weights, composite_scores,
template_recommendations, template_performance). Execution writes to
trades + trade_feedback; the trader and research loops both read the
signal-engine tables. The legacy slot-system tables (live_signals,
strategies, slot_environment_scores) have been retired and are no
longer created on fresh DBs.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import session_state
from memory.repos.schema import (
    SCHEMA_VERSION,
    _DB_PATH,
    get_db,
    get_schema_version,
    init_db,
    reset_connections,
)

logger = logging.getLogger(__name__)

# Back-compat: tests and legacy code may touch these module-level names.
_pending_graduated_params = session_state.pending_graduated_params
_pending_order_context = session_state.pending_order_context


# ═══════════════════════════════════════════════════════════════
# SELF-TUNABLE RESEARCH CONFIG
# ═══════════════════════════════════════════════════════════════
# Implementations live in memory.repos.config_repo (PR12 split).
# Thin shims preserved for back-compat with `from memory import ...`.

def get_research_config(key: str, default: float) -> float:
    from memory.repos.config_repo import get_research_config as _impl
    return _impl(key, default)


def set_research_config(key: str, value: float, reason: str = "") -> None:
    from memory.repos.config_repo import set_research_config as _impl
    return _impl(key, value, reason)


# ═══════════════════════════════════════════════════════════════
# LATEST QUOTES (real-time stock NBBO mirror)
# ═══════════════════════════════════════════════════════════════
# Written by IBKRQuoteSource on every successful tick read.
# Read by the future research daemon (so it doesn't need its own
# IBKR streaming subscription) and by anything that wants a
# point-in-time look at what the trader most recently saw.

def write_latest_quote(quote, source: str = "ibkr") -> None:
    """Upsert a single symbol's latest quote.

    `quote` is anything with .symbol, .last, .bid, .ask, .volume,
    .high, .low, .ts (the IBKRQuote dataclass satisfies this).
    Errors are swallowed — quote mirroring must never block the trader.
    """
    try:
        db = get_db()
        db.execute(
            "INSERT INTO latest_quotes (symbol, last, bid, ask, volume, high, low, ts, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET "
            "last=excluded.last, bid=excluded.bid, ask=excluded.ask, "
            "volume=excluded.volume, high=excluded.high, low=excluded.low, "
            "ts=excluded.ts, source=excluded.source",
            (
                quote.symbol.upper(),
                quote.last,
                quote.bid,
                quote.ask,
                int(quote.volume or 0),
                quote.high,
                quote.low,
                float(quote.ts),
                source,
            ),
        )
        db.commit()
    except Exception as e:
        logger.debug("write_latest_quote(%s) failed: %s", getattr(quote, "symbol", "?"), e)


def read_latest_quote(symbol: str) -> dict | None:
    """Return the most recent latest_quotes row for `symbol` or None."""
    try:
        row = get_db().execute(
            "SELECT symbol, last, bid, ask, volume, high, low, ts, source "
            "FROM latest_quotes WHERE symbol = ?",
            (symbol.upper(),),
        ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.debug("read_latest_quote(%s) failed: %s", symbol, e)
        return None


def get_all_research_config() -> dict[str, float]:
    from memory.repos.config_repo import get_all_research_config as _impl
    return _impl()


# ═══════════════════════════════════════════════════════════════
# IV HISTORY (for trailing-percentile IV rank)
# ═══════════════════════════════════════════════════════════════
# Implementations live in memory.repos.execution_repo (PR13 split).
# `_IV_HISTORY_MIN_SAMPLES` is re-exported for back-compat with
# tests that import it directly from `memory`. The actual import is
# deferred to the bottom of this module so the writer functions
# (insert_execution_snapshot, etc.) that execution_repo re-exports
# back from memory have already been defined.


def record_iv_snapshot(
    symbol: str, iv_current: float | None, source: str = "marketdata"
) -> None:
    from memory.repos.execution_repo import record_iv_snapshot as _impl
    return _impl(symbol, iv_current, source)


def compute_iv_rank_percentile(
    symbol: str,
    iv_current: float | None = None,
    lookback_days: int = 252,
) -> float | None:
    from memory.repos.execution_repo import compute_iv_rank_percentile as _impl
    return _impl(symbol, iv_current=iv_current, lookback_days=lookback_days)


# ═══════════════════════════════════════════════════════════════
# EXECUTION COST MODEL
# ═══════════════════════════════════════════════════════════════

def get_execution_cost(symbol: str | None = None) -> dict:
    # Implementation lives in memory.repos.execution_repo (PR13 split).
    from memory.repos.execution_repo import get_execution_cost as _impl
    return _impl(symbol)


# ═══════════════════════════════════════════════════════════════
# TRADE RECORDING + EXECUTION (memory.repos.execution_repo)
# ═══════════════════════════════════════════════════════════════

def record_trade(
    symbol: str,
    side: str,
    pnl: float,
    held_minutes: int = 0,
    signal_id: int | None = None,
) -> int | None:
    from memory.repos.execution_repo import record_trade as _impl
    return _impl(symbol, side, pnl, held_minutes=held_minutes, signal_id=signal_id)


def _time_bucket(ts_iso: str | None) -> str:
    from memory.repos.execution_repo import _time_bucket as _impl
    return _impl(ts_iso)


def _atr_bucket(atr_pct: float | None) -> str:
    from memory.repos.execution_repo import _atr_bucket as _impl
    return _impl(atr_pct)


def set_pending_graduated_param(symbol: str, param_id: int) -> None:
    session_state.pending_graduated_params[symbol] = param_id


def get_pending_graduated_param(symbol: str) -> int | None:
    return session_state.pending_graduated_params.get(symbol)


def set_pending_order_context(symbol: str, context: dict) -> None:
    session_state.pending_order_context[symbol] = context


def get_pending_order_context(symbol: str) -> dict:
    return session_state.pending_order_context.get(symbol, {})


def insert_execution_snapshot(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    intent: str,
    bid: float | None,
    ask: float | None,
    mid: float | None,
    spread: float | None,
    volume: int | None,
    atr: float | None,
    atr_pct: float | None,
    graduated_param_id: int | None = None,
    algo_strategy: str | None = None,
    order_tif: str | None = None,
) -> int | None:
    from memory.repos.execution_repo import insert_execution_snapshot as _impl
    return _impl(
        symbol, side, quantity, order_type, intent,
        bid, ask, mid, spread, volume, atr, atr_pct,
        graduated_param_id=graduated_param_id,
        algo_strategy=algo_strategy, order_tif=order_tif,
    )


def update_execution_snapshot_fill(
    snapshot_id: int,
    fill_price: float,
    fill_time: str,
    commission: float,
    partial_fills: int = 0,
) -> None:
    from memory.repos.execution_repo import update_execution_snapshot_fill as _impl
    return _impl(snapshot_id, fill_price, fill_time, commission, partial_fills)


def cancel_execution_snapshot(snapshot_id: int) -> None:
    from memory.repos.execution_repo import cancel_execution_snapshot as _impl
    return _impl(snapshot_id)


def get_new_snapshot_count() -> int:
    from memory.repos.execution_repo import get_new_snapshot_count as _impl
    return _impl()


def get_filled_snapshots(since_id: int = 0, limit: int = 500) -> list[dict]:
    from memory.repos.execution_repo import get_filled_snapshots as _impl
    return _impl(since_id=since_id, limit=limit)


def get_snapshots_for_param_review(param_id: int, activated_ts: str) -> dict:
    from memory.repos.execution_repo import get_snapshots_for_param_review as _impl
    return _impl(param_id, activated_ts)


def upsert_calibrated_slippage(
    order_type: str,
    time_bucket: str,
    atr_bucket: str,
    median_bps: float,
    sample_count: int,
    p25_bps: float | None = None,
    p75_bps: float | None = None,
) -> None:
    from memory.repos.execution_repo import upsert_calibrated_slippage as _impl
    return _impl(
        order_type, time_bucket, atr_bucket, median_bps, sample_count,
        p25_bps=p25_bps, p75_bps=p75_bps,
    )



def reset_state(db_path=None) -> None:
    """Reset all module-level mutable state — primarily for tests.

    Single chokepoint that test fixtures (``tests/conftest.py``) and
    future DI scaffolding can call instead of poking five different
    module attributes from outside. Closes any open connection, clears
    the two pending lookup dicts, resets the calibration version, and
    optionally repoints the DB path (legacy; Postgres uses ``DATABASE_URL``).
    """
    global _DB_PATH
    if db_path is not None:
        from memory.repos import schema as _schema
        p = db_path if isinstance(db_path, Path) else Path(db_path)
        _schema._DB_PATH = p
        _DB_PATH = p
    reset_connections()
    session_state.reset_session_state()


def get_graduated_params(active_only: bool = True) -> list[dict]:
    from memory.repos.config_repo import get_graduated_params as _impl
    return _impl(active_only=active_only)


def deactivate_graduated_param(param_id: int, reason: str) -> None:
    from memory.repos.config_repo import deactivate_graduated_param as _impl
    return _impl(param_id, reason)


def validate_param_key(key: str) -> str | None:
    from memory.repos.config_repo import validate_param_key as _impl
    return _impl(key)


def insert_graduated_param(
    param_key: str,
    param_value: str,
    previous_value: str | None,
    evidence_json: str,
    snapshots_analyzed: int,
    improvement_bps: float,
    p_value: float,
) -> int | None:
    from memory.repos.config_repo import insert_graduated_param as _impl
    return _impl(
        param_key, param_value, previous_value, evidence_json,
        snapshots_analyzed, improvement_bps, p_value,
    )


def get_calibration_version() -> int:
    from memory.repos.config_repo import get_calibration_version as _impl
    return _impl()


def get_open_hypotheses(slot: int | None = None, limit: int = 10) -> list[dict]:
    from memory.repos.feedback_repo import get_open_hypotheses as _impl
    return _impl(slot=slot, limit=limit)


def mark_hypothesis_incorporated(hypothesis_type: str, description: str) -> None:
    from memory.repos.feedback_repo import mark_hypothesis_incorporated as _impl
    return _impl(hypothesis_type, description)


def get_calibrated_slippage() -> dict[tuple[str, str, str], float]:
    # Implementation lives in memory.repos.execution_repo (PR13 split).
    from memory.repos.execution_repo import get_calibrated_slippage as _impl
    return _impl()


# ── QualityMatrix provenance (memory.repos.provenance_repo) ────────

def insert_tool_usage_log(
    tool_name: str,
    symbol: str | None = None,
    success: bool = True,
    latency_ms: float = 0.0,
    cycle_id: int = 0,
    decision_context: str | None = None,
) -> int | None:
    from memory.repos.provenance_repo import insert_tool_usage_log as _impl
    return _impl(
        tool_name,
        symbol=symbol,
        success=success,
        latency_ms=latency_ms,
        cycle_id=cycle_id,
        decision_context=decision_context,
    )


def insert_decision_provenance(
    decision_type: str,
    cycle_id: int = 0,
    symbol: str | None = None,
    tools_json: str | None = None,
    quality_state_json: str | None = None,
    context_quality: str = "full",
    outcome: str | None = None,
    notes: str = "",
) -> int | None:
    from memory.repos.provenance_repo import insert_decision_provenance as _impl
    return _impl(
        decision_type,
        cycle_id=cycle_id,
        symbol=symbol,
        tools_json=tools_json,
        quality_state_json=quality_state_json,
        context_quality=context_quality,
        outcome=outcome,
        notes=notes,
    )


def get_recent_tool_usage(limit: int = 50) -> list[dict]:
    from memory.repos.provenance_repo import get_recent_tool_usage as _impl
    return _impl(limit)


def get_recent_decision_provenance(limit: int = 20) -> list[dict]:
    from memory.repos.provenance_repo import get_recent_decision_provenance as _impl
    return _impl(limit)


# -- Back-compat re-exports (PR13 split) -------------------------
# _IV_HISTORY_MIN_SAMPLES is read by tests via `from memory import`.
# Imported at the bottom of this module (after writer functions are
# defined) to avoid circular-import issues with `execution_repo`,
# which re-exports those writers back from `memory`.
from memory.repos.execution_repo import _IV_HISTORY_MIN_SAMPLES  # noqa: E402,F401

