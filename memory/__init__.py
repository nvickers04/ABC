"""
Memory Layer — SQLite persistence for research + trading.

Single file (memory/abc.db), WAL mode, three core tables + live_signals.
Both research and trading agents share this DB with clear ownership:
  - Research writes: strategies, evaluations, signals, live_signals
  - Trading writes: trades
  - Trading reads: strategies, live_signals
  - Research reads: trades (for future feedback)
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / "abc.db"
_connection: sqlite3.Connection | None = None


def _ensure_column(conn, table: str, col: str, col_type: str) -> None:
    """Helper to idempotently add a column if it doesn't exist (simplifies migrations)."""
    try:
        conn.execute(f"SELECT {col} FROM {table} LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass


def init_db() -> sqlite3.Connection:
    """Initialize the database, create tables if missing, return connection."""
    global _connection
    if _connection is not None:
        return _connection

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            slot INTEGER NOT NULL DEFAULT 1,
            methodology TEXT NOT NULL,
            parent_id INTEGER,
            total_signals INTEGER DEFAULT 0,
            hit_rate REAL,
            avg_rr REAL,
            expectancy REAL,
            kept INTEGER DEFAULT 0,
            llm_analysis TEXT,
            FOREIGN KEY (parent_id) REFERENCES strategies(id)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            ts TEXT NOT NULL,
            eval_date TEXT NOT NULL,
            symbols_json TEXT,
            signals_tested INTEGER DEFAULT 0,
            hit_rate REAL,
            expectancy REAL,
            avg_rr REAL,
            profit_factor REAL,
            max_drawdown REAL,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            evaluation_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            entry_ts TEXT,
            direction TEXT,
            order_type TEXT,
            setup_type TEXT,
            entry_price REAL,
            target_price REAL,
            stop_price REAL,
            max_hold_bars INTEGER,
            hit_target INTEGER,
            hit_stop INTEGER,
            timed_out INTEGER,
            exit_ts TEXT,
            exit_price REAL,
            return_pct REAL,
            legs_json TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id),
            FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            pnl REAL,
            signal_id INTEGER,
            held_minutes INTEGER,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        );

        CREATE TABLE IF NOT EXISTS live_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            slot INTEGER NOT NULL DEFAULT 1,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT,
            order_type TEXT,
            setup_type TEXT,
            entry_price REAL,
            target_price REAL,
            stop_price REAL,
            max_hold_bars INTEGER,
            legs_json TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );

        CREATE TABLE IF NOT EXISTS environment_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            round_num INTEGER,
            volatility_regime TEXT,
            trend_regime TEXT,
            breadth_regime TEXT,
            momentum_regime TEXT,
            volume_regime TEXT,
            avg_atr_pct REAL,
            dispersion REAL,
            strategy_fit_json TEXT,
            raw_snapshot_json TEXT
        );

        CREATE TABLE IF NOT EXISTS slot_environment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            slot INTEGER NOT NULL,
            env_snapshot_id INTEGER NOT NULL,
            fitness REAL,
            expectancy REAL,
            total_signals INTEGER,
            strategy_type TEXT,
            FOREIGN KEY (env_snapshot_id) REFERENCES environment_snapshots(id)
        );

        CREATE TABLE IF NOT EXISTS trade_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            trade_id INTEGER,
            signal_id INTEGER,
            slot INTEGER,
            simulated_return REAL,
            actual_pnl REAL,
            execution_gap REAL,
            symbol TEXT,
            FOREIGN KEY (trade_id) REFERENCES trades(id),
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        );

        CREATE INDEX IF NOT EXISTS idx_strategies_kept ON strategies(kept);
        CREATE INDEX IF NOT EXISTS idx_strategies_slot ON strategies(slot);
        CREATE INDEX IF NOT EXISTS idx_evaluations_strategy ON evaluations(strategy_id);
        CREATE INDEX IF NOT EXISTS idx_signals_evaluation ON signals(evaluation_id);
        CREATE INDEX IF NOT EXISTS idx_live_signals_ts ON live_signals(ts);
        CREATE INDEX IF NOT EXISTS idx_live_signals_slot ON live_signals(slot);
        CREATE INDEX IF NOT EXISTS idx_env_snapshots_ts ON environment_snapshots(ts);
        CREATE INDEX IF NOT EXISTS idx_slot_env_scores_slot ON slot_environment_scores(slot);
        CREATE INDEX IF NOT EXISTS idx_slot_env_scores_env ON slot_environment_scores(env_snapshot_id);
        CREATE INDEX IF NOT EXISTS idx_trade_feedback_slot ON trade_feedback(slot);
    """)

    # ── New Phase-2+ tables ──────────────────────────────────────
    conn.executescript("""
        -- Promotion pipeline: records each time a slot candidate is evaluated
        -- against the promotion-grade (historical options-repricing) standard.
        CREATE TABLE IF NOT EXISTS promotion_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            slot INTEGER NOT NULL,
            strategy_id INTEGER,
            eval_type TEXT NOT NULL,          -- 'fast_search' | 'promotion_grade'
            search_fitness REAL,
            promotion_fitness REAL,
            options_coverage_pct REAL,        -- % of options legs successfully repriced
            promoted INTEGER DEFAULT 0,       -- 1 if passed promotion gate
            rejection_reason TEXT,
            notes TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );

        -- Deterministic replay sessions around the live trader agent
        CREATE TABLE IF NOT EXISTS replay_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session_date TEXT NOT NULL,       -- date being replayed 'YYYY-MM-DD'
            slot INTEGER,
            strategy_id INTEGER,
            tool_calls_json TEXT,             -- serialized list of tool call records
            action_trace_json TEXT,           -- serialized list of agent actions
            total_pnl REAL,
            total_fills INTEGER DEFAULT 0,
            outcome TEXT,                     -- 'pass' | 'fail' | 'skipped'
            notes TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );

        -- Hypotheses generated by the trader agent about missed opportunities
        -- Extended for self-improvement mirroring research (fitness, env tracking, keep/discard)
        CREATE TABLE IF NOT EXISTS trader_hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,             -- 'trader_agent' | 'research_agent' | 'human'
            hypothesis_type TEXT NOT NULL,    -- 'missed_opportunity' | 'regime_mismatch' | 'execution_gap' | 'slot_gap'
            description TEXT NOT NULL,
            suggested_action TEXT,
            priority INTEGER DEFAULT 5,       -- 1 (high) to 10 (low)
            status TEXT DEFAULT 'open',       -- 'open' | 'in_progress' | 'incorporated' | 'rejected'
            related_trade_id INTEGER,
            related_slot INTEGER,
            env_key TEXT,                     -- e.g. 'high-vol-bearish'
            signed_fitness REAL,              -- from simulator signed_edge_score / raw_search_fitness
            kept INTEGER DEFAULT 0,           -- 1 if incorporated into trader behavior
            resolution_notes TEXT,
            FOREIGN KEY (related_trade_id) REFERENCES trades(id)
        );

        CREATE INDEX IF NOT EXISTS idx_promotion_runs_slot ON promotion_runs(slot);
        CREATE INDEX IF NOT EXISTS idx_promotion_runs_ts ON promotion_runs(ts);
        CREATE INDEX IF NOT EXISTS idx_replay_episodes_date ON replay_episodes(session_date);
        CREATE INDEX IF NOT EXISTS idx_trader_hyp_status ON trader_hypotheses(status);

        -- Self-tunable research config: key-value store the agent can read + update.
        -- Defaults live in research/config.py; DB overrides take precedence.
        CREATE TABLE IF NOT EXISTS research_config (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL,
            updated_ts TEXT NOT NULL,
            reason TEXT
        );
    """)

    # ── Migrations for existing DBs ─────────────────────────────
    # Use helper for idempotent column adds (simplified from repeated try/except)
    _ensure_column(conn, "strategies", "slot", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "live_signals", "slot", "INTEGER NOT NULL DEFAULT 1")

    # Continuous env metrics
    _env_continuous_cols = [
        ("avg_intraday_range_pct", "REAL"),
        ("avg_cumulative_return", "REAL"),
        ("advance_decline_ratio", "REAL"),
        ("avg_momentum_shift", "REAL"),
        ("avg_volume_ratio", "REAL"),
        ("pct_trending_up", "REAL"),
        ("pct_trending_down", "REAL"),
    ]
    for col_name, col_type in _env_continuous_cols:
        _ensure_column(conn, "environment_snapshots", col_name, col_type)

    conn.commit()
    _connection = conn
    logger.info(f"Memory DB initialized: {_DB_PATH}")
    return conn


def get_db() -> sqlite3.Connection:
    """Get the database connection, initializing if needed."""
    if _connection is None:
        return init_db()
    return _connection


# ═══════════════════════════════════════════════════════════════
# SELF-TUNABLE RESEARCH CONFIG
# ═══════════════════════════════════════════════════════════════

def get_research_config(key: str, default: float) -> float:
    """Read a tunable config value.  Returns DB override if set, else default."""
    row = get_db().execute(
        "SELECT value FROM research_config WHERE key = ?", (key,)
    ).fetchone()
    return float(row["value"]) if row else default


def set_research_config(key: str, value: float, reason: str = "") -> None:
    """Write (upsert) a tunable config value with an audit trail."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    get_db().execute(
        """INSERT INTO research_config (key, value, updated_ts, reason)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(key) DO UPDATE
             SET value = excluded.value,
                 updated_ts = excluded.updated_ts,
                 reason = excluded.reason""",
        (key, value, ts, reason),
    )
    get_db().commit()
    logger.info(f"research_config[{key}] = {value}  ({reason})")


def get_all_research_config() -> dict[str, float]:
    """Return all DB-stored tunable config values as {key: value}."""
    rows = get_db().execute("SELECT key, value FROM research_config").fetchall()
    return {r["key"]: float(r["value"]) for r in rows}
