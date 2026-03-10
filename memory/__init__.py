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

    # ── Migrations for existing DBs ─────────────────────────────
    # Add slot column if upgrading from pre-slot schema (or track-based schema)
    try:
        conn.execute("SELECT slot FROM strategies LIMIT 1")
    except sqlite3.OperationalError:
        # Try adding slot; if track exists, we're migrating from track-based
        try:
            conn.execute("ALTER TABLE strategies ADD COLUMN slot INTEGER NOT NULL DEFAULT 1")
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute("SELECT slot FROM live_signals LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute("ALTER TABLE live_signals ADD COLUMN slot INTEGER NOT NULL DEFAULT 1")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    _connection = conn
    logger.info(f"Memory DB initialized: {_DB_PATH}")
    return conn


def get_db() -> sqlite3.Connection:
    """Get the database connection, initializing if needed."""
    if _connection is None:
        return init_db()
    return _connection
