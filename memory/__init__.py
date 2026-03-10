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
            track TEXT NOT NULL DEFAULT 'market',
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
            track TEXT NOT NULL DEFAULT 'market',
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

        CREATE INDEX IF NOT EXISTS idx_strategies_kept ON strategies(kept);
        CREATE INDEX IF NOT EXISTS idx_strategies_track ON strategies(track);
        CREATE INDEX IF NOT EXISTS idx_evaluations_strategy ON evaluations(strategy_id);
        CREATE INDEX IF NOT EXISTS idx_signals_evaluation ON signals(evaluation_id);
        CREATE INDEX IF NOT EXISTS idx_live_signals_ts ON live_signals(ts);
        CREATE INDEX IF NOT EXISTS idx_live_signals_track ON live_signals(track);
    """)

    # ── Migrations for existing DBs ─────────────────────────────
    # Add track column if upgrading from pre-multi-track schema
    try:
        conn.execute("SELECT track FROM strategies LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE strategies ADD COLUMN track TEXT NOT NULL DEFAULT 'market'")
    try:
        conn.execute("SELECT track FROM live_signals LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE live_signals ADD COLUMN track TEXT NOT NULL DEFAULT 'market'")

    conn.commit()
    _connection = conn
    logger.info(f"Memory DB initialized: {_DB_PATH}")
    return conn


def get_db() -> sqlite3.Connection:
    """Get the database connection, initializing if needed."""
    if _connection is None:
        return init_db()
    return _connection
