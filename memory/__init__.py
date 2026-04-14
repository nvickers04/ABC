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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / "abc.db"
_connection: sqlite3.Connection | None = None


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
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
    conn.execute("PRAGMA wal_autocheckpoint=1000")

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
        CREATE INDEX IF NOT EXISTS idx_trade_feedback_slot_ts ON trade_feedback(slot, ts);
        CREATE INDEX IF NOT EXISTS idx_signals_strategy_date ON signals(strategy_id, evaluation_id);
        CREATE INDEX IF NOT EXISTS idx_live_signals_slot_ts ON live_signals(slot, ts);
        CREATE INDEX IF NOT EXISTS idx_live_signals_symbol ON live_signals(symbol);
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
        CREATE INDEX IF NOT EXISTS idx_trader_hyp_status_env ON trader_hypotheses(status, env_key);
        CREATE INDEX IF NOT EXISTS idx_slot_env_scores_slot_env ON slot_environment_scores(slot, env_snapshot_id);

        -- Self-tunable research config: key-value store the agent can read + update.
        -- Defaults live in research/config.py; DB overrides take precedence.
        CREATE TABLE IF NOT EXISTS research_config (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL,
            updated_ts TEXT NOT NULL,
            reason TEXT
        );

        -- Trader learned rules: behavioral rules the trader agent has evolved
        CREATE TABLE IF NOT EXISTS trader_learned_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            rule_text TEXT NOT NULL,          -- concise rule the agent should follow
            env_key TEXT,                     -- regime this rule applies to (NULL = all)
            signed_fitness REAL DEFAULT 0,    -- rule fitness (positive = helpful)
            kept INTEGER DEFAULT 1,           -- 1 = active, 0 = discarded
            actual_pnl REAL,                  -- realized P&L when rule was followed
            simulated_pnl REAL,               -- simulated P&L for comparison
            parent_id INTEGER,                -- if evolved from another rule
            source TEXT DEFAULT 'daily_review', -- 'daily_review' | 'hypothesis' | 'human'
            FOREIGN KEY (parent_id) REFERENCES trader_learned_rules(id)
        );

        -- Trader evaluations: daily performance review of rules
        CREATE TABLE IF NOT EXISTS trader_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id INTEGER NOT NULL,
            eval_date TEXT NOT NULL,
            trades_tested INTEGER DEFAULT 0,
            hit_rate REAL,
            pnl REAL,
            execution_gap REAL,              -- avg gap between simulated and actual
            FOREIGN KEY (rule_id) REFERENCES trader_learned_rules(id)
        );

        CREATE INDEX IF NOT EXISTS idx_trader_rules_env ON trader_learned_rules(env_key);
        CREATE INDEX IF NOT EXISTS idx_trader_rules_kept ON trader_learned_rules(kept);
        CREATE INDEX IF NOT EXISTS idx_trader_evals_rule ON trader_evaluations(rule_id);
        CREATE INDEX IF NOT EXISTS idx_trader_evals_date ON trader_evaluations(eval_date);
    """)

    # ── Execution autoresearch tables ────────────────────────────
    conn.executescript("""
        -- Execution snapshots: ground truth for every order placed
        CREATE TABLE IF NOT EXISTS execution_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,                -- submit timestamp
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,              -- 'BUY' or 'SELL'
            quantity INTEGER,
            order_type TEXT,                 -- canonical: 'market', 'limit', 'adaptive', etc.
            intent TEXT,                     -- 'entry', 'exit', 'stop'
            -- Market data at submit time
            bid_at_submit REAL,
            ask_at_submit REAL,
            mid_at_submit REAL,
            spread_at_submit REAL,
            volume_at_submit INTEGER,
            atr_at_submit REAL,
            -- Fill data (updated after fill)
            fill_price REAL,
            fill_time TEXT,
            commission REAL,
            partial_fills INTEGER DEFAULT 0,
            -- Derived metrics (computed on fill)
            slippage_bps REAL,              -- (fill_price - mid_at_submit) / mid_at_submit * 10000
            latency_ms REAL,                -- fill_time - submit_time in ms
            -- Classification buckets
            time_bucket TEXT,               -- 'open' | 'morning' | 'midday' | 'close'
            atr_bucket TEXT,                -- 'low' | 'medium' | 'high'
            -- Link to graduated param override if any
            graduated_param_id INTEGER,
            -- Status
            status TEXT DEFAULT 'submitted' -- 'submitted' | 'filled' | 'cancelled'
        );

        -- Graduated parameters: statistically-backed config overrides
        CREATE TABLE IF NOT EXISTS graduated_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            param_key TEXT NOT NULL,          -- e.g. 'market.entry.open.high'
            param_value TEXT NOT NULL,        -- e.g. 'adaptive'
            previous_value TEXT,              -- what it replaced
            evidence_json TEXT,               -- stats that justified the change
            snapshots_analyzed INTEGER,       -- how many snapshots backed this
            improvement_bps REAL,             -- estimated improvement in bps
            p_value REAL,                     -- statistical significance
            active INTEGER DEFAULT 1,         -- 1 = in effect, 0 = rolled back
            rollback_reason TEXT
        );

        -- Calibrated slippage: empirical slippage lookup for simulator
        CREATE TABLE IF NOT EXISTS calibrated_slippage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            order_type TEXT NOT NULL,
            time_bucket TEXT NOT NULL,        -- 'open' | 'morning' | 'midday' | 'close' | 'all'
            atr_bucket TEXT NOT NULL,         -- 'low' | 'medium' | 'high' | 'all'
            median_slippage_bps REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            p25_bps REAL,
            p75_bps REAL,
            UNIQUE(order_type, time_bucket, atr_bucket)
        );

        CREATE INDEX IF NOT EXISTS idx_exec_snap_ts ON execution_snapshots(ts);
        CREATE INDEX IF NOT EXISTS idx_exec_snap_symbol ON execution_snapshots(symbol);
        CREATE INDEX IF NOT EXISTS idx_exec_snap_status ON execution_snapshots(status);
        CREATE INDEX IF NOT EXISTS idx_exec_snap_buckets ON execution_snapshots(order_type, time_bucket, atr_bucket);
        CREATE INDEX IF NOT EXISTS idx_grad_params_active ON graduated_params(active);
        CREATE INDEX IF NOT EXISTS idx_grad_params_key ON graduated_params(param_key);
        CREATE INDEX IF NOT EXISTS idx_calib_slip_lookup ON calibrated_slippage(order_type, time_bucket, atr_bucket);
    """)

    # ── Signal combination engine tables ─────────────────────────
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signal_scores (
            signal_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            ts REAL NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            components_json TEXT,
            PRIMARY KEY (signal_name, symbol, ts)
        );

        CREATE TABLE IF NOT EXISTS signal_weights (
            signal_name TEXT NOT NULL,
            weight REAL NOT NULL,
            n_eff REAL,
            category TEXT,
            updated_ts REAL NOT NULL,
            PRIMARY KEY (signal_name)
        );

        CREATE TABLE IF NOT EXISTS signal_returns (
            signal_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            ts REAL NOT NULL,
            score_at_entry REAL NOT NULL,
            forward_return REAL NOT NULL,
            r_value REAL NOT NULL,
            horizon_bars INTEGER NOT NULL,
            PRIMARY KEY (signal_name, symbol, ts)
        );

        CREATE TABLE IF NOT EXISTS composite_scores (
            symbol TEXT NOT NULL,
            ts REAL NOT NULL,
            composite_score REAL NOT NULL,
            signal_breakdown_json TEXT,
            PRIMARY KEY (symbol, ts)
        );

        CREATE TABLE IF NOT EXISTS template_performance (
            template_name TEXT NOT NULL,
            regime_key TEXT NOT NULL,
            composite_bucket TEXT,
            trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            avg_return_pct REAL DEFAULT 0,
            avg_gap_pct REAL DEFAULT 0,
            sharpe REAL,
            updated_ts REAL NOT NULL,
            PRIMARY KEY (template_name, regime_key, composite_bucket)
        );

        CREATE TABLE IF NOT EXISTS template_boundaries (
            template_name TEXT NOT NULL,
            param_name TEXT NOT NULL,
            param_value REAL NOT NULL,
            generation INTEGER DEFAULT 0,
            fitness REAL,
            updated_ts REAL NOT NULL,
            PRIMARY KEY (template_name, param_name)
        );

        CREATE TABLE IF NOT EXISTS template_recommendations (
            symbol TEXT NOT NULL,
            ts REAL NOT NULL,
            template_name TEXT NOT NULL,
            direction TEXT NOT NULL,
            composite_score REAL NOT NULL,
            order_type TEXT,
            entry_price REAL,
            target_price REAL,
            stop_price REAL,
            legs_json TEXT,
            PRIMARY KEY (symbol, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_signal_scores_ts ON signal_scores(ts);
        CREATE INDEX IF NOT EXISTS idx_signal_returns_signal ON signal_returns(signal_name, ts);
        CREATE INDEX IF NOT EXISTS idx_composite_scores_ts ON composite_scores(ts);
        CREATE INDEX IF NOT EXISTS idx_template_recs_ts ON template_recommendations(ts);
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


# ═══════════════════════════════════════════════════════════════
# EXECUTION COST MODEL
# ═══════════════════════════════════════════════════════════════

def get_execution_cost(symbol: str | None = None) -> dict:
    """Query aggregated execution gaps from trade_feedback.

    Returns per-symbol cost model from last 30 days:
      {symbol -> {avg_gap, n, total_pnl, total_sim}}

    If symbol given, returns single entry.
    If not, returns top-10 symbols by trade count.
    """
    db = get_db()
    cutoff = "datetime('now', '-30 days')"
    if symbol:
        row = db.execute(
            f"""SELECT symbol,
                       AVG(execution_gap) as avg_gap,
                       COUNT(*) as n,
                       SUM(actual_pnl) as total_pnl,
                       SUM(simulated_return) as total_sim
                FROM trade_feedback
                WHERE symbol = ? AND ts >= {cutoff}""",
            (symbol.upper(),),
        ).fetchone()
        if row and row["n"]:
            return {
                "symbol": row["symbol"],
                "avg_gap_pct": round(row["avg_gap"] or 0, 4),
                "trades": row["n"],
                "total_pnl": round(row["total_pnl"] or 0, 2),
                "total_sim": round(row["total_sim"] or 0, 4),
            }
        return {}
    else:
        rows = db.execute(
            f"""SELECT symbol,
                       AVG(execution_gap) as avg_gap,
                       COUNT(*) as n
                FROM trade_feedback
                WHERE ts >= {cutoff}
                GROUP BY symbol
                ORDER BY n DESC LIMIT 10"""
        ).fetchall()
        return {
            r["symbol"]: {"avg_gap_pct": round(r["avg_gap"] or 0, 4), "trades": r["n"]}
            for r in rows
        }


# ═══════════════════════════════════════════════════════════════
# TRADE RECORDING (for fitness feedback loop)
# ═══════════════════════════════════════════════════════════════

def record_trade(
    symbol: str,
    side: str,
    pnl: float,
    held_minutes: int = 0,
    signal_id: int | None = None,
) -> int | None:
    """Record a closed trade into the trades table.

    This feeds the feedback loop: _match_trade_to_signal() matches these
    rows to live_signals by symbol + time proximity, then writes
    trade_feedback rows tracking execution gaps.

    Returns the trade row id, or None on failure.
    """
    try:
        db = get_db()
        cur = db.execute(
            """INSERT INTO trades (ts, symbol, side, pnl, signal_id, held_minutes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                symbol.upper(),
                side,
                round(pnl, 2),
                signal_id,
                held_minutes,
            ),
        )
        db.commit()
        trade_id = cur.lastrowid
        logger.info(f"Trade recorded: {symbol} {side} pnl=${pnl:.2f} id={trade_id}")

        # Immediately try to match this trade to a signal for feedback
        try:
            _match_trade_to_signal(db, trade_id, symbol)
        except Exception as match_err:
            logger.debug(f"Trade-signal matching failed: {match_err}")

        return trade_id
    except Exception as e:
        logger.warning(f"Failed to record trade: {e}")
        return None


def _match_trade_to_signal(db, trade_id: int, symbol: str) -> None:
    """Match a single trade to the closest live_signal by symbol + time proximity.

    Writes to trade_feedback table for execution quality tracking.
    """
    trade = db.execute(
        "SELECT id, symbol, side, pnl, ts FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    if not trade:
        return

    signal = db.execute(
        """SELECT ls.id, ls.slot, ls.direction, ls.entry_price,
                  s.return_pct as sim_return, s.strategy_id
           FROM live_signals ls
           LEFT JOIN signals s ON s.strategy_id = ls.strategy_id
               AND s.symbol = ls.symbol
           WHERE ls.symbol = ?
             AND ABS(julianday(ls.ts) - julianday(?)) < 7.0
           ORDER BY ABS(julianday(ls.ts) - julianday(?)) ASC
           LIMIT 1""",
        (symbol.upper(), trade["ts"], trade["ts"]),
    ).fetchone()

    if not signal:
        logger.debug(f"No matching signal for trade {trade_id} ({symbol})")
        return

    sim_return = signal["sim_return"] or 0
    actual_pnl = trade["pnl"] or 0
    entry_price = signal["entry_price"]

    if entry_price and entry_price > 0:
        actual_return_pct = (actual_pnl / entry_price) * 100
        execution_gap = actual_return_pct - sim_return
    else:
        execution_gap = None

    db.execute(
        """INSERT INTO trade_feedback
           (ts, trade_id, signal_id, slot, simulated_return,
            actual_pnl, execution_gap, symbol)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            trade_id,
            signal["id"],
            signal["slot"],
            sim_return,
            actual_pnl,
            execution_gap,
            symbol.upper(),
        ),
    )
    db.commit()
    gap_str = f"{execution_gap:.2f}%" if execution_gap is not None else "unknown"
    logger.info(
        f"Trade feedback: trade {trade_id} -> signal {signal['id']} "
        f"slot={signal['slot']} gap={gap_str}"
    )


# ═══════════════════════════════════════════════════════════════
# EXECUTION AUTORESEARCH
# ═══════════════════════════════════════════════════════════════

def _time_bucket(ts_iso: str | None) -> str:
    """Classify a timestamp into a time-of-day bucket (ET).

    open     = 09:30–09:45
    morning  = 09:45–12:00
    midday   = 12:00–15:45
    close    = 15:45–16:00
    extended = everything else
    """
    if not ts_iso:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_iso)
        et = dt.astimezone(ZoneInfo("America/New_York"))
        minutes = et.hour * 60 + et.minute
        if 570 <= minutes < 585:
            return "open"
        elif 585 <= minutes < 720:
            return "morning"
        elif 720 <= minutes < 945:
            return "midday"
        elif 945 <= minutes < 960:
            return "close"
        else:
            return "extended"
    except Exception:
        return "unknown"


def _atr_bucket(atr_pct: float | None) -> str:
    """Classify ATR % into volatility bucket."""
    if atr_pct is None:
        return "unknown"
    if atr_pct < 1.5:
        return "low"
    elif atr_pct < 3.0:
        return "medium"
    else:
        return "high"


# Short-lived lookup: symbol → graduated_param_id.
# Set by _plan_order, consumed (popped) by insert_execution_snapshot.
_pending_graduated_params: dict[str, int] = {}

# Short-lived lookup: symbol → {"intent": ..., "atr_pct": ...}.
# Bridges plan_order context to snapshot capture across the LLM tool-call gap.
_pending_order_context: dict[str, dict] = {}

# ── Canonical order type mapping ─────────────────────────────
# IBKR order type string → canonical name used in param_key schema.
# Many IBKR types share the same base order_type (e.g. MKT) but differ
# by algoStrategy or tif, so a bare mapping is insufficient.
# `_normalize_order_type` accepts optional context to disambiguate.
_IBKR_TO_CANONICAL: dict[str, str] = {
    "MKT": "market",
    "LMT": "limit",
    "STP": "stop_entry",
    "STP LMT": "stop_entry",
    "TRAIL": "trailing_stop",
    "TRAIL LIMIT": "trailing_stop",
    "MIDPRICE": "midprice",
    "MOC": "moc",
    "LOC": "loc",
    "REL": "relative",
    "SNAP MID": "snap_mid",
}

# Algo strategies that override the base IBKR type.
_ALGO_TO_CANONICAL: dict[str, str] = {
    "Adaptive": "adaptive",
    "Vwap": "vwap",
    "Twap": "twap",
}


def _normalize_order_type(
    raw: str,
    *,
    algo_strategy: str | None = None,
    tif: str | None = None,
    order_name: str | None = None,
) -> str:
    """Normalize IBKR order type to canonical name for consistent analysis.

    Priority: algo_strategy > tif-based special types > base mapping > pass-through.
    """
    # 1. Algo strategies override the base type entirely
    if algo_strategy and algo_strategy in _ALGO_TO_CANONICAL:
        return _ALGO_TO_CANONICAL[algo_strategy]
    # 2. tif='OPG' with MKT=moo, LMT=loo (opening-auction orders)
    if tif == "OPG":
        if raw == "MKT":
            return "moo"
        if raw == "LMT":
            return "loo"
    # 3. Standard IBKR type mapping
    return _IBKR_TO_CANONICAL.get(raw, raw)


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
    """Insert a snapshot at order submission time. Returns snapshot id."""
    # Normalize IBKR order type to canonical name using full context
    order_type = _normalize_order_type(
        order_type, algo_strategy=algo_strategy, tif=order_tif,
    )
    # Auto-consume pending graduated param if not explicitly provided
    if graduated_param_id is None:
        graduated_param_id = _pending_graduated_params.pop(symbol, None)
    # Auto-consume pending order context (intent, atr_pct) from _plan_order
    _ctx = _pending_order_context.pop(symbol, {})
    if intent == 'unknown' and 'intent' in _ctx:
        intent = _ctx['intent']
    if atr_pct is None and 'atr_pct' in _ctx:
        atr_pct = _ctx['atr_pct']
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO execution_snapshots
               (ts, symbol, side, quantity, order_type, intent,
                bid_at_submit, ask_at_submit, mid_at_submit, spread_at_submit,
                volume_at_submit, atr_at_submit,
                time_bucket, atr_bucket, graduated_param_id, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted')""",
            (
                ts, symbol, side, quantity, order_type, intent,
                bid, ask, mid, spread,
                volume, atr,
                _time_bucket(ts), _atr_bucket(atr_pct),
                graduated_param_id,
            ),
        )
        db.commit()
        return cur.lastrowid
    except Exception as e:
        logger.warning(f"Failed to insert execution snapshot: {e}")
        return None


def update_execution_snapshot_fill(
    snapshot_id: int,
    fill_price: float,
    fill_time: str,
    commission: float,
    partial_fills: int = 0,
) -> None:
    """Update a snapshot with fill data and compute derived metrics."""
    db = get_db()
    try:
        row = db.execute(
            "SELECT mid_at_submit, ts FROM execution_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if not row:
            return

        mid = row["mid_at_submit"]
        submit_ts = row["ts"]

        # Compute slippage in bps (positive = worse than mid)
        slippage_bps = None
        if mid and mid > 0:
            slippage_bps = round((fill_price - mid) / mid * 10000, 2)

        # Compute latency
        latency_ms = None
        try:
            from datetime import datetime as _dt
            submit_dt = _dt.fromisoformat(submit_ts)
            fill_dt = _dt.fromisoformat(fill_time)
            latency_ms = round((fill_dt - submit_dt).total_seconds() * 1000, 0)
        except Exception:
            pass

        db.execute(
            """UPDATE execution_snapshots
               SET fill_price = ?, fill_time = ?, commission = ?,
                   partial_fills = ?, slippage_bps = ?, latency_ms = ?,
                   status = 'filled'
               WHERE id = ?""",
            (fill_price, fill_time, commission, partial_fills, slippage_bps, latency_ms, snapshot_id),
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to update execution snapshot fill: {e}")


def cancel_execution_snapshot(snapshot_id: int) -> None:
    """Mark a snapshot as cancelled."""
    try:
        db = get_db()
        db.execute(
            "UPDATE execution_snapshots SET status = 'cancelled' WHERE id = ?",
            (snapshot_id,),
        )
        db.commit()
    except Exception:
        pass


def get_new_snapshot_count() -> int:
    """Count filled snapshots since last analysis run."""
    db = get_db()
    last_analysis = get_research_config("last_analysis_snapshot_id", 0.0)
    row = db.execute(
        "SELECT COUNT(*) as n FROM execution_snapshots WHERE status = 'filled' AND id > ?",
        (int(last_analysis),),
    ).fetchone()
    return row["n"] if row else 0


def get_filled_snapshots(since_id: int = 0, limit: int = 500) -> list[dict]:
    """Get filled snapshots since a given id."""
    db = get_db()
    rows = db.execute(
        """SELECT * FROM execution_snapshots
           WHERE status = 'filled' AND id > ?
           ORDER BY id ASC LIMIT ?""",
        (since_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_graduated_params(active_only: bool = True) -> list[dict]:
    """Get graduated parameter overrides."""
    db = get_db()
    where = "WHERE active = 1" if active_only else ""
    rows = db.execute(
        f"SELECT * FROM graduated_params {where} ORDER BY ts DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def deactivate_graduated_param(param_id: int, reason: str) -> None:
    """Deactivate a graduated param and record why."""
    db = get_db()
    try:
        db.execute(
            "UPDATE graduated_params SET active = 0, rollback_reason = ? WHERE id = ?",
            (reason, param_id),
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to deactivate graduated param {param_id}: {e}")


def get_snapshots_for_param_review(param_id: int, activated_ts: str) -> dict:
    """Get fill stats before and after a graduated param was activated.

    Returns {"before": [slippage_bps, ...], "after": [slippage_bps, ...]}.
    The 'before' window matches the same (order_type, time_bucket, atr_bucket)
    from snapshots created before the param's activation timestamp.
    The 'after' window uses snapshots linked to this param_id or created after activation.
    """
    db = get_db()
    try:
        # Get the param's target buckets from its key
        param = db.execute(
            "SELECT param_key FROM graduated_params WHERE id = ?", (param_id,)
        ).fetchone()
        if not param:
            return {"before": [], "after": []}

        key = param["param_key"]
        parts = key.split(".")
        # Structured key: order_type.intent.time_bucket.atr_bucket
        if len(parts) >= 4:
            ot, intent, tb, ab = parts[0], parts[1], parts[2], parts[3]
        else:
            return {"before": [], "after": []}

        # Build WHERE clause for matching snapshots (before window)
        conditions = ["status = 'filled'", "slippage_bps IS NOT NULL"]
        params_list: list = []
        if ot != "all":
            conditions.append("order_type = ?")
            params_list.append(ot)
        if intent != "all":
            conditions.append("intent = ?")
            params_list.append(intent)
        if tb != "all":
            conditions.append("time_bucket = ?")
            params_list.append(tb)
        if ab != "all":
            conditions.append("atr_bucket = ?")
            params_list.append(ab)

        where = " AND ".join(conditions)

        # Before: matching snapshots before activation (same order type + buckets)
        before_rows = db.execute(
            f"SELECT slippage_bps FROM execution_snapshots WHERE {where} AND ts < ? ORDER BY id DESC LIMIT 50",
            (*params_list, activated_ts),
        ).fetchall()

        # After: only snapshots directly linked to this graduated param.
        # Bucket-only matching would introduce noise from unrelated orders
        # (e.g., exits or different order types in the same time/atr bucket).
        after_rows = db.execute(
            "SELECT slippage_bps FROM execution_snapshots "
            "WHERE graduated_param_id = ? AND status = 'filled' AND slippage_bps IS NOT NULL "
            "ORDER BY id DESC LIMIT 50",
            (param_id,),
        ).fetchall()

        return {
            "before": [abs(r["slippage_bps"]) for r in before_rows],
            "after": [abs(r["slippage_bps"]) for r in after_rows],
        }
    except Exception as e:
        logger.warning(f"Failed to get param review snapshots: {e}")
        return {"before": [], "after": []}


# ── Param key schema ────────────────────────────────────────────
# Structured format: {order_type}.{intent}.{time_bucket}.{atr_bucket}
# Each component must be from a known set or 'all' for wildcard.
_VALID_ORDER_TYPES = {
    "market", "limit", "stop_entry", "bracket", "trailing_stop",
    "oca_exit", "midprice", "adaptive", "vwap", "twap",
    "relative", "snap_mid", "moc", "moo", "loc", "loo", "all",
}
_VALID_INTENTS = {"entry", "exit", "stop", "all"}
_VALID_TIME_BUCKETS = {"open", "morning", "midday", "close", "extended", "all"}
_VALID_ATR_BUCKETS = {"low", "medium", "high", "all"}


def validate_param_key(key: str) -> str | None:
    """Validate a graduated param key matches the structured schema.

    Returns None if valid, or an error message if invalid.
    """
    parts = key.split(".")
    if len(parts) != 4:
        return f"Expected 4 dot-separated parts, got {len(parts)}: {key}"
    ot, intent, tb, ab = parts
    if ot not in _VALID_ORDER_TYPES:
        return f"Invalid order_type '{ot}' in key: {key}"
    if intent not in _VALID_INTENTS:
        return f"Invalid intent '{intent}' in key: {key}"
    if tb not in _VALID_TIME_BUCKETS:
        return f"Invalid time_bucket '{tb}' in key: {key}"
    if ab not in _VALID_ATR_BUCKETS:
        return f"Invalid atr_bucket '{ab}' in key: {key}"
    return None


def insert_graduated_param(
    param_key: str,
    param_value: str,
    previous_value: str | None,
    evidence_json: str,
    snapshots_analyzed: int,
    improvement_bps: float,
    p_value: float,
) -> int | None:
    """Insert a new graduated parameter override."""
    from datetime import datetime, timezone
    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO graduated_params
               (ts, param_key, param_value, previous_value, evidence_json,
                snapshots_analyzed, improvement_bps, p_value, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                datetime.now(timezone.utc).isoformat(),
                param_key, param_value, previous_value, evidence_json,
                snapshots_analyzed, improvement_bps, p_value,
            ),
        )
        db.commit()
        return cur.lastrowid
    except Exception as e:
        logger.warning(f"Failed to insert graduated param: {e}")
        return None


# Monotonic version counter — incremented on each calibration upsert.
# Consumers (simulator) compare against their cached version to detect staleness.
_calibration_version: int = 0


def get_calibration_version() -> int:
    """Return current calibration version (monotonic counter)."""
    return _calibration_version


def get_open_hypotheses(slot: int | None = None, limit: int = 10) -> list[dict]:
    """Return open trader hypotheses, optionally filtered by slot."""
    db = get_db()
    if slot is not None:
        rows = db.execute(
            """SELECT hypothesis_type, description, suggested_action, priority, related_slot
               FROM trader_hypotheses
               WHERE status = 'open' AND (related_slot = ? OR related_slot IS NULL)
               ORDER BY priority ASC, ts DESC LIMIT ?""",
            (slot, limit),
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT hypothesis_type, description, suggested_action, priority, related_slot
               FROM trader_hypotheses
               WHERE status = 'open'
               ORDER BY priority ASC, ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_hypothesis_incorporated(hypothesis_type: str, description: str) -> None:
    """Mark matching open hypotheses as incorporated."""
    db = get_db()
    db.execute(
        """UPDATE trader_hypotheses SET status = 'incorporated', kept = 1
           WHERE status = 'open' AND hypothesis_type = ? AND description = ?""",
        (hypothesis_type, description),
    )
    db.commit()


def upsert_calibrated_slippage(
    order_type: str,
    time_bucket: str,
    atr_bucket: str,
    median_bps: float,
    sample_count: int,
    p25_bps: float | None = None,
    p75_bps: float | None = None,
) -> None:
    """Upsert a calibrated slippage entry."""
    global _calibration_version
    from datetime import datetime, timezone
    db = get_db()
    try:
        db.execute(
            """INSERT INTO calibrated_slippage
               (ts, order_type, time_bucket, atr_bucket, median_slippage_bps,
                sample_count, p25_bps, p75_bps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(order_type, time_bucket, atr_bucket) DO UPDATE
                 SET ts = excluded.ts,
                     median_slippage_bps = excluded.median_slippage_bps,
                     sample_count = excluded.sample_count,
                     p25_bps = excluded.p25_bps,
                     p75_bps = excluded.p75_bps""",
            (
                datetime.now(timezone.utc).isoformat(),
                order_type, time_bucket, atr_bucket,
                median_bps, sample_count, p25_bps, p75_bps,
            ),
        )
        db.commit()
        _calibration_version += 1
    except Exception as e:
        logger.warning(f"Failed to upsert calibrated slippage: {e}")


def get_calibrated_slippage() -> dict[tuple[str, str, str], float]:
    """Get all calibrated slippage values as {(order_type, time_bucket, atr_bucket): median_bps}."""
    db = get_db()
    rows = db.execute(
        "SELECT order_type, time_bucket, atr_bucket, median_slippage_bps FROM calibrated_slippage"
    ).fetchall()
    return {
        (r["order_type"], r["time_bucket"], r["atr_bucket"]): r["median_slippage_bps"]
        for r in rows
    }
