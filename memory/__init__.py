"""
Memory Layer — SQLite persistence for research + trading.

Single file (memory/abc.db), WAL mode. The signal-combination engine owns
the live tables (signal_scores, signal_weights, composite_scores,
template_recommendations, template_performance). Execution writes to
trades + trade_feedback; the trader and research loops both read the
signal-engine tables. The legacy slot-system tables (live_signals,
strategies, slot_environment_scores) have been retired and are no
longer created on fresh DBs.
"""

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
from psycopg import sql as psql_sql
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

_PG_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _apply_database_app_role(raw_conn) -> None:
    """Optional SET ROLE for shared schema ownership (see infra/postgres/init/02)."""
    role_raw = os.getenv("DATABASE_APP_ROLE", "").strip()
    if not role_raw:
        return
    if not _PG_IDENTIFIER_RE.fullmatch(role_raw):
        raise RuntimeError(
            f"Invalid DATABASE_APP_ROLE {role_raw!r} "
            "(use letters, digits, underscore; e.g. abc_app)"
        )
    try:
        with raw_conn.cursor() as cur:
            cur.execute(psql_sql.SQL("SET ROLE {}").format(psql_sql.Identifier(role_raw)))
        raw_conn.commit()
    except Exception as e:
        raw_conn.rollback()
        raise RuntimeError(
            f"SET ROLE {role_raw!r} failed (is the login user a member of that role?): {e}"
        ) from e
    logger.debug("Session using DATABASE_APP_ROLE=%s", role_raw)

_DB_PATH = Path(__file__).parent / "abc.db"
_connections_by_thread: dict[int, "_CompatConnection"] = {}


class _CompatRow(dict):
    """Row wrapper that supports both dict and positional indexing."""

    def __init__(self, row_dict: dict):
        super().__init__(row_dict)
        self._ordered_values = list(row_dict.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ordered_values[key]
        return super().__getitem__(key)

    def __iter__(self):
        # Match sqlite3.Row iteration semantics (values in column order),
        # so existing tuple-unpacking code keeps working.
        return iter(self._ordered_values)

    def __len__(self):
        return len(self._ordered_values)


class _CompatCursor:
    def __init__(self, cursor, lastrowid=None):
        self._cursor = cursor
        self.lastrowid = lastrowid if lastrowid is not None else getattr(cursor, "lastrowid", None)
        self.rowcount = getattr(cursor, "rowcount", -1)

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return _CompatRow(dict(row))

    def fetchall(self):
        rows = self._cursor.fetchall()
        return [_CompatRow(dict(r)) for r in rows]

    def __getattr__(self, name):
        # Preserve compatibility for cursor attrs used by existing code.
        return getattr(self._cursor, name)


class _CompatConnection:
    def __init__(self, conn):
        self._conn = conn

    def _adapt_sql(self, sql: str) -> str:
        sql = self._rewrite_insert_or_replace(sql)
        sql = sql.replace("?", "%s")
        sql = re.sub(
            r"datetime\('now',\s*'-([0-9]+)\s+days'\)",
            r"(NOW() - INTERVAL '\1 days')",
            sql,
            flags=re.IGNORECASE,
        )
        return sql

    def _rewrite_insert_or_replace(self, sql: str) -> str:
        conflict_keys = {
            "signal_scores": ["signal_name", "symbol", "ts"],
            "signal_weights": ["signal_name"],
            "signal_returns": ["signal_name", "symbol", "ts"],
            "composite_scores": ["symbol", "ts"],
            "template_performance": ["template_name", "regime_key", "composite_bucket"],
            "template_boundaries": ["template_name", "param_name"],
            "template_recommendations": ["symbol", "ts"],
            "iv_history": ["symbol", "ts"],
            "signal_symbol_ic": ["signal_name", "symbol", "horizon_bars"],
            "latest_quotes": ["symbol"],
            "research_config": ["key"],
            "calibrated_slippage": ["order_type", "time_bucket", "atr_bucket"],
        }
        match = re.match(
            r"^\s*INSERT\s+OR\s+REPLACE\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*"
            r"\(([^)]+)\)\s*VALUES\s*\((.*)\)\s*;?\s*$",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return sql
        table = match.group(1)
        columns_raw = match.group(2)
        values_expr = match.group(3)
        key_cols = conflict_keys.get(table.lower())
        if not key_cols:
            return sql
        columns = [c.strip() for c in columns_raw.split(",")]
        update_cols = [c for c in columns if c not in key_cols]
        if update_cols:
            updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
            action = f"DO UPDATE SET {updates}"
        else:
            action = "DO NOTHING"
        keys = ", ".join(key_cols)
        return (
            f"INSERT INTO {table} ({columns_raw}) VALUES ({values_expr}) "
            f"ON CONFLICT ({keys}) {action}"
        )

    def _adapt_schema_sql(self, sql: str) -> str:
        # SQLite -> PostgreSQL schema compatibility transforms.
        sql = re.sub(
            r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
            "BIGSERIAL PRIMARY KEY",
            sql,
            flags=re.IGNORECASE,
        )
        # Legacy FKs reference old tables that may not exist in fresh DBs.
        # Keep FK clauses, but ensure placeholder parent tables exist.
        bootstrap = (
            "CREATE TABLE IF NOT EXISTS strategies (id BIGINT PRIMARY KEY);\n"
            "CREATE TABLE IF NOT EXISTS signals (id BIGINT PRIMARY KEY);\n"
        )
        return bootstrap + sql

    def execute(self, sql: str, params=None):
        cur = self._conn.cursor(row_factory=dict_row)
        adapted = self._adapt_sql(sql)
        try:
            if params is None:
                cur.execute(adapted)
            else:
                cur.execute(adapted, params)
        except Exception:
            # Clear failed transaction state so long-running loops can recover.
            self._conn.rollback()
            raise
        return _CompatCursor(cur)

    def executemany(self, sql: str, seq_of_parameters):
        """Batch execute; mirrors sqlite3.Connection.executemany."""
        adapted = self._adapt_sql(sql)
        with self._conn.cursor() as cur:
            try:
                cur.executemany(adapted, seq_of_parameters)
            except Exception:
                # psycopg pipeline mode can leave the transaction INERROR.
                self._conn.rollback()
                raise

    def executescript(self, sql: str):
        script = self._adapt_schema_sql(sql)
        statements = self._split_sql_statements(script)
        with self._conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def _split_sql_statements(self, script: str) -> list[str]:
        statements: list[str] = []
        buf: list[str] = []
        in_single = False
        in_double = False
        in_line_comment = False
        in_block_comment = False
        i = 0
        while i < len(script):
            ch = script[i]
            nxt = script[i + 1] if i + 1 < len(script) else ""
            if in_line_comment:
                buf.append(ch)
                if ch == "\n":
                    in_line_comment = False
                i += 1
                continue
            if in_block_comment:
                buf.append(ch)
                if ch == "*" and nxt == "/":
                    buf.append(nxt)
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if not in_single and not in_double and ch == "-" and nxt == "-":
                in_line_comment = True
                buf.append(ch)
                buf.append(nxt)
                i += 2
                continue
            if not in_single and not in_double and ch == "/" and nxt == "*":
                in_block_comment = True
                buf.append(ch)
                buf.append(nxt)
                i += 2
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                buf.append(ch)
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                buf.append(ch)
                i += 1
                continue
            if ch == ";" and not in_single and not in_double:
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            statements.append(tail)
        return statements


def _resolve_postgres_dsn() -> str:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if dsn:
        return dsn
    host = os.getenv("PGHOST", "").strip()
    port = os.getenv("PGPORT", "5432").strip()
    dbname = os.getenv("PGDATABASE", "").strip()
    user = os.getenv("PGUSER", "").strip()
    password = os.getenv("PGPASSWORD", "").strip()
    if all([host, dbname, user, password]):
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    raise RuntimeError(
        "PostgreSQL is required. Set DATABASE_URL or PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
    )


def _ensure_column(conn, table: str, col: str, col_type: str) -> None:
    """Helper to idempotently add a column if it doesn't exist (simplifies migrations)."""
    table_exists = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = ?",
        (table,),
    ).fetchone()
    if not table_exists:
        return
    row = conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = ? AND column_name = ?",
        (table, col),
    ).fetchone()
    if row:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    except Exception:
        pass


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = ?",
        (table,),
    ).fetchone()
    return bool(row)


def init_db():
    """Initialize PostgreSQL connection and schema, return connection."""
    tid = threading.get_ident()
    existing = _connections_by_thread.get(tid)
    if existing is not None:
        return existing

    raw_conn = psycopg.connect(_resolve_postgres_dsn(), autocommit=False)
    _apply_database_app_role(raw_conn)
    conn = _CompatConnection(raw_conn)

    conn.executescript("""
        -- Closed-trade ledger (the legacy `signals` FK is kept for back-compat;
        --  the table is no longer created so the FK is unenforced on fresh DBs).
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            pnl REAL,
            signal_id INTEGER,
            held_minutes INTEGER
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
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        );

        CREATE INDEX IF NOT EXISTS idx_env_snapshots_ts ON environment_snapshots(ts);
        CREATE INDEX IF NOT EXISTS idx_trade_feedback_ts ON trade_feedback(ts);
        CREATE INDEX IF NOT EXISTS idx_trade_feedback_symbol_ts ON trade_feedback(symbol, ts);
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

        -- Daily IV snapshots used to compute trailing-percentile IV rank.
        CREATE TABLE IF NOT EXISTS iv_history (
            symbol TEXT NOT NULL,
            ts REAL NOT NULL,
            iv_current REAL NOT NULL,
            source TEXT,
            PRIMARY KEY (symbol, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_signal_scores_ts ON signal_scores(ts);
        CREATE INDEX IF NOT EXISTS idx_signal_returns_signal ON signal_returns(signal_name, ts);
        CREATE INDEX IF NOT EXISTS idx_signal_returns_signal_symbol ON signal_returns(signal_name, symbol, ts);
        CREATE INDEX IF NOT EXISTS idx_composite_scores_ts ON composite_scores(ts);

        -- Per-(signal, symbol, horizon) Information Coefficient.
        -- Computed by signals/per_symbol_ic.py from rows already present
        -- in signal_returns (so it's pure derived state — safe to drop and
        -- recompute).  Used by the cognitive layer to scale a signal's
        -- weight per symbol when its per-symbol reliability diverges from
        -- its global IC.
        CREATE TABLE IF NOT EXISTS signal_symbol_ic (
            signal_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            horizon_bars INTEGER NOT NULL,
            ic REAL NOT NULL,
            t_stat REAL NOT NULL,
            n_obs INTEGER NOT NULL,
            ic_neg_streak INTEGER NOT NULL DEFAULT 0,
            last_updated_ts REAL NOT NULL,
            PRIMARY KEY (signal_name, symbol, horizon_bars)
        );
        CREATE INDEX IF NOT EXISTS idx_sym_ic_symbol ON signal_symbol_ic(symbol);
        CREATE INDEX IF NOT EXISTS idx_sym_ic_signal ON signal_symbol_ic(signal_name);

        -- Trader's working memory: short-term, structured monologue.
        -- Holds *interpretations* (theses, verdicts, watch-fors), not
        -- *observations* (prices, quotes).  Persisted so the trader
        -- survives a restart with today's context intact; entries auto-
        -- expire via expires_ts and the in-process curator drops them at
        -- the top of each cycle.  See memory/working_memory.py.
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT NOT NULL,
            entry_text TEXT NOT NULL,
            created_ts REAL NOT NULL,
            expires_ts REAL NOT NULL,
            metadata_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_wm_section_expires ON working_memory(section, expires_ts);
        CREATE INDEX IF NOT EXISTS idx_wm_created ON working_memory(created_ts);

        -- Attention triggers — structured "watching_for" entries the
        -- agent registers (directly via metadata or parsed from a
        -- watching_for entry).  Evaluated each scorer round; when a
        -- trigger fires we record fired_ts/fire_value and the cycle
        -- prompt renders an ATTENTION block.  Cap: 10 active rows
        -- (oldest active evicted on insert).
        -- See docs/PLAN_COGNITIVE_ARCHITECTURE.md §4 and
        -- core/runtime/attention.py.
        CREATE TABLE IF NOT EXISTS attention_triggers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            condition TEXT NOT NULL,
            threshold REAL,
            confirm_with_json TEXT,
            source_entry_id INTEGER,
            source_text TEXT,
            created_ts REAL NOT NULL,
            fired_ts REAL,
            fire_value REAL,
            fire_note TEXT,
            last_value REAL,
            state TEXT NOT NULL DEFAULT 'active'
        );
        CREATE INDEX IF NOT EXISTS idx_attn_state_created ON attention_triggers(state, created_ts);
        CREATE INDEX IF NOT EXISTS idx_attn_symbol ON attention_triggers(symbol);
        CREATE INDEX IF NOT EXISTS idx_attn_source_entry ON attention_triggers(source_entry_id);

        CREATE INDEX IF NOT EXISTS idx_template_recs_ts ON template_recommendations(ts);
        CREATE INDEX IF NOT EXISTS idx_iv_history_symbol_ts ON iv_history(symbol, ts);

        -- Latest real-time stock quote per symbol (single row per symbol;
        -- written by the IBKRQuoteSource on every successful tick read).
        -- Exists so the future research daemon can read what the trader
        -- saw without holding its own IBKR streaming subscription.
        CREATE TABLE IF NOT EXISTS latest_quotes (
            symbol TEXT PRIMARY KEY,
            last REAL,
            bid REAL,
            ask REAL,
            volume INTEGER,
            high REAL,
            low REAL,
            ts REAL NOT NULL,
            source TEXT NOT NULL
        );
    """)

    # ── Migrations for existing DBs ─────────────────────────────
    # Idempotent column adds. Legacy `live_signals.slot` migration was removed
    # along with the table itself.
    _ensure_column(conn, "trade_feedback", "template_name", "TEXT")

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

    # ── Schema-version migrations ────────────────────────────────
    # Any change that invalidates historical data (not just column adds)
    # should be expressed as a migration below, bumping SCHEMA_VERSION.
    # Existing DBs run the missing migrations in order on next start-up,
    # so every live run always works on data that matches the code.
    _apply_schema_migrations(conn)

    conn.commit()
    _connections_by_thread[tid] = conn
    logger.info("Memory DB initialized on PostgreSQL")
    return conn


# ── Schema migrations ────────────────────────────────────────────
# Bump SCHEMA_VERSION when you add an entry to _MIGRATIONS.  Each
# migration is a ``(version, description, fn)`` triple where ``fn``
# takes a connection and does whatever work is needed.  Migrations run
# inside the init_db transaction so a failure leaves the DB unchanged.

SCHEMA_VERSION = 2


def _migration_v2_reset_forward_return_pipeline(conn) -> None:
    """Phase D (2026-04-20) changed signal_returns keying from score_ts
    to entry_bar_ts.  Existing rows mix the two semantics and re-inflate
    IC; drop them.  Also drop signal_scores so the 30d TTL doesn't have
    to grind through pre-fix timestamps.  composite_scores is derived
    and safe to clear."""
    for tbl in ("signal_returns", "signal_scores", "composite_scores"):
        if _table_exists(conn, tbl):
            conn.execute(f"DELETE FROM {tbl}")


_MIGRATIONS: list[tuple[int, str, callable]] = [
    (2, "reset forward-return pipeline for per-signal entry_bar_ts keying",
     _migration_v2_reset_forward_return_pipeline),
]


def _apply_schema_migrations(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "version INTEGER PRIMARY KEY, applied_ts REAL NOT NULL, description TEXT)"
    )
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current = int(row[0]) if row and row[0] is not None else 1  # legacy DBs start at 1

    # Forward-compatibility guard: a DB stamped with a NEWER schema than
    # this build understands almost certainly means the user rolled back
    # the code without rolling back the DB.  Fail fast with a clear path
    # forward instead of silently running with the wrong assumptions.
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {current} is newer than this build "
            f"supports (max={SCHEMA_VERSION}). The database was likely "
            f"written by a newer build of the bot. Either upgrade the code "
            f"or restore a DB backup that matches this version."
        )

    if current >= SCHEMA_VERSION:
        return

    import time as _time
    for version, desc, fn in _MIGRATIONS:
        if version <= current:
            continue
        logger.warning("Applying schema migration v%d: %s", version, desc)
        fn(conn)
        conn.execute(
            "INSERT INTO schema_version (version, applied_ts, description) VALUES (?, ?, ?)",
            (version, _time.time(), desc),
        )
    logger.info("Schema at version %d (was %d)", SCHEMA_VERSION, current)


def get_db():
    """Get the database connection, initializing if needed."""
    tid = threading.get_ident()
    if tid not in _connections_by_thread:
        return init_db()
    return _connections_by_thread[tid]


def get_schema_version(conn=None) -> int:
    """Return the current schema version stamped in the DB (1 for legacy)."""
    db = conn if conn is not None else get_db()
    try:
        row = db.execute("SELECT MAX(version) FROM schema_version").fetchone()
    except Exception:
        return 1
    return int(row[0]) if row and row[0] is not None else 1


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
    rows to template_recommendations by symbol + time proximity and writes
    trade_feedback rows tracking execution gaps.

    Returns the trade row id, or None on failure.
    """
    try:
        db = get_db()
        cur = db.execute(
            """INSERT INTO trades (ts, symbol, side, pnl, signal_id, held_minutes)
               VALUES (?, ?, ?, ?, ?, ?)
               RETURNING id""",
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
        row = cur.fetchone()
        trade_id = int(row["id"]) if row else None
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
    """Match a closed trade to the most recent template_recommendation for the
    symbol and write a trade_feedback row.

    The legacy `live_signals` ⨝ `signals` join was removed when those tables
    were retired; we now match against the signal-engine's recommendations.
    """
    trade = db.execute(
        "SELECT id, symbol, side, pnl, ts FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    if not trade:
        return

    # Convert the trade's ISO timestamp to a unix epoch float so we can
    # compare against the REAL `ts` column on template_recommendations.
    try:
        trade_epoch = datetime.fromisoformat(trade["ts"]).timestamp()
    except Exception:
        trade_epoch = None

    rec = None
    if trade_epoch is not None:
        # Within a 7-day window pick the closest recommendation by ts.
        rec = db.execute(
            """SELECT rowid AS id, template_name, direction, entry_price,
                      composite_score, ts
               FROM template_recommendations
               WHERE symbol = ?
                 AND ABS(ts - ?) < 7.0 * 86400
               ORDER BY ABS(ts - ?) ASC
               LIMIT 1""",
            (symbol.upper(), trade_epoch, trade_epoch),
        ).fetchone()

    if not rec:
        logger.debug(f"No matching recommendation for trade {trade_id} ({symbol})")
        return

    sim_return = rec["composite_score"] or 0.0
    actual_pnl = trade["pnl"] or 0
    entry_price = rec["entry_price"]

    if entry_price and entry_price > 0:
        actual_return_pct = (actual_pnl / entry_price) * 100
        execution_gap = actual_return_pct - sim_return
    else:
        execution_gap = None

    db.execute(
        """INSERT INTO trade_feedback
           (ts, trade_id, signal_id, slot, simulated_return,
            actual_pnl, execution_gap, symbol, template_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            trade_id,
            rec["id"],
            None,                  # legacy slot column — unused for new rows
            sim_return,
            actual_pnl,
            execution_gap,
            symbol.upper(),
            rec["template_name"],
        ),
    )
    db.commit()
    gap_str = f"{execution_gap:.2f}%" if execution_gap is not None else "unknown"
    logger.info(
        f"Trade feedback: trade {trade_id} -> template {rec['template_name']} "
        f"gap={gap_str}"
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


# ── Accessors for the pending* dicts ────────────────────────────
# These exist so callers (tools/tools_executor.py, execution/ibkr_orders.py,
# tests) don't reach into private module-level dicts.  The dicts remain
# private; new code should go through these functions exclusively.


def set_pending_graduated_param(symbol: str, param_id: int) -> None:
    """Record a graduated_param_id for a symbol, to be consumed at the next
    ``insert_execution_snapshot(symbol)`` call."""
    _pending_graduated_params[symbol] = param_id


def get_pending_graduated_param(symbol: str) -> int | None:
    """Peek at the pending graduated_param_id without consuming it."""
    return _pending_graduated_params.get(symbol)


def set_pending_order_context(symbol: str, context: dict) -> None:
    """Record plan-time context (intent, atr_pct, ...) for a symbol, to be
    consumed at the next ``insert_execution_snapshot(symbol)`` call."""
    _pending_order_context[symbol] = context


def get_pending_order_context(symbol: str) -> dict:
    """Peek at the pending order context (returns ``{}`` if absent)."""
    return _pending_order_context.get(symbol, {})


def reset_state(db_path=None) -> None:
    """Reset all module-level mutable state — primarily for tests.

    Single chokepoint that test fixtures (``tests/conftest.py``) and
    future DI scaffolding can call instead of poking five different
    module attributes from outside. Closes any open connection, clears
    the two pending lookup dicts, resets the calibration version, and
    optionally repoints the DB path.
    """
    global _DB_PATH, _calibration_version
    if db_path is not None:
        _DB_PATH = db_path
    for conn in list(_connections_by_thread.values()):
        try:
            conn.close()
        except Exception:  # pragma: no cover - defensive
            pass
    _connections_by_thread.clear()
    _calibration_version = 0
    _pending_graduated_params.clear()
    _pending_order_context.clear()


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
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted')
               RETURNING id""",
            (
                ts, symbol, side, quantity, order_type, intent,
                bid, ask, mid, spread,
                volume, atr,
                _time_bucket(ts), _atr_bucket(atr_pct),
                graduated_param_id,
            ),
        )
        db.commit()
        row = cur.fetchone()
        return int(row["id"]) if row else None
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
    # Implementation lives in memory.repos.config_repo (PR12 split).
    from memory.repos.config_repo import get_graduated_params as _impl
    return _impl(active_only=active_only)


def deactivate_graduated_param(param_id: int, reason: str) -> None:
    # Implementation lives in memory.repos.config_repo (PR12 split).
    from memory.repos.config_repo import deactivate_graduated_param as _impl
    return _impl(param_id, reason)


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
# Constants and validation moved to memory.repos.config_repo (PR12).
# `validate_param_key` and `insert_graduated_param` remain importable
# from `memory` via thin shims below.

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


# Monotonic version counter — incremented on each calibration upsert.
# Consumers (simulator) compare against their cached version to detect staleness.
_calibration_version: int = 0


def get_calibration_version() -> int:
    # Implementation lives in memory.repos.config_repo (PR12 split).
    # The underlying counter (`_calibration_version`) remains owned here
    # because `upsert_calibrated_slippage` (execution domain) mutates it.
    from memory.repos.config_repo import get_calibration_version as _impl
    return _impl()


def get_open_hypotheses(slot: int | None = None, limit: int = 10) -> list[dict]:
    # Implementation lives in memory.repos.feedback_repo (PR8 split).
    from memory.repos.feedback_repo import get_open_hypotheses as _impl
    return _impl(slot=slot, limit=limit)


def mark_hypothesis_incorporated(hypothesis_type: str, description: str) -> None:
    # Implementation lives in memory.repos.feedback_repo (PR8 split).
    from memory.repos.feedback_repo import mark_hypothesis_incorporated as _impl
    return _impl(hypothesis_type, description)


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
    # Implementation lives in memory.repos.execution_repo (PR13 split).
    from memory.repos.execution_repo import get_calibrated_slippage as _impl
    return _impl()


# -- Back-compat re-exports (PR13 split) -------------------------
# _IV_HISTORY_MIN_SAMPLES is read by tests via `from memory import`.
# Imported at the bottom of this module (after writer functions are
# defined) to avoid circular-import issues with `execution_repo`,
# which re-exports those writers back from `memory`.
from memory.repos.execution_repo import _IV_HISTORY_MIN_SAMPLES  # noqa: E402,F401

