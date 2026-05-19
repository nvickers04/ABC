"""PostgreSQL connection, schema DDL, and versioned migrations."""

from __future__ import annotations

import logging
import os
import re
import threading
import time as _time
from pathlib import Path

import psycopg
from psycopg import sql as psql_sql
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "abc.db"
_connections_by_thread: dict[int, "_CompatConnection"] = {}

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

        -- QualityMatrix provenance (tool usage + decision snapshots)
        -- Additive, decision-scoped. Populated by QualityMatrixService hooks.
        CREATE TABLE IF NOT EXISTS tool_usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cycle_id INTEGER DEFAULT 0,
            tool_name TEXT NOT NULL,
            symbol TEXT,
            success INTEGER DEFAULT 1,
            latency_ms REAL DEFAULT 0,
            decision_context TEXT            -- JSON or free text (cycle_id is advisory)
        );

        CREATE TABLE IF NOT EXISTS decision_provenance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cycle_id INTEGER DEFAULT 0,
            decision_type TEXT NOT NULL,     -- cycle_decision | entry_idea | sizing | review | done
            symbol TEXT,
            tools_json TEXT,                 -- JSON array of ToolUsageRecord summaries
            quality_state_json TEXT,         -- snapshot of QualityMatrix state at decision
            context_quality TEXT,
            outcome TEXT,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tool_usage_ts ON tool_usage_log(ts);
        CREATE INDEX IF NOT EXISTS idx_tool_usage_name ON tool_usage_log(tool_name, ts);
        CREATE INDEX IF NOT EXISTS idx_tool_usage_symbol ON tool_usage_log(symbol, ts);
        CREATE INDEX IF NOT EXISTS idx_provenance_ts ON decision_provenance(ts);
        CREATE INDEX IF NOT EXISTS idx_provenance_cycle ON decision_provenance(cycle_id);
        CREATE INDEX IF NOT EXISTS idx_provenance_type ON decision_provenance(decision_type);
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
            track_record_json TEXT,
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
        -- See core/runtime/attention.py.
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
    _ensure_column(conn, "template_recommendations", "track_record_json", "TEXT")

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

SCHEMA_VERSION = 3


def _migration_v2_reset_forward_return_pipeline(conn) -> None:
    """Phase D (2026-04-20) changed signal_returns keying from score_ts
    to entry_bar_ts.  Existing rows mix the two semantics and re-inflate
    IC; drop them.  Also drop signal_scores so the 30d TTL doesn't have
    to grind through pre-fix timestamps.  composite_scores is derived
    and safe to clear."""
    for tbl in ("signal_returns", "signal_scores", "composite_scores"):
        if _table_exists(conn, tbl):
            conn.execute(f"DELETE FROM {tbl}")


def _migration_v3_add_quality_provenance_tables(conn) -> None:
    """Schema v3: QualityMatrix provenance tables (version stamp; tables also in init DDL)."""
    # Tables are created idempotently in the main init script; migration just records
    # that v3 was applied. No data transform needed.
    pass


_MIGRATIONS: list[tuple[int, str, callable]] = [
    (2, "reset forward-return pipeline for per-signal entry_bar_ts keying",
     _migration_v2_reset_forward_return_pipeline),
    (3, "add tool_usage_log + decision_provenance tables for QualityMatrix",
     _migration_v3_add_quality_provenance_tables),
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


def reset_connections() -> None:
    """Close all thread-local connections (used by ``memory.reset_state``)."""
    for conn in list(_connections_by_thread.values()):
        try:
            conn.close()
        except Exception:  # pragma: no cover - defensive
            pass
    _connections_by_thread.clear()


__all__ = [
    "SCHEMA_VERSION",
    "_DB_PATH",
    "get_db",
    "get_schema_version",
    "init_db",
    "reset_connections",
]
