"""
Research Agent — Karpathy autoresearch pattern for day trading.

Architecture:
  - NUM_SLOTS independent strategy slots, each evolved concurrently
  - All slots run in parallel via asyncio.gather (LLM calls are async)
  - Strategy evals run in thread pool (CPU-bound pandas/numpy)
  - Meta-learning selector assesses market environment, correlates slot
    performance with environment, and allocates slots intelligently
  - Real trade feedback loop compares simulated vs actual P&L
  - Git commits serialized via asyncio.Lock
"""

import ast
import asyncio
import json
import logging
import random
import re
import subprocess
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

import pandas as pd

from core.grok_llm import get_grok_llm, REASONING_MODEL
from data.cost_tracker import get_cost_tracker
from data.data_provider import get_data_provider
from memory import get_db
from research.config import (
    ANALYSIS_PROMPT_TEMPLATE,
    CANDLE_RESOLUTION,
    EVAL_DAYS_BACK,
    MAX_SELECTOR_REPLACEMENTS,
    NUM_SLOTS,
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_UNIVERSE,
    SANDBOX_ALLOWED_IMPORTS,
    SANDBOX_BLOCKED_CALLS,
    SELECTOR_EVERY_N_ROUNDS,
    SELECTOR_PROMPT,
    SIGNAL_SCHEMA,
)
from research.environment import compute_environment, format_environment_for_prompt
from research.simulator import compute_expectancy, simulate

logger = logging.getLogger(__name__)

_SLOTS_DIR = Path(__file__).parent / "slots"
_GIT_LOCK = asyncio.Lock()
_LLM_SEMAPHORE: asyncio.Semaphore | None = None  # initialized in run_research


def _strategy_path(slot: int) -> Path:
    """Return the file path for a given slot's strategy."""
    return _SLOTS_DIR / f"strategy_{slot:02d}.py"


# ═══════════════════════════════════════════════════════════════
# CANDLE FETCHING
# ═══════════════════════════════════════════════════════════════

def _candles_to_df(candles) -> pd.DataFrame:
    """Convert a data_provider Candles dataclass to a DataFrame."""
    return pd.DataFrame({
        "ts": candles.timestamps,
        "open": candles.open,
        "high": candles.high,
        "low": candles.low,
        "close": candles.close,
        "volume": candles.volume,
    })


def _split_by_trading_day(
    df: pd.DataFrame, *, exclude_today: bool = False,
) -> dict[str, pd.DataFrame]:
    """Split a multi-day candle DataFrame into per-day DataFrames.

    Returns {date_str: df} where date_str is YYYY-MM-DD.
    If exclude_today is True, today's (potentially incomplete) data is dropped.
    """
    if df.empty:
        return {}

    # Convert unix timestamps to proper ET using zoneinfo (handles EST/EDT)
    df = df.copy()
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["dt_et"] = df["dt"].dt.tz_convert(_ET)
    df["trade_date"] = df["dt_et"].dt.date.astype(str)

    today_str = date.today().isoformat()

    days = {}
    for d, group in df.groupby("trade_date"):
        if exclude_today and d == today_str:
            continue
        day_df = group.drop(columns=["dt", "dt_et", "trade_date"]).reset_index(drop=True)
        if len(day_df) >= 30:  # need at least 30 bars for a meaningful test
            days[d] = day_df
    return days


def _fetch_universe_candles(
    days_back: int, *, exclude_today: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch candles for all symbols, split by trading day.

    Returns: {symbol: {date_str: df}}
    """
    provider = get_data_provider()
    universe: dict[str, dict[str, pd.DataFrame]] = {}

    # Use explicit date range so intraday resolution gets full days, not N bars
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=days_back + 4)).isoformat()  # +4 for weekends

    for sym in RESEARCH_UNIVERSE:
        try:
            raw = provider.get_candles(
                sym, resolution=CANDLE_RESOLUTION,
                from_date=from_date, to_date=to_date,
            )
            if raw and len(raw) > 0:
                df = _candles_to_df(raw)
                days = _split_by_trading_day(df, exclude_today=exclude_today)
                if days:
                    universe[sym] = days
        except Exception as e:
            logger.warning(f"Failed to fetch candles for {sym}: {e}")

    return universe


def _train_test_split(
    universe: dict[str, dict[str, pd.DataFrame]],
    train_ratio: float = 0.7,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]]]:
    """Split universe into train/test by date.  Chronological — no leakage."""
    # Collect all unique dates
    all_dates = sorted({d for days in universe.values() for d in days})
    if len(all_dates) < 3:
        # Not enough days to split — use everything for both
        return universe, universe

    split_idx = max(1, int(len(all_dates) * train_ratio))
    train_dates = set(all_dates[:split_idx])
    test_dates = set(all_dates[split_idx:])

    train: dict[str, dict[str, pd.DataFrame]] = {}
    test: dict[str, dict[str, pd.DataFrame]] = {}
    for sym, days in universe.items():
        tr = {d: df for d, df in days.items() if d in train_dates}
        te = {d: df for d, df in days.items() if d in test_dates}
        if tr:
            train[sym] = tr
        if te:
            test[sym] = te

    return train, test


# ═══════════════════════════════════════════════════════════════
# STRATEGY SANDBOX
# ═══════════════════════════════════════════════════════════════

def _validate_strategy_source(source: str) -> str | None:
    """Validate strategy source code. Returns error message or None if safe."""
    # Must parse
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    # Must have scan() function
    func_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    if "scan" not in func_names:
        return "Missing scan() function"

    # Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in SANDBOX_ALLOWED_IMPORTS:
                    return f"Blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in SANDBOX_ALLOWED_IMPORTS:
                    return f"Blocked import: {node.module}"

    # Check for blocked calls (os.system, subprocess, etc.)
    source_lower = source.lower()
    for blocked in SANDBOX_BLOCKED_CALLS:
        if f"{blocked}." in source_lower or f"import {blocked}" in source_lower:
            return f"Blocked module usage: {blocked}"

    return None


def _load_strategy_module(source: str):
    """Load strategy source into a module object and return it."""
    import types
    mod = types.ModuleType("strategy_sandbox")
    # Provide allowed imports in the module namespace
    import pandas as _pd
    import numpy as _np
    import math as _math
    import statistics as _statistics
    mod.__dict__["pd"] = _pd
    mod.__dict__["pandas"] = _pd
    mod.__dict__["np"] = _np
    mod.__dict__["numpy"] = _np
    mod.__dict__["math"] = _math
    mod.__dict__["statistics"] = _statistics
    exec(compile(source, "<strategy>", "exec"), mod.__dict__)  # noqa: S102 — sandboxed
    return mod


def _execute_strategy(source: str, candles: pd.DataFrame, symbol: str) -> list[dict]:
    """Execute strategy.scan() in sandbox. Returns signals or empty list."""
    try:
        mod = _load_strategy_module(source)
        scan_fn = getattr(mod, "scan", None)
        if scan_fn is None:
            return []
        signals = scan_fn(candles, symbol)
        if not isinstance(signals, list):
            return []
        return signals
    except Exception as e:
        logger.warning(f"Strategy execution error for {symbol}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def _deduplicate_signals(signals: list[dict]) -> list[dict]:
    """Remove overlapping signals for the same symbol.

    After a signal fires at bar N, skip any signals within the next
    max_hold_bars bars (cooldown period) to avoid correlated trades.
    """
    if not signals:
        return []
    # Sort by entry_bar
    sorted_sigs = sorted(signals, key=lambda s: s.get("entry_bar", 0))
    result = []
    next_allowed_bar = -1
    for sig in sorted_sigs:
        bar = sig.get("entry_bar", 0)
        if bar >= next_allowed_bar:
            result.append(sig)
            next_allowed_bar = bar + sig.get("max_hold_bars", 60)
    return result


def _evaluate_strategy(
    source: str,
    universe: dict[str, dict[str, pd.DataFrame]],
) -> tuple[dict, list[dict], list[dict]]:
    """Evaluate strategy across all symbols and trading days.

    Returns:
        (aggregate_stats, per_day_results, all_sim_results)
    """
    all_results: list[dict] = []
    per_day: list[dict] = []

    for sym, days in universe.items():
        for day_str, day_df in days.items():
            signals = _execute_strategy(source, day_df, sym)
            if not signals:
                continue
            # Deduplicate overlapping signals per symbol/day
            signals = _deduplicate_signals(signals)
            sim_results = simulate(signals, day_df)
            if not sim_results:
                continue

            # Tag with symbol and date
            for r in sim_results:
                r["symbol"] = sym
                r["eval_date"] = day_str
            all_results.extend(sim_results)

            day_stats = compute_expectancy(sim_results)
            day_stats["symbol"] = sym
            day_stats["eval_date"] = day_str
            per_day.append(day_stats)

    aggregate = compute_expectancy(all_results)
    return aggregate, per_day, all_results


# ═══════════════════════════════════════════════════════════════
# LLM INTERACTION
# ═══════════════════════════════════════════════════════════════

async def _llm_analyze(
    source: str,
    aggregate: dict,
    per_day: list[dict],
    all_results: list[dict],
    *,
    environment_text: str = "",
) -> str:
    """Ask the LLM to analyze backtest results. Returns analysis text."""
    winners = [r for r in all_results if r.get("return_pct", 0) > 0]
    losers = [r for r in all_results if r.get("return_pct", 0) < 0]
    timed = [r for r in all_results if r.get("timed_out")]
    # Random sample for diverse symbol/date coverage
    winners = random.sample(winners, min(10, len(winners))) if winners else []
    losers = random.sample(losers, min(10, len(losers))) if losers else []
    timed = random.sample(timed, min(5, len(timed))) if timed else []

    # Format per-day results as a readable table
    day_lines = []
    for d in sorted(per_day, key=lambda x: x.get("eval_date", "")):
        day_lines.append(
            f"  {d.get('eval_date','')} {d.get('symbol','')} — "
            f"signals={d.get('total_signals',0)}, hit={d.get('hit_rate',0):.0f}%, "
            f"exp={d.get('expectancy',0):.4f}"
        )

    def _sig_summary(sigs: list[dict]) -> str:
        parts = []
        for s in sigs:
            parts.append(
                f"  {s.get('symbol','')} {s.get('eval_date','')} "
                f"entry={s.get('fill_price',0):.2f} exit={s.get('exit_price',0):.2f} "
                f"ret={s.get('return_pct',0):+.2f}% "
                f"setup={s.get('setup_type','')} hold={s.get('hold_bars',0)}bars"
            )
        return "\n".join(parts) if parts else "  (none)"

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        environment_context=environment_text,
        strategy_code=source,
        eval_results="\n".join(day_lines),
        winners=_sig_summary(winners),
        losers=_sig_summary(losers),
        timed_out=_sig_summary(timed),
        hit_rate=aggregate.get("hit_rate", 0),
        avg_win=aggregate.get("avg_win", 0),
        avg_loss=aggregate.get("avg_loss", 0),
        expectancy=aggregate.get("expectancy", 0),
        profit_factor=aggregate.get("profit_factor", 0),
        max_drawdown=aggregate.get("max_drawdown", 0),
    )

    from xai_sdk.chat import user as sdk_user
    llm = get_grok_llm()
    chat = llm.client.chat.create(
        model=llm.model,
        messages=[sdk_user(prompt)],
        temperature=0.3,
        max_tokens=2048,
    )
    response = await chat.sample()

    # Track cost
    usage = response.usage
    get_cost_tracker().log_llm_call(
        model=llm.model,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        purpose="research_analysis",
    )

    return response.content


async def _llm_propose_strategy(
    current_source: str,
    analysis: str,
    history_summary: str,
    *,
    temperature: float = 0.5,
    slot: int = 1,
    environment_text: str = "",
) -> str:
    """Ask the LLM to propose a new strategy.py. Returns raw source code."""
    prompt = f"""CURRENT STRATEGY:
```python
{current_source}
```

ANALYSIS OF CURRENT STRATEGY:
{analysis}

HISTORY OF PAST ATTEMPTS (recent first):
{history_summary}

DATA CONTEXT (important!):
- Each trading day has ~390 bars (1-min candles, 9:30-16:00 ET)
- Universe: {len(RESEARCH_UNIVERSE)} medium-cap stocks
- candles DataFrame columns: ts, open, high, low, close, volume
- The scan() function receives ONE day of candles at a time (not multi-day)
- Bars are indexed 0-389 for a full day
- If your strategy produces 0 signals, it will be immediately discarded
- Prefer generous filters that produce 50+ signals over strict filters that produce 0

Based on the analysis, write an improved strategy.py.
Output ONLY the complete Python code -- no markdown fences, no explanation outside the code."""

    system_prompt = RESEARCH_SYSTEM_PROMPT.format(
        slot_id=f"{slot:02d}",
        num_slots=NUM_SLOTS,
        signal_schema=SIGNAL_SCHEMA,
        environment_context=environment_text,
    )

    from xai_sdk.chat import system as sdk_system, user as sdk_user
    llm = get_grok_llm()
    chat = llm.client.chat.create(
        model=llm.model,
        messages=[
            sdk_system(system_prompt),
            sdk_user(prompt),
        ],
        temperature=temperature,
        max_tokens=4096,
    )
    response = await chat.sample()

    # Track cost
    usage = response.usage
    get_cost_tracker().log_llm_call(
        model=llm.model,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        purpose="research_proposal",
    )

    # Strip markdown fences if the LLM wraps them anyway
    text = response.content.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def _store_environment_snapshot(
    snapshot: dict,
    round_num: int,
) -> int:
    """Store environment snapshot in DB. Returns snapshot id."""
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    fit = snapshot.get("strategy_fit", {})

    cur = db.execute(
        """INSERT INTO environment_snapshots
           (ts, round_num, volatility_regime, trend_regime, breadth_regime,
            momentum_regime, volume_regime, avg_atr_pct, dispersion,
            strategy_fit_json, raw_snapshot_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            round_num,
            snapshot.get("volatility_regime"),
            snapshot.get("trend_regime"),
            snapshot.get("breadth_regime"),
            snapshot.get("momentum_regime"),
            snapshot.get("volume_regime"),
            snapshot.get("avg_atr_pct"),
            snapshot.get("dispersion"),
            json.dumps(fit),
            json.dumps({k: v for k, v in snapshot.items() if k != "per_symbol"}),
        ),
    )
    db.commit()
    return cur.lastrowid


def _store_slot_environment_score(
    slot: int,
    env_snapshot_id: int,
    fitness: float,
    expectancy: float,
    total_signals: int,
    strategy_type: str,
):
    """Record how a slot performed in a given environment."""
    db = get_db()
    db.execute(
        """INSERT INTO slot_environment_scores
           (ts, slot, env_snapshot_id, fitness, expectancy, total_signals, strategy_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            slot,
            env_snapshot_id,
            fitness,
            expectancy,
            total_signals,
            strategy_type,
        ),
    )
    db.commit()


def _get_env_slot_history() -> str:
    """Get historical performance of strategy types across different environments.

    Returns formatted text showing which strategy types worked in which regimes.
    """
    db = get_db()
    rows = db.execute(
        """SELECT
               e.volatility_regime, e.trend_regime, e.breadth_regime,
               s.strategy_type, s.slot,
               AVG(s.fitness) as avg_fitness,
               AVG(s.expectancy) as avg_exp,
               COUNT(*) as observations
           FROM slot_environment_scores s
           JOIN environment_snapshots e ON s.env_snapshot_id = e.id
           GROUP BY e.volatility_regime, e.trend_regime, s.strategy_type
           HAVING observations >= 2
           ORDER BY avg_fitness DESC
           LIMIT 20"""
    ).fetchall()

    if not rows:
        return "(No environment-slot history yet — this data builds over time)"

    lines = []
    for r in rows:
        lines.append(
            f"  {r['strategy_type']:<20s} in vol={r['volatility_regime']:<7s} "
            f"trend={r['trend_regime']:<10s} breadth={r['breadth_regime']:<8s} "
            f"→ avg_fitness={r['avg_fitness']:.4f} avg_exp={r['avg_exp']:.4f} "
            f"(n={r['observations']})"
        )
    return "\n".join(lines)


def _get_trade_feedback() -> str:
    """Get real trade feedback — compare signals to actual trade outcomes.

    Returns formatted text showing execution gaps and real P&L vs simulated.
    """
    db = get_db()

    # Get recent trades with their linked signals
    trades = db.execute(
        """SELECT t.symbol, t.side, t.pnl, t.held_minutes, t.ts,
                  tf.simulated_return, tf.execution_gap, tf.slot
           FROM trades t
           LEFT JOIN trade_feedback tf ON tf.trade_id = t.id
           ORDER BY t.ts DESC LIMIT 30"""
    ).fetchall()

    if not trades:
        return "(No real trades recorded yet — feedback builds as the trader executes)"

    total_trades = len(trades)
    with_feedback = [t for t in trades if t["simulated_return"] is not None]

    lines = [f"  Recent trades: {total_trades}"]

    if with_feedback:
        avg_gap = sum(t["execution_gap"] or 0 for t in with_feedback) / len(with_feedback)
        avg_sim = sum(t["simulated_return"] or 0 for t in with_feedback) / len(with_feedback)
        avg_actual = sum(t["pnl"] or 0 for t in with_feedback) / len(with_feedback)
        lines.append(f"  Trades with feedback: {len(with_feedback)}")
        lines.append(f"  Avg simulated return: {avg_sim:+.2f}%")
        lines.append(f"  Avg actual PnL: ${avg_actual:+.2f}")
        lines.append(f"  Avg execution gap: {avg_gap:+.2f}")

        # Per-slot breakdown if available
        slot_gaps = {}
        for t in with_feedback:
            s = t["slot"]
            if s is not None:
                slot_gaps.setdefault(s, []).append(t["execution_gap"] or 0)
        for s, gaps in sorted(slot_gaps.items()):
            avg = sum(gaps) / len(gaps)
            lines.append(f"    Slot {s:02d}: avg gap={avg:+.3f} ({len(gaps)} trades)")
    else:
        # Just summarize raw trades
        total_pnl = sum(t["pnl"] or 0 for t in trades)
        winners = sum(1 for t in trades if (t["pnl"] or 0) > 0)
        lines.append(f"  Total P&L: ${total_pnl:+.2f}")
        lines.append(f"  Win rate: {winners}/{total_trades}")

    return "\n".join(lines)


def _match_trades_to_signals():
    """Attempt to match real trades to research signals and compute execution gaps.

    Matches by symbol + time proximity (within 30 minutes). Writes to trade_feedback table.
    """
    db = get_db()

    # Get unmatched trades
    unmatched = db.execute(
        """SELECT t.id, t.symbol, t.side, t.pnl, t.ts, t.held_minutes
           FROM trades t
           LEFT JOIN trade_feedback tf ON tf.trade_id = t.id
           WHERE tf.id IS NULL
           ORDER BY t.ts DESC LIMIT 50"""
    ).fetchall()

    if not unmatched:
        return

    matched = 0
    for trade in unmatched:
        # Find closest live signal for this symbol
        signal = db.execute(
            """SELECT ls.id, ls.slot, ls.direction,
                      s.return_pct as sim_return, s.strategy_id
               FROM live_signals ls
               LEFT JOIN signals s ON s.strategy_id = ls.strategy_id
                   AND s.symbol = ls.symbol
               WHERE ls.symbol = ?
               ORDER BY ABS(julianday(ls.ts) - julianday(?)) ASC
               LIMIT 1""",
            (trade["symbol"], trade["ts"]),
        ).fetchone()

        if signal:
            sim_return = signal["sim_return"] or 0
            actual_pnl = trade["pnl"] or 0
            # Normalize: execution gap in percentage terms
            # (positive gap = real trade did better than simulated)
            execution_gap = actual_pnl - sim_return  # rough comparison

            db.execute(
                """INSERT INTO trade_feedback
                   (ts, trade_id, signal_id, slot, simulated_return,
                    actual_pnl, execution_gap, symbol)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    trade["id"],
                    signal["id"],
                    signal["slot"],
                    sim_return,
                    actual_pnl,
                    execution_gap,
                    trade["symbol"],
                ),
            )
            matched += 1

    if matched:
        db.commit()
        logger.info(f"Trade feedback: matched {matched} trades to signals")


def _store_strategy(
    source: str,
    aggregate: dict,
    per_day: list[dict],
    all_results: list[dict],
    analysis: str,
    kept: bool,
    parent_id: Optional[int] = None,
    slot: int = 1,
) -> int:
    """Store strategy + evaluation results in SQLite. Returns strategy id."""
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    cur = db.execute(
        """INSERT INTO strategies
           (ts, slot, methodology, parent_id, total_signals, hit_rate, avg_rr, expectancy, kept, llm_analysis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            slot,
            source,
            parent_id,
            aggregate.get("total_signals", 0),
            aggregate.get("hit_rate"),
            aggregate.get("avg_rr"),
            aggregate.get("expectancy"),
            1 if kept else 0,
            analysis,
        ),
    )
    strategy_id = cur.lastrowid

    # Store per-day evaluations
    for d in per_day:
        eval_cur = db.execute(
            """INSERT INTO evaluations
               (strategy_id, ts, eval_date, symbols_json, signals_tested,
                hit_rate, expectancy, avg_rr, profit_factor, max_drawdown)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                strategy_id,
                now,
                d.get("eval_date", ""),
                json.dumps([d.get("symbol", "")]),
                d.get("total_signals", 0),
                d.get("hit_rate"),
                d.get("expectancy"),
                d.get("avg_rr"),
                d.get("profit_factor"),
                d.get("max_drawdown"),
            ),
        )
        eval_id = eval_cur.lastrowid

        # Store individual signals for this day/symbol
        day_signals = [
            r for r in all_results
            if r.get("eval_date") == d.get("eval_date") and r.get("symbol") == d.get("symbol")
        ]
        for sig in day_signals:
            db.execute(
                """INSERT INTO signals
                   (strategy_id, evaluation_id, symbol, entry_ts, direction,
                    order_type, setup_type, entry_price, target_price, stop_price,
                    max_hold_bars, hit_target, hit_stop, timed_out,
                    exit_ts, exit_price, return_pct, legs_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    strategy_id,
                    eval_id,
                    sig.get("symbol", ""),
                    str(sig.get("entry_bar", "")),
                    sig.get("direction"),
                    sig.get("order_type"),
                    sig.get("setup_type"),
                    sig.get("fill_price"),
                    sig.get("target_price"),
                    sig.get("stop_price"),
                    sig.get("max_hold_bars"),
                    sig.get("hit_target", 0),
                    sig.get("hit_stop", 0),
                    sig.get("timed_out", 0),
                    sig.get("exit_ts"),
                    sig.get("exit_price"),
                    sig.get("return_pct"),
                    json.dumps(sig.get("legs_json")) if sig.get("legs_json") else None,
                ),
            )

    db.commit()
    return strategy_id


def _get_history_summary(limit: int = 10, slot: int = 1) -> str:
    """Get recent strategy history for LLM context, scoped to slot."""
    db = get_db()
    rows = db.execute(
        """SELECT id, ts, total_signals, hit_rate, expectancy, kept, llm_analysis, methodology
           FROM strategies WHERE slot = ? ORDER BY id DESC LIMIT ?""",
        (slot, limit),
    ).fetchall()

    if not rows:
        return "(No prior attempts.)"

    lines = []
    failed_code_shown = 0
    for r in rows:
        kept_str = "KEPT" if r["kept"] else "DISCARDED"
        signals = r["total_signals"] or 0
        hit = r["hit_rate"] or 0
        exp = r["expectancy"] or 0
        lines.append(
            f"  #{r['id']} [{kept_str}] signals={signals} "
            f"hit={hit:.0f}% exp={exp:.4f}"
        )
        if signals == 0:
            lines.append("    WARNING: 0 signals = filters too strict. Loosen thresholds!")
        if r["llm_analysis"]:
            snippet = r["llm_analysis"][:200].replace("\n", " ")
            lines.append(f"    Analysis: {snippet}...")
        # Include parameter block of last 3 failed strategies so LLM avoids repeating
        if not r["kept"] and failed_code_shown < 3 and r["methodology"]:
            code_lines = r["methodology"].split("\n")[:30]
            lines.append("    Code (first 30 lines):")
            for cl in code_lines:
                lines.append(f"      {cl}")
            failed_code_shown += 1
    return "\n".join(lines)


def _compute_fitness(agg: dict) -> float:
    """Composite fitness score — expectancy weighted by signal count and consistency."""
    exp = agg.get("expectancy", 0)
    signals = agg.get("total_signals", 0)
    sharpe = agg.get("sharpe_approx", 0)
    # Discount strategies with fewer than 100 independent signals
    signal_factor = min(1.0, signals / 100)
    # Reward consistency (Sharpe ~1.0 is neutral, >1.5 capped)
    sharpe_factor = max(0.5, min(1.5, sharpe / 1.0)) if sharpe != 0 else 0.75
    return exp * signal_factor * sharpe_factor


def _get_best_strategy(slot: int = 1) -> tuple[Optional[int], Optional[str], float]:
    """Get the best kept strategy for a slot. Returns (id, source, expectancy)."""
    db = get_db()
    row = db.execute(
        """SELECT id, methodology, expectancy
           FROM strategies WHERE kept = 1 AND slot = ?
           ORDER BY expectancy DESC LIMIT 1""",
        (slot,),
    ).fetchone()
    if row:
        return row["id"], row["methodology"], row["expectancy"]
    return None, None, 0.0


# ═══════════════════════════════════════════════════════════════
# LIVE SCAN
# ═══════════════════════════════════════════════════════════════

def _run_live_scan(strategy_id: int, source: str, slot: int = 1):
    """Run best strategy against today's candles and write live_signals for this slot."""
    provider = get_data_provider()
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Clear old live signals for THIS slot only
    db.execute("DELETE FROM live_signals WHERE slot = ?", (slot,))

    for sym in RESEARCH_UNIVERSE:
        try:
            raw = provider.get_candles(sym, resolution=CANDLE_RESOLUTION, days_back=1)
            if not raw or len(raw) == 0:
                continue
            df = _candles_to_df(raw)
            if len(df) < 30:
                continue

            signals = _execute_strategy(source, df, sym)
            for sig in signals:
                db.execute(
                    """INSERT INTO live_signals
                       (strategy_id, slot, ts, symbol, direction, order_type,
                        setup_type, entry_price, target_price, stop_price,
                        max_hold_bars, legs_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        strategy_id,
                        slot,
                        now,
                        sym,
                        sig.get("direction"),
                        sig.get("order_type"),
                        sig.get("setup_type"),
                        sig.get("entry_price"),
                        sig.get("target_price"),
                        sig.get("stop_price"),
                        sig.get("max_hold_bars"),
                        json.dumps(sig.get("legs_json")) if sig.get("legs_json") else None,
                    ),
                )
        except Exception as e:
            logger.warning(f"Live scan failed for {sym}: {e}")

    db.commit()
    count = db.execute("SELECT COUNT(*) FROM live_signals WHERE slot = ?", (slot,)).fetchone()[0]
    logger.info(f"Live scan [slot {slot:02d}]: {count} signals written for {len(RESEARCH_UNIVERSE)} symbols")


# ═══════════════════════════════════════════════════════════════
# GIT COMMIT (Karpathy pattern — version-control the evolving artifact)
# ═══════════════════════════════════════════════════════════════

_REPO_ROOT = Path(__file__).resolve().parent.parent


async def _git_commit(iteration: int, kept: bool, expectancy: float, strategy_id: int | None, slot: int = 1):
    """Commit slot strategy file after each iteration so git log shows full evolution."""
    async with _GIT_LOCK:
        try:
            strategy_rel = f"research/slots/strategy_{slot:02d}.py"
            subprocess.run(
                ["git", "add", strategy_rel],
                cwd=_REPO_ROOT, capture_output=True, timeout=10,
            )
            status = "KEPT" if kept else "DISCARDED"
            msg = f"research [slot {slot:02d}] iter {iteration}: {status} exp={expectancy:.4f} (#{strategy_id})"
            result = subprocess.run(
                ["git", "commit", "-m", msg, "--", strategy_rel],
                cwd=_REPO_ROOT, capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"Git commit: {msg}")
                push = subprocess.run(
                    ["git", "push"],
                    cwd=_REPO_ROOT, capture_output=True, timeout=30,
                )
                if push.returncode == 0:
                    logger.debug("Git push: success")
                else:
                    logger.debug(f"Git push failed: {push.stderr.decode(errors='replace').strip()}")
            else:
                logger.debug("Git: nothing to commit (strategy unchanged)")
        except Exception as e:
            logger.debug(f"Git commit skipped: {e}")


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

async def _run_slot(
    slot: int,
    iteration: int,
    train_universe: dict,
    test_universe: dict,
    state: dict,
    *,
    dry_run: bool = False,
    environment_text: str = "",
) -> dict:
    """Run one evolution iteration for a single slot. Returns updated state."""
    logger.info(f"[Slot {slot:02d}] Iteration {iteration} starting")

    # ── 1. Load current strategy ────────────────────────────────
    strat_path = _strategy_path(slot)
    try:
        current_source = strat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(f"[Slot {slot:02d}] strategy file not found — skipping")
        return state

    # ── 2. Evaluate current strategy on TRAIN set ───────────────
    logger.info(f"[Slot {slot:02d}] Evaluating current (train)...")
    train_agg, train_per_day, train_results = await asyncio.to_thread(
        _evaluate_strategy, current_source, train_universe,
    )
    logger.info(
        f"[Slot {slot:02d}] Train: signals={train_agg['total_signals']} "
        f"hit={train_agg['hit_rate']:.0f}% exp={train_agg['expectancy']:.4f} "
        f"pf={train_agg['profit_factor']:.2f}"
    )

    # ── 3. Evaluate current strategy on TEST set ────────────────
    test_agg, test_per_day, test_results = await asyncio.to_thread(
        _evaluate_strategy, current_source, test_universe,
    )
    best_test_fitness = _compute_fitness(test_agg)
    state["best_test_fitness"] = max(state["best_test_fitness"], best_test_fitness)
    logger.info(
        f"[Slot {slot:02d}] Test:  signals={test_agg['total_signals']} "
        f"hit={test_agg['hit_rate']:.0f}% exp={test_agg['expectancy']:.4f} "
        f"fitness={best_test_fitness:.4f}"
    )

    # ── 4. LLM analysis (on TRAIN results only) ────────────────
    logger.info(f"[Slot {slot:02d}] Running LLM analysis...")
    async with _LLM_SEMAPHORE:
        analysis = await _llm_analyze(
            current_source, train_agg, train_per_day, train_results,
            environment_text=environment_text,
        )
    logger.info(f"[Slot {slot:02d}] Analysis: {analysis[:120]}...")

    # ── 5. Store baseline if first iteration ────────────────────
    best_id, best_source, best_exp = _get_best_strategy(slot)
    is_first = best_id is None

    if is_first:
        current_id = _store_strategy(
            current_source, train_agg, train_per_day, train_results, analysis,
            kept=True, parent_id=None, slot=slot,
        )
        best_id = current_id
        logger.info(f"[Slot {slot:02d}] First strategy stored as baseline (id={current_id})")
    else:
        current_id = best_id

    # ── 6. Propose new strategy ─────────────────────────────────
    consecutive_failures = state["consecutive_failures"]
    proposal_temp = min(1.0, 0.5 + consecutive_failures * 0.05)
    history = _get_history_summary(slot=slot)
    logger.info(
        f"[Slot {slot:02d}] Proposing new strategy "
        f"(temp={proposal_temp:.2f}, failures={consecutive_failures})..."
    )
    async with _LLM_SEMAPHORE:
        new_source = await _llm_propose_strategy(
            current_source, analysis, history,
            temperature=proposal_temp, slot=slot,
            environment_text=environment_text,
        )

    # ── 7. Validate ─────────────────────────────────────────────
    validation_err = _validate_strategy_source(new_source)
    if validation_err:
        logger.warning(f"[Slot {slot:02d}] Validation failed: {validation_err}")
        _store_strategy(
            new_source, {"total_signals": 0}, [], [], validation_err,
            kept=False, parent_id=current_id, slot=slot,
        )
        state["consecutive_failures"] += 1
        return state

    # ── 8. Evaluate proposed on TRAIN ───────────────────────────
    logger.info(f"[Slot {slot:02d}] Evaluating proposed (train)...")
    new_train_agg, new_train_pd, new_train_res = await asyncio.to_thread(
        _evaluate_strategy, new_source, train_universe,
    )
    logger.info(
        f"[Slot {slot:02d}] Proposed train: signals={new_train_agg['total_signals']} "
        f"hit={new_train_agg['hit_rate']:.0f}% exp={new_train_agg['expectancy']:.4f} "
        f"pf={new_train_agg['profit_factor']:.2f}"
    )

    # ── 9. Evaluate proposed on TEST ────────────────────────────
    new_test_agg, new_test_pd, new_test_res = await asyncio.to_thread(
        _evaluate_strategy, new_source, test_universe,
    )
    new_test_fitness = _compute_fitness(new_test_agg)
    logger.info(
        f"[Slot {slot:02d}] Proposed test:  signals={new_test_agg['total_signals']} "
        f"hit={new_test_agg['hit_rate']:.0f}% exp={new_test_agg['expectancy']:.4f} "
        f"fitness={new_test_fitness:.4f}"
    )

    # ── 10. LLM analysis of new strategy ────────────────────────
    if new_train_agg.get("total_signals", 0) > 0:
        async with _LLM_SEMAPHORE:
            new_analysis = await _llm_analyze(
                new_source, new_train_agg, new_train_pd, new_train_res,
                environment_text=environment_text,
            )
    else:
        new_analysis = (
            "ZERO SIGNALS produced. The filters are too strict for the data. "
            "Each trading day has ~390 bars (1-min, 9:30-16:00 ET). "
            "Loosen conditions, lower thresholds, or simplify the logic."
        )
        logger.warning(f"[Slot {slot:02d}] Proposed strategy produced 0 signals")

    # ── 11. Keep or discard (fitness + annealing) ───────────────
    min_test_signals = 15
    has_enough_signals = new_test_agg.get("total_signals", 0) >= min_test_signals
    train_positive = new_train_agg.get("expectancy", 0) > 0

    best_fitness = state["best_test_fitness"]
    strict_improved = (
        new_test_fitness > best_fitness
        and has_enough_signals
        and train_positive
    )

    annealing_accept = False
    if (
        not strict_improved
        and has_enough_signals
        and train_positive
        and consecutive_failures >= 5
        and best_fitness > 0
        and new_test_fitness >= best_fitness * 0.9
    ):
        annealing_accept = True

    kept = strict_improved or annealing_accept

    if kept:
        if annealing_accept:
            logger.info(
                f"[Slot {slot:02d}] ACCEPTING (annealing): fitness {new_test_fitness:.4f} "
                f"vs best {best_fitness:.4f} after {consecutive_failures} failures"
            )
        else:
            logger.info(
                f"[Slot {slot:02d}] KEEPING: fitness {best_fitness:.4f} "
                f"-> {new_test_fitness:.4f} (+{new_test_fitness - best_fitness:.4f})"
            )
        new_id = _store_strategy(
            new_source, new_train_agg, new_train_pd, new_train_res, new_analysis,
            kept=True, parent_id=current_id, slot=slot,
        )
        if not dry_run:
            strat_path.write_text(new_source, encoding="utf-8")
            logger.info(f"[Slot {slot:02d}] Strategy updated (id={new_id})")
        best_id = new_id
        state["best_test_fitness"] = new_test_fitness
        state["consecutive_failures"] = 0
        state["last_expectancy"] = new_train_agg.get("expectancy", 0)
        state["last_total_signals"] = new_train_agg.get("total_signals", 0)
    else:
        reasons = []
        if new_test_fitness <= best_fitness:
            reasons.append(f"lower fitness ({new_test_fitness:.4f} vs {best_fitness:.4f})")
        if not has_enough_signals:
            reasons.append(
                f"too few test signals ({new_test_agg.get('total_signals', 0)}<{min_test_signals})"
            )
        if not train_positive:
            reasons.append("negative train expectancy")
        logger.info(f"[Slot {slot:02d}] DISCARDING: {', '.join(reasons)}")
        _store_strategy(
            new_source, new_train_agg, new_train_pd, new_train_res, new_analysis,
            kept=False, parent_id=current_id, slot=slot,
        )
        state["consecutive_failures"] += 1

    # ── 12. Live scan ───────────────────────────────────────────
    if best_id is not None:
        best_source = strat_path.read_text(encoding="utf-8") if not dry_run else current_source
        await asyncio.to_thread(_run_live_scan, best_id, best_source, slot)

    # ── 13. Git commit ──────────────────────────────────────────
    if not dry_run:
        await _git_commit(
            iteration, kept,
            new_test_fitness if kept else best_fitness,
            best_id, slot=slot,
        )

    logger.info(
        f"[Slot {slot:02d}] Iteration {iteration} complete — "
        f"best fitness={state['best_test_fitness']:.4f} (#{best_id}) "
        f"failures={state['consecutive_failures']}"
    )
    return state


# ═══════════════════════════════════════════════════════════════
# META-LEARNING SELECTOR AGENT
# ═══════════════════════════════════════════════════════════════

def _extract_strategy_type(source: str) -> str:
    """Extract the primary strategy type from strategy source code."""
    for line in source.split("\n"):
        if "setup_type" in line and "=" in line:
            parts = line.split('"')
            if len(parts) >= 2:
                return parts[-2]
    # Fallback: detect from order_type
    if "iron_condor" in source:
        return "iron_condor"
    if "straddle" in source:
        return "straddle"
    if "vertical_spread" in source:
        return "vertical_spread"
    if "strangle" in source:
        return "strangle"
    if "mean_rev" in source.lower() or "rsi" in source.lower():
        return "mean_reversion"
    if "breakout" in source.lower() or "momentum" in source.lower():
        return "momentum_breakout"
    if "vwap" in source.lower():
        return "vwap"
    return "unknown"


async def _run_selector(
    slot_states: dict[int, dict],
    env_snapshot: dict,
    env_snapshot_id: int,
):
    """Meta-learning selector: assess environment, correlate with slot performance,
    and make intelligent slot allocation decisions.

    Unlike the old selector (replace worst by fitness), this one:
    1. Reads the current market environment
    2. Looks up historical environment-slot performance correlations
    3. Considers real trade feedback (execution gap)
    4. Ensures portfolio diversity
    5. Can act on multiple slots per run
    """
    logger.info("=" * 60)
    logger.info("META-LEARNING SELECTOR — Environment-Aware Portfolio Optimization")
    logger.info("=" * 60)

    # Match any unmatched real trades to signals before analysis
    _match_trades_to_signals()

    # ── Build slot rankings with environment scores ─────────────
    rankings = []
    for slot_num in range(1, NUM_SLOTS + 1):
        state = slot_states[slot_num]
        strat_path = _strategy_path(slot_num)
        try:
            source = strat_path.read_text(encoding="utf-8")
            strategy_type = _extract_strategy_type(source)
        except FileNotFoundError:
            source = ""
            strategy_type = "unknown"

        # Record this slot's performance in this environment
        _store_slot_environment_score(
            slot=slot_num,
            env_snapshot_id=env_snapshot_id,
            fitness=state["best_test_fitness"],
            expectancy=state.get("last_expectancy", 0),
            total_signals=state.get("last_total_signals", 0),
            strategy_type=strategy_type,
        )

        # Get environment fit for this strategy type
        fit_scores = env_snapshot.get("strategy_fit", {})
        # Map strategy types to fit score keys
        type_to_fit = {
            "momentum_breakout": "momentum_breakout",
            "breakout_volume": "momentum_breakout",
            "opening_range_breakout": "momentum_breakout",
            "mean_reversion": "mean_reversion",
            "mean_reversion_rsi": "mean_reversion",
            "vwap": "vwap",
            "vwap_bounce": "vwap",
            "iron_condor": "short_premium",
            "cash_secured_put": "short_premium",
            "straddle": "straddle",
            "strangle": "straddle",
            "vertical_spread": "vertical_spread",
            "bull_call_spread": "vertical_spread",
            "bear_put_spread": "vertical_spread",
            "long_call": "long_options",
            "long_put": "long_options",
            "bracket": "bracket",
        }
        fit_key = type_to_fit.get(strategy_type, "momentum_breakout")
        env_fit = fit_scores.get(fit_key, 0.5)

        rankings.append({
            "slot": slot_num,
            "fitness": state["best_test_fitness"],
            "iterations": state["iteration"],
            "failures": state["consecutive_failures"],
            "strategy_type": strategy_type,
            "env_fit": env_fit,
            # Composite: raw fitness weighted by environment fit
            "env_adjusted_fitness": state["best_test_fitness"] * (0.5 + 0.5 * env_fit),
        })

    rankings.sort(key=lambda x: x["env_adjusted_fitness"], reverse=True)

    # Log rankings
    for i, r in enumerate(rankings):
        marker = " ★" if i == 0 else " ✗" if i >= len(rankings) - 2 else ""
        logger.info(
            f"  #{i+1} Slot {r['slot']:02d}: fitness={r['fitness']:.4f} "
            f"env_fit={r['env_fit']:.2f} adj={r['env_adjusted_fitness']:.4f} "
            f"iters={r['iterations']} type={r['strategy_type']}{marker}"
        )

    # ── Get historical and feedback context ─────────────────────
    env_slot_history = _get_env_slot_history()
    trade_feedback = _get_trade_feedback()
    environment_text = format_environment_for_prompt(env_snapshot)

    # ── Build prompt for meta-learning selector ─────────────────
    slot_ranking_text = "\n".join(
        f"  Slot {r['slot']:02d}: fitness={r['fitness']:.4f} "
        f"env_fit={r['env_fit']:.2f} adj_fitness={r['env_adjusted_fitness']:.4f} "
        f"iters={r['iterations']} failures={r['failures']} type={r['strategy_type']}"
        for r in rankings
    )

    prompt = SELECTOR_PROMPT.format(
        num_slots=NUM_SLOTS,
        slot_rankings=slot_ranking_text,
        environment_context=environment_text,
        env_slot_history=env_slot_history,
        trade_feedback=trade_feedback,
        max_replacements=MAX_SELECTOR_REPLACEMENTS,
    )

    from xai_sdk.chat import system as sdk_system, user as sdk_user
    llm = get_grok_llm()
    async with _LLM_SEMAPHORE:
        chat = llm.client.chat.create(
            model=llm.model,
            messages=[sdk_user(prompt)],
            temperature=0.7,
            max_tokens=8192,
        )
        response = await chat.sample()

    usage = response.usage
    get_cost_tracker().log_llm_call(
        model=llm.model,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        purpose="meta_selector",
    )

    # ── Parse response ──────────────────────────────────────────
    text = response.content.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        logger.warning("Selector: Could not parse JSON response — skipping")
        return

    try:
        decision = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("Selector: Invalid JSON — skipping")
        return

    analysis = decision.get("analysis", "")
    actions = decision.get("actions", [])
    env_learning = decision.get("environment_learning", "")
    recommended_focus = decision.get("recommended_focus", [])

    logger.info(f"Selector analysis: {analysis[:200]}...")
    logger.info(f"Environment learning: {env_learning}")
    logger.info(f"Recommended focus: {recommended_focus}")

    # ── Execute actions ─────────────────────────────────────────
    replacements_done = 0
    for action_item in actions:
        if replacements_done >= MAX_SELECTOR_REPLACEMENTS:
            logger.info(f"Selector: hit max replacements ({MAX_SELECTOR_REPLACEMENTS})")
            break

        slot_num = action_item.get("slot")
        action = action_item.get("action", "keep")
        reason = action_item.get("reason", "")
        seed_code = action_item.get("seed_code")

        if not slot_num or slot_num < 1 or slot_num > NUM_SLOTS:
            continue

        logger.info(f"Selector: Slot {slot_num:02d} → {action} — {reason}")

        if action == "keep":
            continue

        if action in ("replace", "mutate"):
            if not seed_code:
                # If mutate without code, try copying from seed_from_slot
                seed_from = action_item.get("seed_from_slot")
                if seed_from and 1 <= seed_from <= NUM_SLOTS:
                    try:
                        seed_code = _strategy_path(seed_from).read_text(encoding="utf-8")
                    except FileNotFoundError:
                        logger.warning(f"Selector: seed slot {seed_from} not found — skipping")
                        continue
                else:
                    logger.warning("Selector: No seed_code or seed_from_slot — skipping")
                    continue

            # Validate
            validation_err = _validate_strategy_source(seed_code)
            if validation_err:
                logger.warning(f"Selector: Seed code for slot {slot_num:02d} failed validation: {validation_err}")
                continue

            # Replace
            strat_path = _strategy_path(slot_num)
            strat_path.write_text(seed_code, encoding="utf-8")
            logger.info(f"Selector: Replaced Slot {slot_num:02d} ({action})")

            # Reset slot state
            slot_states[slot_num] = {
                "consecutive_failures": 0,
                "best_test_fitness": 0.0,
                "iteration": 0,
                "last_expectancy": 0,
                "last_total_signals": 0,
            }

            await _git_commit(0, True, 0.0, None, slot=slot_num)
            replacements_done += 1

    logger.info(f"Selector complete: {replacements_done} replacements made")


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

async def run_research(*, verbose: bool = False, dry_run: bool = False):
    """Main research loop — all slots evolve concurrently, meta-learner optimizes."""
    global _LLM_SEMAPHORE
    _LLM_SEMAPHORE = asyncio.Semaphore(4)  # max 4 concurrent LLM calls

    from memory import init_db
    init_db()

    slot_ids = list(range(1, NUM_SLOTS + 1))
    logger.info("=" * 60)
    logger.info("RESEARCH AGENT — Population-Based Strategy Evolution")
    logger.info(f"Slots: {NUM_SLOTS} concurrent")
    logger.info(f"Universe: {len(RESEARCH_UNIVERSE)} symbols")
    logger.info(f"Eval window: last {EVAL_DAYS_BACK} trading days, {CANDLE_RESOLUTION}-min bars")
    logger.info(f"Selector runs every {SELECTOR_EVERY_N_ROUNDS} rounds (meta-learning)")
    logger.info("=" * 60)

    # Per-slot state
    slot_states: dict[int, dict] = {}
    for s in slot_ids:
        slot_states[s] = {
            "consecutive_failures": 0,
            "best_test_fitness": 0.0,
            "iteration": 0,
            "last_expectancy": 0,
            "last_total_signals": 0,
        }

    round_num = 0

    while True:
        round_num += 1
        logger.info(f"{'='*20} Round {round_num} {'='*20}")

        # Fetch candles once per round (shared across all slots)
        logger.info("Fetching candle data...")
        universe = _fetch_universe_candles(EVAL_DAYS_BACK, exclude_today=True)
        total_days = sum(len(days) for days in universe.values())
        logger.info(f"Got data for {len(universe)} symbols across {total_days} trading days")

        if not universe:
            logger.warning("No candle data available — waiting 5 min")
            await asyncio.sleep(300)
            continue

        # ── Compute market environment ──────────────────────────
        logger.info("Computing market environment...")
        env_snapshot = compute_environment(universe)
        env_text = format_environment_for_prompt(env_snapshot)
        env_snapshot_id = _store_environment_snapshot(env_snapshot, round_num)
        logger.info(f"Environment snapshot stored (id={env_snapshot_id})")

        train_universe, test_universe = _train_test_split(universe)
        train_days = sum(len(d) for d in train_universe.values())
        test_days = sum(len(d) for d in test_universe.values())
        logger.info(f"Split: train={train_days} days, test={test_days} days")

        # Run all slots concurrently (with environment context)
        for s in slot_ids:
            slot_states[s]["iteration"] += 1

        tasks = [
            _run_slot(
                slot=s,
                iteration=slot_states[s]["iteration"],
                train_universe=train_universe,
                test_universe=test_universe,
                state=slot_states[s],
                dry_run=dry_run,
                environment_text=env_text,
            )
            for s in slot_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update states from results
        for s, result in zip(slot_ids, results):
            if isinstance(result, Exception):
                logger.error(f"[Slot {s:02d}] Failed: {result}")
            elif isinstance(result, dict):
                slot_states[s] = result

        # Meta-learning selector agent
        if round_num % SELECTOR_EVERY_N_ROUNDS == 0:
            try:
                await _run_selector(slot_states, env_snapshot, env_snapshot_id)
            except Exception as e:
                logger.error(f"Selector failed: {e}")

        # Summary
        logger.info(f"Round {round_num} complete. Slot fitness summary:")
        for s in slot_ids:
            st = slot_states[s]
            logger.info(
                f"  Slot {s:02d}: fitness={st['best_test_fitness']:.4f} "
                f"iter={st['iteration']} failures={st['consecutive_failures']}"
            )
        logger.info("Starting next round...")
