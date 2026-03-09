"""
Research Agent — Karpathy autoresearch pattern for day trading.

Core loop: load strategy → evaluate across trading days → LLM analysis →
propose new strategy.py → safety validate → evaluate → keep if improved → repeat.

Also runs a live scan: best strategy against today's accruing candles.
"""

import ast
import asyncio
import json
import logging
import subprocess
import textwrap
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from core.grok_llm import get_grok_llm, REASONING_MODEL
from data.cost_tracker import get_cost_tracker
from data.data_provider import get_data_provider
from memory import get_db
from research.config import (
    ANALYSIS_PROMPT_TEMPLATE,
    CANDLE_RESOLUTION,
    EVAL_DAYS_BACK,
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_UNIVERSE,
    SANDBOX_ALLOWED_IMPORTS,
    SANDBOX_BLOCKED_CALLS,
)
from research.simulator import compute_expectancy, simulate

logger = logging.getLogger(__name__)

_STRATEGY_PATH = Path(__file__).parent / "strategy.py"


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


def _split_by_trading_day(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a multi-day candle DataFrame into per-day DataFrames.

    Returns {date_str: df} where date_str is YYYY-MM-DD.
    """
    if df.empty:
        return {}

    # Convert unix timestamps to ET datetimes
    df = df.copy()
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    # Approximate ET as UTC-4 (EDT). Good enough for date boundaries.
    df["dt_et"] = df["dt"] - pd.Timedelta(hours=4)
    df["trade_date"] = df["dt_et"].dt.date.astype(str)

    days = {}
    for d, group in df.groupby("trade_date"):
        day_df = group.drop(columns=["dt", "dt_et", "trade_date"]).reset_index(drop=True)
        if len(day_df) >= 30:  # need at least 30 bars for a meaningful test
            days[d] = day_df
    return days


def _fetch_universe_candles(days_back: int) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch candles for all symbols, split by trading day.

    Returns: {symbol: {date_str: df}}
    """
    provider = get_data_provider()
    universe: dict[str, dict[str, pd.DataFrame]] = {}

    for sym in RESEARCH_UNIVERSE:
        try:
            raw = provider.get_candles(sym, resolution=CANDLE_RESOLUTION, days_back=days_back)
            if raw and len(raw) > 0:
                df = _candles_to_df(raw)
                days = _split_by_trading_day(df)
                if days:
                    universe[sym] = days
        except Exception as e:
            logger.warning(f"Failed to fetch candles for {sym}: {e}")

    return universe


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
) -> str:
    """Ask the LLM to analyze backtest results. Returns analysis text."""
    winners = [r for r in all_results if r.get("return_pct", 0) > 0][:10]
    losers = [r for r in all_results if r.get("return_pct", 0) < 0][:10]
    timed = [r for r in all_results if r.get("timed_out")][:5]

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

Based on the analysis, write an improved strategy.py.
Output ONLY the complete Python code — no markdown fences, no explanation outside the code."""

    from xai_sdk.chat import system as sdk_system, user as sdk_user
    llm = get_grok_llm()
    chat = llm.client.chat.create(
        model=llm.model,
        messages=[
            sdk_system(RESEARCH_SYSTEM_PROMPT),
            sdk_user(prompt),
        ],
        temperature=0.4,
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

def _store_strategy(
    source: str,
    aggregate: dict,
    per_day: list[dict],
    all_results: list[dict],
    analysis: str,
    kept: bool,
    parent_id: Optional[int] = None,
) -> int:
    """Store strategy + evaluation results in SQLite. Returns strategy id."""
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    cur = db.execute(
        """INSERT INTO strategies
           (ts, methodology, parent_id, total_signals, hit_rate, avg_rr, expectancy, kept, llm_analysis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
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


def _get_history_summary(limit: int = 10) -> str:
    """Get recent strategy history for LLM context."""
    db = get_db()
    rows = db.execute(
        """SELECT id, ts, total_signals, hit_rate, expectancy, kept, llm_analysis
           FROM strategies ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()

    if not rows:
        return "(No prior attempts.)"

    lines = []
    for r in rows:
        kept_str = "KEPT" if r["kept"] else "DISCARDED"
        lines.append(
            f"  #{r['id']} [{kept_str}] signals={r['total_signals']} "
            f"hit={r['hit_rate']:.0f}% exp={r['expectancy']:.4f}"
        )
        if r["llm_analysis"]:
            # First 200 chars of analysis
            snippet = r["llm_analysis"][:200].replace("\n", " ")
            lines.append(f"    Analysis: {snippet}...")
    return "\n".join(lines)


def _get_best_strategy() -> tuple[Optional[int], Optional[str], float]:
    """Get the best kept strategy. Returns (id, source, expectancy)."""
    db = get_db()
    row = db.execute(
        """SELECT id, methodology, expectancy
           FROM strategies WHERE kept = 1
           ORDER BY expectancy DESC LIMIT 1"""
    ).fetchone()
    if row:
        return row["id"], row["methodology"], row["expectancy"]
    return None, None, 0.0


# ═══════════════════════════════════════════════════════════════
# LIVE SCAN
# ═══════════════════════════════════════════════════════════════

def _run_live_scan(strategy_id: int, source: str):
    """Run best strategy against today's candles and write live_signals."""
    provider = get_data_provider()
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Clear old live signals
    db.execute("DELETE FROM live_signals")

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
                       (strategy_id, ts, symbol, direction, order_type,
                        setup_type, entry_price, target_price, stop_price,
                        max_hold_bars, legs_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        strategy_id,
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
    count = db.execute("SELECT COUNT(*) FROM live_signals").fetchone()[0]
    logger.info(f"Live scan: {count} signals written for {len(RESEARCH_UNIVERSE)} symbols")


# ═══════════════════════════════════════════════════════════════
# GIT COMMIT (Karpathy pattern — version-control the evolving artifact)
# ═══════════════════════════════════════════════════════════════

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit(iteration: int, kept: bool, expectancy: float, strategy_id: int | None):
    """Commit research/strategy.py after each iteration so git log shows full evolution."""
    try:
        # Stage only the strategy file
        subprocess.run(
            ["git", "add", "research/strategy.py"],
            cwd=_REPO_ROOT, capture_output=True, timeout=10,
        )
        status = "KEPT" if kept else "DISCARDED"
        msg = f"research iter {iteration}: {status} exp={expectancy:.4f} (#{strategy_id})"
        result = subprocess.run(
            ["git", "commit", "-m", msg, "--", "research/strategy.py"],
            cwd=_REPO_ROOT, capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            logger.info(f"Git commit: {msg}")
        else:
            # Nothing to commit (strategy unchanged) — that's fine
            logger.debug("Git: nothing to commit (strategy unchanged)")
    except Exception as e:
        logger.debug(f"Git commit skipped: {e}")


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

async def run_research(*, verbose: bool = False, dry_run: bool = False):
    """Main research loop — hypothesize, evaluate, keep/discard, repeat."""
    from memory import init_db
    init_db()

    logger.info("=" * 60)
    logger.info("RESEARCH AGENT — Strategy Evolution")
    logger.info(f"Universe: {len(RESEARCH_UNIVERSE)} symbols")
    logger.info(f"Eval window: last {EVAL_DAYS_BACK} trading days, {CANDLE_RESOLUTION}-min bars")
    logger.info("=" * 60)

    iteration = 0

    while True:
        iteration += 1
        logger.info(f"{'─'*40} Iteration {iteration} {'─'*40}")

        # ── 1. Load current strategy ────────────────────────────
        try:
            current_source = _STRATEGY_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(f"strategy.py not found at {_STRATEGY_PATH}")
            return

        # ── 2. Fetch candles ────────────────────────────────────
        logger.info("Fetching candle data...")
        universe = _fetch_universe_candles(EVAL_DAYS_BACK)
        total_days = sum(len(days) for days in universe.values())
        logger.info(f"Got data for {len(universe)} symbols across {total_days} trading days")

        if not universe:
            logger.warning("No candle data available — waiting 5 min")
            await asyncio.sleep(300)
            continue

        # ── 3. Evaluate current strategy ────────────────────────
        logger.info("Evaluating current strategy...")
        agg, per_day, all_results = _evaluate_strategy(current_source, universe)
        logger.info(
            f"Current: signals={agg['total_signals']} hit={agg['hit_rate']:.0f}% "
            f"exp={agg['expectancy']:.4f} pf={agg['profit_factor']:.2f}"
        )

        if agg["total_signals"] == 0:
            logger.warning("Current strategy produced zero signals")

        # ── 4. LLM analysis ─────────────────────────────────────
        logger.info("Running LLM analysis...")
        analysis = await _llm_analyze(current_source, agg, per_day, all_results)
        logger.info(f"Analysis: {analysis[:150]}...")

        # Store current strategy evaluation
        best_id, _, best_exp = _get_best_strategy()
        is_first = best_id is None

        current_id = _store_strategy(
            current_source, agg, per_day, all_results, analysis,
            kept=is_first,  # keep first strategy by default
            parent_id=best_id,
        )

        if is_first:
            best_id = current_id
            best_exp = agg.get("expectancy", 0)
            logger.info(f"First strategy stored as baseline (id={current_id})")

        # ── 5. Propose new strategy ─────────────────────────────
        history = _get_history_summary()
        logger.info("Proposing new strategy...")
        new_source = await _llm_propose_strategy(current_source, analysis, history)

        # ── 6. Validate new strategy ────────────────────────────
        validation_err = _validate_strategy_source(new_source)
        if validation_err:
            logger.warning(f"Proposed strategy failed validation: {validation_err}")
            # Store as discarded
            _store_strategy(new_source, {"total_signals": 0}, [], [], validation_err, kept=False, parent_id=current_id)
            continue

        # ── 7. Evaluate new strategy ────────────────────────────
        logger.info("Evaluating proposed strategy...")
        new_agg, new_per_day, new_all_results = _evaluate_strategy(new_source, universe)
        logger.info(
            f"Proposed: signals={new_agg['total_signals']} hit={new_agg['hit_rate']:.0f}% "
            f"exp={new_agg['expectancy']:.4f} pf={new_agg['profit_factor']:.2f}"
        )

        # ── 8. LLM analysis of new strategy ─────────────────────
        new_analysis = await _llm_analyze(new_source, new_agg, new_per_day, new_all_results)

        # ── 9. Keep or discard ──────────────────────────────────
        new_exp = new_agg.get("expectancy", 0)
        improved = new_exp > best_exp and new_agg.get("total_signals", 0) >= 5

        if improved:
            logger.info(
                f"KEEPING new strategy: exp {best_exp:.4f} → {new_exp:.4f} "
                f"(+{new_exp - best_exp:.4f})"
            )
            new_id = _store_strategy(
                new_source, new_agg, new_per_day, new_all_results, new_analysis,
                kept=True, parent_id=current_id,
            )

            if not dry_run:
                _STRATEGY_PATH.write_text(new_source, encoding="utf-8")
                logger.info(f"strategy.py updated (id={new_id})")

            best_id = new_id
            best_exp = new_exp
        else:
            reason = "lower expectancy" if new_exp <= best_exp else "too few signals"
            logger.info(
                f"DISCARDING: exp {new_exp:.4f} vs best {best_exp:.4f} ({reason})"
            )
            _store_strategy(
                new_source, new_agg, new_per_day, new_all_results, new_analysis,
                kept=False, parent_id=current_id,
            )

        # ── 10. Live scan ───────────────────────────────────────
        if best_id is not None:
            best_source = _STRATEGY_PATH.read_text(encoding="utf-8") if not dry_run else current_source
            _run_live_scan(best_id, best_source)

        # ── 11. Git commit ──────────────────────────────────────
        if not dry_run:
            _git_commit(iteration, improved, best_exp, best_id)

        logger.info(
            f"Iteration {iteration} complete — "
            f"best exp={best_exp:.4f} (strategy #{best_id})"
        )
