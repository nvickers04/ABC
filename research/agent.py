"""
Research Agent — Karpathy autoresearch pattern for day trading.

Architecture:
  - NUM_SLOTS independent strategy slots, each evolved concurrently
  - All slots run in parallel via asyncio.gather (LLM calls are async)
  - Strategy evals run in thread pool (CPU-bound pandas/numpy)
  - Selector agent periodically replaces worst slot with fresh or mutated strategy
  - Git commits serialized via asyncio.Lock
"""

import ast
import asyncio
import json
import logging
import random
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
    NUM_SLOTS,
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_UNIVERSE,
    SANDBOX_ALLOWED_IMPORTS,
    SANDBOX_BLOCKED_CALLS,
    SELECTOR_EVERY_N_ROUNDS,
    SELECTOR_PROMPT,
    SIGNAL_SCHEMA,
)
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
        analysis = await _llm_analyze(current_source, train_agg, train_per_day, train_results)
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
# SELECTOR AGENT
# ═══════════════════════════════════════════════════════════════

async def _run_selector(slot_states: dict[int, dict]):
    """Evaluate all slots and replace the weakest if appropriate."""
    logger.info("=" * 60)
    logger.info("SELECTOR AGENT — Evaluating slot population")
    logger.info("=" * 60)

    # Rank slots by fitness
    rankings = []
    for slot_num in range(1, NUM_SLOTS + 1):
        state = slot_states[slot_num]
        strat_path = _strategy_path(slot_num)
        setup_type = "unknown"
        try:
            source = strat_path.read_text(encoding="utf-8")
            # Extract setup_type from code
            for line in source.split("\n"):
                if "setup_type" in line and "=" in line:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        setup_type = parts[-2]
                        break
        except FileNotFoundError:
            source = ""
        rankings.append({
            "slot": slot_num,
            "fitness": state["best_test_fitness"],
            "iterations": state["iteration"],
            "failures": state["consecutive_failures"],
            "setup_type": setup_type,
        })

    rankings.sort(key=lambda x: x["fitness"], reverse=True)

    # Log rankings
    for i, r in enumerate(rankings):
        marker = " ★" if i == 0 else " ✗" if i == len(rankings) - 1 else ""
        logger.info(
            f"  #{i+1} Slot {r['slot']:02d}: fitness={r['fitness']:.4f} "
            f"iters={r['iterations']} type={r['setup_type']}{marker}"
        )

    worst = rankings[-1]
    best = rankings[0]

    # Don't replace if worst has < 3 iterations (give it a chance)
    if worst["iterations"] < 3:
        logger.info(f"Selector: Slot {worst['slot']:02d} only has {worst['iterations']} iterations — keeping")
        return

    # Don't replace if worst is actually decent (positive fitness)
    if worst["fitness"] > 0.005:
        logger.info(f"Selector: Slot {worst['slot']:02d} fitness={worst['fitness']:.4f} is decent — keeping")
        return

    # Ask LLM for replacement decision
    slot_ranking_text = "\n".join(
        f"  Slot {r['slot']:02d}: fitness={r['fitness']:.4f} iters={r['iterations']} "
        f"consecutive_failures={r['failures']} type={r['setup_type']}"
        for r in rankings
    )
    current_types = ", ".join(r["setup_type"] for r in rankings)

    prompt = SELECTOR_PROMPT.format(
        num_slots=NUM_SLOTS,
        slot_rankings=slot_ranking_text,
        worst_slot=f"{worst['slot']:02d}",
        worst_fitness=worst["fitness"],
        worst_iterations=worst["iterations"],
        current_types=current_types,
    )

    from xai_sdk.chat import system as sdk_system, user as sdk_user
    llm = get_grok_llm()
    async with _LLM_SEMAPHORE:
        chat = llm.client.chat.create(
            model=llm.model,
            messages=[sdk_user(prompt)],
            temperature=0.7,
            max_tokens=4096,
        )
        response = await chat.sample()

    usage = response.usage
    get_cost_tracker().log_llm_call(
        model=llm.model,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        purpose="selector_agent",
    )

    # Parse response
    text = response.content.strip()
    # Try to extract JSON
    import re
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        logger.warning("Selector: Could not parse JSON response — skipping")
        return

    try:
        decision = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("Selector: Invalid JSON — skipping")
        return

    action = decision.get("action", "keep")
    reasoning = decision.get("reasoning", "")
    seed_code = decision.get("seed_code")

    logger.info(f"Selector decision: {action} — {reasoning}")

    if action == "keep":
        return

    if not seed_code:
        logger.warning("Selector: No seed_code provided — skipping")
        return

    # Validate the seed code
    validation_err = _validate_strategy_source(seed_code)
    if validation_err:
        logger.warning(f"Selector: Seed code failed validation: {validation_err}")
        return

    # Replace worst slot
    worst_slot = worst["slot"]
    strat_path = _strategy_path(worst_slot)
    strat_path.write_text(seed_code, encoding="utf-8")
    logger.info(f"Selector: Replaced Slot {worst_slot:02d} ({action})")

    # Reset slot state
    slot_states[worst_slot] = {
        "consecutive_failures": 0,
        "best_test_fitness": 0.0,
        "iteration": 0,
    }

    # Git commit the replacement
    await _git_commit(0, True, 0.0, None, slot=worst_slot)


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

async def run_research(*, verbose: bool = False, dry_run: bool = False):
    """Main research loop — all slots evolve concurrently, selector prunes weakest."""
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
    logger.info(f"Selector runs every {SELECTOR_EVERY_N_ROUNDS} rounds")
    logger.info("=" * 60)

    # Per-slot state
    slot_states: dict[int, dict] = {}
    for s in slot_ids:
        slot_states[s] = {
            "consecutive_failures": 0,
            "best_test_fitness": 0.0,
            "iteration": 0,
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

        train_universe, test_universe = _train_test_split(universe)
        train_days = sum(len(d) for d in train_universe.values())
        test_days = sum(len(d) for d in test_universe.values())
        logger.info(f"Split: train={train_days} days, test={test_days} days")

        # Run all slots concurrently
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

        # Selector agent
        if round_num % SELECTOR_EVERY_N_ROUNDS == 0:
            try:
                await _run_selector(slot_states)
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
