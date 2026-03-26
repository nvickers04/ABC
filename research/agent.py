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
from collections import Counter
import json
import logging
import random
import re
import subprocess
import statistics
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
from memory import get_db, get_research_config, set_research_config, get_all_research_config
from research.config import (
    ANALYSIS_PROMPT_TEMPLATE,
    BATCH_SIZE,
    CANDLE_RESOLUTION,
    CIRCUIT_BREAKER_ALL_FAIL_ROUNDS,
    CIRCUIT_BREAKER_COOLDOWN_SECS,
    CIRCUIT_BREAKER_SLOT_MAX_FAILURES,
    DARWINIAN_WEIGHT_CEILING,
    DARWINIAN_WEIGHT_DOWN,
    DARWINIAN_WEIGHT_FLOOR,
    DARWINIAN_WEIGHT_UP,
    EVAL_DAYS_BACK,
    EXTREME_VOL_FITNESS_BOOST,
    MAX_SELECTOR_REPLACEMENTS,
    MIN_KEEP_CONDITION_TRADES,
    MIN_KEEP_CONFIDENCE_SCORE,
    MIN_KEEP_TEST_SIGNALS,
    NUM_SLOTS,
    REGIME_GATE_ENABLED,
    REGIME_TO_SLOTS,
    RESEARCH_DAILY_LLM_BUDGET,
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_UNIVERSE,
    ROUND_DELAY_SECS,
    SANDBOX_ALLOWED_IMPORTS,
    SANDBOX_BLOCKED_CALLS,
    SELECTOR_EVERY_N_ROUNDS,
    SELECTOR_PROMPT,
    SIGNAL_SCHEMA,
    SLOT_CORRELATION_THRESHOLD,
    SLOT_MANDATES,
    validate_legs_json,
)
from research.environment import compute_environment, compute_environment_by_date, format_environment_for_prompt
from research.promoter import OptionsPromoter, score_repriced_signals
from research.simulator import (
    compute_expectancy,
    compute_sample_confidence,
    format_confidence_line,
    simulate,
)
from xai_sdk.chat import system as sdk_system, user as sdk_user

logger = logging.getLogger(__name__)

_SLOTS_DIR = Path(__file__).parent / "slots"
_GIT_LOCK = asyncio.Lock()
_LLM_SEMAPHORE: asyncio.Semaphore | None = None  # initialized in run_research

# ── Feature flags (toggled by __main__.py CLI args) ─────────────
FEATURE_FLAGS: dict[str, bool] = {
    "options_promotion_enabled": True,   # run historical repricing for options signals
    "strict_slippage": True,             # use calibrated slippage from live execution data
    "replay_gating": True,               # require replay pass before promotion
}


# ── Tunable config (DB-backed, refreshed every round) ──────────
# Maps DB key → config.py default.  The agent can update these in the DB;
# _load_tunable_config() reads the latest values each round.

_TUNABLE_DEFAULTS: dict[str, float] = {
    "eval_days_back":              EVAL_DAYS_BACK,
    "min_keep_test_signals":       MIN_KEEP_TEST_SIGNALS,
    "min_keep_confidence_score":   MIN_KEEP_CONFIDENCE_SCORE,
    "min_keep_condition_trades":   MIN_KEEP_CONDITION_TRADES,
    "selector_every_n_rounds":     SELECTOR_EVERY_N_ROUNDS,
    "max_selector_replacements":   MAX_SELECTOR_REPLACEMENTS,
    "darwinian_weight_floor":      DARWINIAN_WEIGHT_FLOOR,
    "darwinian_weight_ceiling":    DARWINIAN_WEIGHT_CEILING,
    "darwinian_weight_up":         DARWINIAN_WEIGHT_UP,
    "darwinian_weight_down":       DARWINIAN_WEIGHT_DOWN,
    "slot_correlation_threshold":  SLOT_CORRELATION_THRESHOLD,
    "extreme_vol_fitness_boost":   EXTREME_VOL_FITNESS_BOOST,
    "regime_gate_enabled":         float(REGIME_GATE_ENABLED),
}


def _load_tunable_config() -> dict[str, float]:
    """Load all tunable config values (DB overrides > config.py defaults)."""
    cfg = dict(_TUNABLE_DEFAULTS)
    for key, default in _TUNABLE_DEFAULTS.items():
        cfg[key] = get_research_config(key, default)
    return cfg


def _format_mandate(slot: int) -> str:
    """Format the slot mandate text for the LLM system prompt."""
    mandate = SLOT_MANDATES.get(slot)
    if not mandate:
        return "(No mandate assigned — evolve freely.)"
    allowed = ", ".join(sorted(mandate["allowed_order_types"]))
    return (
        f"Regime group: {mandate['regime'].upper()} | "
        f"Direction: {mandate['direction']} | "
        f"Order class: {mandate['order_class']}\n"
        f"Allowed order_type values: {allowed}\n"
        f"Role: {mandate['description']}"
    )


def _check_mandate_compliance(slot: int, source: str, signals: list[dict]) -> str | None:
    """Check if a strategy's signals comply with its slot mandate.

    Returns an error string if non-compliant, or None if OK.
    """
    mandate = SLOT_MANDATES.get(slot)
    if not mandate:
        return None  # no mandate = no constraint

    # Direction check (skip for "any" or "neutral")
    required_dir = mandate["direction"]
    if required_dir in ("short", "long") and signals:
        wrong_dir = [s for s in signals if s.get("direction") != required_dir]
        if len(wrong_dir) > len(signals) * 0.2:  # allow up to 20% hedging
            pct = len(wrong_dir) / len(signals) * 100
            return (
                f"mandate violation: {pct:.0f}% signals are "
                f"{wrong_dir[0].get('direction', '?')}, mandate requires {required_dir}"
            )

    # Order type check
    allowed = mandate["allowed_order_types"]
    if signals:
        order_types = {s.get("order_type") for s in signals}
        violations = order_types - allowed
        if violations:
            return (
                f"mandate violation: order_type {violations} not in "
                f"allowed set {sorted(allowed)}"
            )

    return None


def _get_active_regime(env_snapshot: dict | None) -> str:
    """Determine the current market regime label from the environment snapshot."""
    if not env_snapshot:
        return "flat"
    return env_snapshot.get("trend_regime", "flat")


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


async def _fetch_universe_candles(
    days_back: int, *, exclude_today: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch candles for all symbols, split by trading day.

    Uses async get_candles_async to avoid nest_asyncio event-loop corruption
    when called from within the already-running event loop.

    Returns: {symbol: {date_str: df}}
    """
    provider = get_data_provider()
    universe: dict[str, dict[str, pd.DataFrame]] = {}

    # Use explicit date range so intraday resolution gets full days, not N bars
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=days_back + 4)).isoformat()  # +4 for weekends

    for sym in RESEARCH_UNIVERSE:
        try:
            raw = await provider.get_candles_async(
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
    """Split universe into train/test by date.  Chronological — no leakage.

    Uses rolling walk-forward: train on the earliest train_ratio of days,
    test on the remainder.  The split walks forward each call relative to the
    data window, ensuring out-of-sample integrity.
    """
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


def walk_forward_splits(
    universe: dict[str, dict[str, pd.DataFrame]],
    train_days: int = 7,
    test_days: int = 1,
) -> list[tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]]]]:
    """Generate rolling walk-forward train/test splits.

    Yields (train_universe, test_universe) pairs where each window advances
    by test_days.  E.g. with 10 days: train 1-7/test 8, train 2-8/test 9,
    train 3-9/test 10.
    """
    all_dates = sorted({d for days in universe.values() for d in days})
    window = train_days + test_days
    if len(all_dates) < window:
        # Not enough data — fall back to single split
        return [_train_test_split(universe)]

    splits = []
    for start in range(0, len(all_dates) - window + 1, test_days):
        train_set = set(all_dates[start:start + train_days])
        test_set = set(all_dates[start + train_days:start + window])

        train: dict[str, dict[str, pd.DataFrame]] = {}
        test: dict[str, dict[str, pd.DataFrame]] = {}
        for sym, days in universe.items():
            tr = {d: df for d, df in days.items() if d in train_set}
            te = {d: df for d, df in days.items() if d in test_set}
            if tr:
                train[sym] = tr
            if te:
                test[sym] = te
        if train and test:
            splits.append((train, test))

    return splits if splits else [_train_test_split(universe)]


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


def _execute_strategy(
    source: str,
    candles: pd.DataFrame,
    symbol: str,
    env: dict | None = None,
) -> list[dict]:
    """Execute strategy.scan() in sandbox. Returns signals or empty list.

    If the strategy's scan() accepts a third `env` parameter it receives
    a dict with regime/environment data (volatility_regime, trend_regime, etc.).
    Strategies that only accept (candles, symbol) continue to work unchanged.
    """
    try:
        mod = _load_strategy_module(source)
        scan_fn = getattr(mod, "scan", None)
        if scan_fn is None:
            return []
        # Try 3-arg call first (candles, symbol, env); fall back to 2-arg
        if env is not None:
            try:
                signals = scan_fn(candles, symbol, env)
            except TypeError:
                signals = scan_fn(candles, symbol)
        else:
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
    slippage = "strict" if FEATURE_FLAGS.get("strict_slippage") else "fast"
    env_by_date = compute_environment_by_date(universe)

    for sym, days in universe.items():
        for day_str, day_df in days.items():
            day_env = env_by_date.get(day_str, {})
            signals = _execute_strategy(source, day_df, sym, env=day_env)
            if not signals:
                continue
            # Deduplicate overlapping signals per symbol/day
            signals = _deduplicate_signals(signals)
            sim_results = simulate(signals, day_df, slippage_preset=slippage)
            if not sim_results:
                continue

            # Tag with symbol and date
            for r in sim_results:
                r["symbol"] = sym
                r["eval_date"] = day_str
                r["env_key"] = day_env.get("env_key", "env=unknown")
                r["volatility_regime"] = day_env.get("volatility_regime", "unknown")
                r["trend_regime"] = day_env.get("trend_regime", "unknown")
                r["breadth_regime"] = day_env.get("breadth_regime", "unknown")
                r["momentum_regime"] = day_env.get("momentum_regime", "unknown")
            all_results.extend(sim_results)

            day_stats = compute_expectancy(sim_results)
            day_stats["symbol"] = sym
            day_stats["eval_date"] = day_str
            day_stats["env_key"] = day_env.get("env_key", "env=unknown")
            per_day.append(day_stats)

    aggregate = compute_expectancy(all_results)
    return aggregate, per_day, all_results


# ═══════════════════════════════════════════════════════════════
# LLM INTERACTION
# ═══════════════════════════════════════════════════════════════


async def _call_llm(
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    purpose: str = "research",
    timeout_seconds: float = 120,
):
    """Unified LLM call helper to reduce duplication across analyze/propose/selector.

    Centralizes client setup, async sampling, and cost tracking.
    Returns response.content (str) for compatibility with callers.
    """
    llm = get_grok_llm()
    chat = llm.client.chat.create(
        model=llm.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    async with asyncio.timeout(timeout_seconds):
        response = await chat.sample()

    # Track cost
    usage = response.usage
    get_cost_tracker().log_llm_call(
        model=llm.model,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        purpose=purpose,
    )

    return response.content


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
            f"exp={d.get('expectancy',0):.4f}, cond={d.get('env_key','unknown')}"
        )

    condition_metrics = aggregate.get("condition_metrics", []) or []
    if condition_metrics:
        condition_summary = "\n".join(
            (
                f"  {c.get('env_key','')} — trades={c.get('trades',0)}, "
                f"exp={c.get('expectancy',0):+.4f}, "
                f"ci=[{c.get('ci95_low',0):+.4f}, {c.get('ci95_high',0):+.4f}], "
                f"conf={c.get('confidence_score',0):.2f}, "
                f"luck={c.get('luck_pressure',0):.2f}"
            )
            for c in condition_metrics
        )
    else:
        condition_summary = "  (insufficient condition data)"

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
        ci95_low=aggregate.get("ci95_low", 0),
        ci95_high=aggregate.get("ci95_high", 0),
        confidence_score=aggregate.get("confidence_score", 0),
        luck_pressure=aggregate.get("luck_pressure", 1),
        condition_confidence=aggregate.get("condition_confidence", 0),
        condition_summary=condition_summary,
    )

    messages = [sdk_user(prompt)]
    return await _call_llm(
        messages,
        temperature=0.3,
        max_tokens=2048,
        purpose="research_analysis",
    )


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
        slot_mandate=_format_mandate(slot),
    )

    messages = [
        sdk_system(system_prompt),
        sdk_user(prompt),
    ]
    content = await _call_llm(
        messages,
        temperature=temperature,
        max_tokens=4096,
        purpose="research_proposal",
    )

    # Strip markdown fences if the LLM wraps them anyway
    text = content.strip()
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
            avg_intraday_range_pct, avg_cumulative_return, advance_decline_ratio,
            avg_momentum_shift, avg_volume_ratio, pct_trending_up, pct_trending_down,
            strategy_fit_json, raw_snapshot_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            snapshot.get("avg_intraday_range_pct"),
            snapshot.get("avg_cumulative_return"),
            snapshot.get("advance_decline_ratio"),
            snapshot.get("avg_momentum_shift"),
            snapshot.get("avg_volume_ratio"),
            snapshot.get("pct_trending_up"),
            snapshot.get("pct_trending_down"),
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
    # Range-based binning: group by continuous metric bands instead of labels.
    # ATR bands: <1.5 low, 1.5-2.5 normal, 2.5-3.5 high, >3.5 extreme
    # A/D bands: <-0.2 bearish, -0.2-0.15 mixed, 0.15-0.35 neutral, >0.35 bullish
    rows = db.execute(
        """SELECT
               s.strategy_type,
               CASE
                   WHEN e.avg_atr_pct < 1.5 THEN 'low'
                   WHEN e.avg_atr_pct < 2.5 THEN 'normal'
                   WHEN e.avg_atr_pct < 3.5 THEN 'high'
                   ELSE 'extreme'
               END as vol_band,
               CASE
                   WHEN e.advance_decline_ratio < -0.2 THEN 'bearish'
                   WHEN e.advance_decline_ratio < 0.15 THEN 'mixed'
                   WHEN e.advance_decline_ratio < 0.35 THEN 'neutral'
                   ELSE 'bullish'
               END as breadth_band,
               AVG(s.fitness) as avg_fitness,
               AVG(s.expectancy) as avg_exp,
               AVG(e.avg_atr_pct) as avg_atr,
               AVG(e.advance_decline_ratio) as avg_ad,
               COUNT(*) as observations
           FROM slot_environment_scores s
           JOIN environment_snapshots e ON s.env_snapshot_id = e.id
           WHERE e.avg_atr_pct IS NOT NULL
           GROUP BY vol_band, breadth_band, s.strategy_type
           HAVING observations >= 2
           ORDER BY avg_fitness DESC
           LIMIT 20"""
    ).fetchall()

    if not rows:
        # Fall back to label-based grouping for older data without continuous cols
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
                f"→ avg_fitness={r['avg_fitness'] or 0:.4f} avg_exp={r['avg_exp'] or 0:.4f} "
                f"(n={r['observations']})"
            )
        return "\n".join(lines)

    lines = []
    for r in rows:
        lines.append(
            f"  {r['strategy_type']:<20s} vol_band={r['vol_band']:<7s} (ATR~{r['avg_atr'] or 0:.1f}%) "
            f"breadth={r['breadth_band']:<8s} (A/D~{r['avg_ad'] or 0:+.2f}) "
            f"→ avg_fitness={r['avg_fitness'] or 0:.4f} avg_exp={r['avg_exp'] or 0:.4f} "
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
        sim_returns = [t["simulated_return"] for t in with_feedback if t["simulated_return"] is not None]
        actual_pnls = [t["pnl"] for t in with_feedback if t["pnl"] is not None]
        gaps = [t["execution_gap"] for t in with_feedback if t["execution_gap"] is not None]
        lines.append(f"  Trades with feedback: {len(with_feedback)}")
        lines.append(f"  Avg simulated return: {avg_sim:+.2f}%")
        lines.append(f"  Avg actual PnL: ${avg_actual:+.2f}")
        lines.append(f"  Avg execution gap: {avg_gap:+.2f}")
        for line in (
            format_confidence_line("Simulated return", sim_returns, kind="pct"),
            format_confidence_line("Actual P&L", actual_pnls, kind="usd"),
            format_confidence_line("Execution gap metric", gaps),
        ):
            if line:
                lines.append(line)

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
        # Find closest live signal for this symbol + slot within 30 min window
        signal = db.execute(
            """SELECT ls.id, ls.slot, ls.direction, ls.entry_price,
                      s.return_pct as sim_return, s.strategy_id
               FROM live_signals ls
               LEFT JOIN signals s ON s.strategy_id = ls.strategy_id
                   AND s.symbol = ls.symbol
               WHERE ls.symbol = ?
                 AND ABS(julianday(ls.ts) - julianday(?)) < (30.0 / 1440.0)
               ORDER BY ABS(julianday(ls.ts) - julianday(?)) ASC
               LIMIT 1""",
            (trade["symbol"], trade["ts"], trade["ts"]),
        ).fetchone()

        if signal:
            sim_return = signal["sim_return"] or 0  # percentage
            actual_pnl = trade["pnl"] or 0  # dollars
            entry_price = signal["entry_price"]

            # Convert dollar PnL to percentage return for comparable gap.
            # If entry_price is missing, store gap as None (unknown).
            if entry_price and entry_price > 0:
                actual_return_pct = (actual_pnl / entry_price) * 100
                execution_gap = actual_return_pct - sim_return
            else:
                actual_return_pct = None
                execution_gap = None

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


def _store_promotion_run(
    slot: int,
    strategy_id: Optional[int],
    eval_type: str,
    search_fitness: Optional[float],
    promotion_score: Optional[dict],
    promoted: bool,
    rejection_reason: Optional[str] = None,
) -> None:
    """Persist a promotion evaluation record to the promotion_runs table."""
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    promotion_fitness = promotion_score.get("promotion_fitness") if promotion_score else None
    options_coverage_pct = promotion_score.get("options_coverage_pct") if promotion_score else None
    db.execute(
        """INSERT INTO promotion_runs
           (ts, slot, strategy_id, eval_type, search_fitness, promotion_fitness,
            options_coverage_pct, promoted, rejection_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now, slot, strategy_id, eval_type,
            search_fitness, promotion_fitness, options_coverage_pct,
            1 if promoted else 0, rejection_reason,
        ),
    )
    db.commit()


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


def _compute_fitness(agg: dict, *, slot: int | None = None) -> float:
    """Use the simulator's uncertainty-adjusted search fitness when available.

    When *slot* is provided, applies a hard penalty based on real execution
    gaps from the trade_feedback table: if the slot's recent trades have a
    consistently negative gap (sim over-estimates reality), fitness is
    penalised proportionally.  This makes the feedback loop numeric rather
    than relying on the LLM to interpret text feedback.
    """
    if "search_fitness" in agg:
        raw = float(agg.get("search_fitness", 0.0) or 0.0)
    else:
        exp = agg.get("expectancy", 0)
        signals = agg.get("total_signals", 0)
        sharpe = agg.get("sharpe_approx", 0)
        signal_factor = min(1.0, signals / 100)
        sharpe_factor = max(0.5, min(1.5, sharpe / 1.0)) if sharpe != 0 else 0.75
        raw = exp * signal_factor * sharpe_factor

    if slot is not None and raw > 0:
        penalty = _execution_gap_penalty(slot)
        if penalty > 0:
            raw *= (1.0 - penalty)
    return raw


# Minimum number of feedback trades before we apply the gap penalty
_MIN_GAP_TRADES = 5
# Per-percentage-point penalty cap (e.g. avg gap of -3% → 3 * 0.05 = 15% penalty)
_GAP_PENALTY_PER_PCT = 0.05
_GAP_PENALTY_CAP = 0.40  # never penalise more than 40%


def _execution_gap_penalty(slot: int) -> float:
    """Return a 0–_GAP_PENALTY_CAP penalty factor for *slot* based on trade feedback.

    Penalty is zero when there are fewer than _MIN_GAP_TRADES feedback rows or
    when the average gap is non-negative (real execution >= simulation).
    """
    try:
        db = get_db()
        rows = db.execute(
            """SELECT execution_gap FROM trade_feedback
               WHERE slot = ? AND execution_gap IS NOT NULL
               ORDER BY ts DESC LIMIT 20""",
            (slot,),
        ).fetchall()
        if len(rows) < _MIN_GAP_TRADES:
            return 0.0
        avg_gap = sum(r["execution_gap"] for r in rows) / len(rows)
        if avg_gap >= 0:
            return 0.0  # real execution meets or beats sim — no penalty
        # avg_gap is negative; convert to positive penalty
        penalty = min(_GAP_PENALTY_CAP, abs(avg_gap) * _GAP_PENALTY_PER_PCT)
        return penalty
    except Exception:
        return 0.0


def _format_fitness_metrics(agg: dict) -> str:
    """Format clamped and raw search fitness so negative edges remain visible."""
    fitness = _compute_fitness(agg)
    raw_fitness = float(agg.get("raw_search_fitness", fitness) or 0.0)
    signed_edge = float(agg.get("signed_edge_score", raw_fitness) or 0.0)
    expectancy = float(agg.get("expectancy", 0.0) or 0.0)
    if abs(raw_fitness - fitness) >= 0.00005 or (fitness == 0.0 and expectancy != 0.0):
        return f"fitness={fitness:.4f} raw={raw_fitness:.6f} edge={signed_edge:.6f}"
    return f"fitness={fitness:.4f}"


def _max_condition_trades(agg: dict) -> int:
    """Return the largest repeated environment bucket size in an aggregate result."""
    condition_rows = agg.get("condition_metrics", []) or []
    return max((int(row.get("trades", 0) or 0) for row in condition_rows), default=0)


# ═══════════════════════════════════════════════════════════════
# DARWINIAN WEIGHTING
# ═══════════════════════════════════════════════════════════════

def _update_darwinian_weights(
    slot_states: dict[int, dict], slot_ids: list[int],
    cfg: dict[str, float] | None = None,
) -> None:
    """Update Darwinian weights: top quartile get louder, bottom quartile get quieter."""
    ceil = (cfg or {}).get("darwinian_weight_ceiling", DARWINIAN_WEIGHT_CEILING)
    floor = (cfg or {}).get("darwinian_weight_floor", DARWINIAN_WEIGHT_FLOOR)
    up = (cfg or {}).get("darwinian_weight_up", DARWINIAN_WEIGHT_UP)
    down = (cfg or {}).get("darwinian_weight_down", DARWINIAN_WEIGHT_DOWN)

    ranked = sorted(slot_ids, key=lambda s: slot_states[s]["best_test_fitness"], reverse=True)
    n = len(ranked)
    top_cutoff = max(1, n // 4)
    bottom_cutoff = max(1, n // 4)
    top_quartile = set(ranked[:top_cutoff])
    bottom_quartile = set(ranked[-bottom_cutoff:])

    for s in slot_ids:
        w = slot_states[s].get("darwinian_weight", 1.0)
        if s in top_quartile:
            w = min(ceil, w * up)
        elif s in bottom_quartile:
            w = max(floor, w * down)
        slot_states[s]["darwinian_weight"] = round(w, 4)


# ═══════════════════════════════════════════════════════════════
# CROSS-SLOT CORRELATION
# ═══════════════════════════════════════════════════════════════

def _pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length sequences."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / max(n - 1, 1)) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / max(n - 1, 1)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    return round(cov / (sx * sy), 4)


def _compute_slot_correlations(
    slot_states: dict[int, dict],
    cfg: dict[str, float] | None = None,
) -> dict:
    """Cross-slot return correlation from daily return profiles.

    Each slot's state['_daily_returns'] maps {date_str: avg_return}.
    We compute pairwise Pearson r on shared dates and flag redundant pairs.
    """
    slot_returns: dict[int, dict[str, float]] = {}
    for slot, state in slot_states.items():
        dr = state.get("_daily_returns", {})
        if dr:
            slot_returns[slot] = dr

    slots = sorted(slot_returns.keys())
    if len(slots) < 2:
        return {"pairs": [], "avg_correlation": 0.0, "redundant_pairs": []}

    pairs = []
    redundant = []
    for i, s1 in enumerate(slots):
        for s2 in slots[i + 1:]:
            shared_dates = sorted(set(slot_returns[s1]) & set(slot_returns[s2]))
            if len(shared_dates) < 10:
                continue
            r1 = [slot_returns[s1][d] for d in shared_dates]
            r2 = [slot_returns[s2][d] for d in shared_dates]
            corr = _pearson_r(r1, r2)
            pairs.append({"slots": (s1, s2), "correlation": corr, "shared": len(shared_dates)})
            # Only positive correlation = redundancy; negative = diversification
            corr_thresh = (cfg or {}).get("slot_correlation_threshold", SLOT_CORRELATION_THRESHOLD)
            if corr > corr_thresh:
                redundant.append((s1, s2, round(corr, 2)))

    avg = (sum(p["correlation"] for p in pairs) / len(pairs)) if pairs else 0.0
    return {"pairs": pairs, "avg_correlation": round(avg, 4), "redundant_pairs": redundant}





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

async def _fetch_live_candles() -> dict[str, pd.DataFrame]:
    """Fetch today's 1-min candles for the full universe (called once per round).

    Returns an empty dict immediately when the regular market session is not
    open — avoids flooding the API with 404s every round during pre-market.

    Uses async get_candles_async to avoid nest_asyncio event-loop corruption.
    """
    from data.market_hours import get_market_hours_provider
    if not get_market_hours_provider().is_market_open():
        logger.debug("Live candle fetch skipped \u2014 market not open")
        return {}

    provider = get_data_provider()
    today = date.today().isoformat()
    result: dict[str, pd.DataFrame] = {}
    for sym in RESEARCH_UNIVERSE:
        try:
            raw = await provider.get_candles_async(
                sym, resolution=CANDLE_RESOLUTION, from_date=today,
            )
            if raw and len(raw) > 0:
                df = _candles_to_df(raw)
                if len(df) >= 30:
                    result[sym] = df
        except Exception as e:
            logger.warning(f"Live candle fetch failed for {sym}: {e}")
    logger.info(f"Live candles: {len(result)}/{len(RESEARCH_UNIVERSE)} symbols with >= 30 bars")
    return result


def _run_live_scan(
    strategy_id: int, source: str, slot: int,
    live_candles: dict[str, pd.DataFrame],
):
    """Run best strategy against pre-fetched today's candles and write live_signals."""
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Prune live signals older than 30 days; keep recent ones for feedback
    db.execute(
        "DELETE FROM live_signals WHERE slot = ? AND ts < datetime('now', '-30 days')",
        (slot,),
    )

    for sym, df in live_candles.items():
        try:
            signals = _execute_strategy(source, df, sym)
            for sig in signals:
                # Validate legs_json before inserting (same check simulator uses)
                legs = sig.get("legs_json")
                if legs:
                    legs_err = validate_legs_json(legs)
                    if legs_err:
                        logger.debug(f"Live scan [{sym}]: skipping signal with invalid legs_json: {legs_err}")
                        continue
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
                        json.dumps(legs) if legs else None,
                    ),
                )
        except Exception as e:
            logger.warning(f"Live scan failed for {sym}: {e}")

    db.commit()
    count = db.execute("SELECT COUNT(*) FROM live_signals WHERE slot = ?", (slot,)).fetchone()[0]
    logger.info(f"Live scan [slot {slot:02d}]: {count} signals written for {len(live_candles)} symbols")


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
    env_snapshot: dict | None = None,
    live_candles: dict[str, pd.DataFrame] | None = None,
    cfg: dict[str, float] | None = None,
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
        f"pf={train_agg['profit_factor']:.2f} "
        f"ci=[{train_agg.get('ci95_low', 0):+.4f},{train_agg.get('ci95_high', 0):+.4f}] "
        f"conf={train_agg.get('confidence_score', 0):.2f} "
        f"luck={train_agg.get('luck_pressure', 1):.2f}"
    )

    # ── 3. Evaluate current strategy on TEST set ────────────────
    test_agg, test_per_day, test_results = await asyncio.to_thread(
        _evaluate_strategy, current_source, test_universe,
    )
    best_test_fitness = _compute_fitness(test_agg, slot=slot)
    gap_pen = _execution_gap_penalty(slot)
    # Clamp historical best to fitness cap (old uncapped values may be inflated)
    state["best_test_fitness"] = min(state["best_test_fitness"], 5.0)
    state["best_test_fitness"] = max(state["best_test_fitness"], best_test_fitness)
    logger.info(
        f"[Slot {slot:02d}] Test:  signals={test_agg['total_signals']} "
        f"hit={test_agg['hit_rate']:.0f}% exp={test_agg['expectancy']:.4f} "
        f"{_format_fitness_metrics(test_agg)} "
        f"conf={test_agg.get('confidence_score', 0):.2f} "
        f"cond_conf={test_agg.get('condition_confidence', 0):.2f}"
        + (f" gap_penalty={gap_pen:.2%}" if gap_pen > 0 else "")
    )

    # Store daily return profile for cross-slot correlation analysis
    _daily_rets: dict[str, list[float]] = {}
    for _r in test_results:
        _d = _r.get("eval_date", "")
        if _d:
            _daily_rets.setdefault(_d, []).append(_r.get("return_pct", 0.0))
    state["_daily_returns"] = {
        d: sum(v) / len(v) for d, v in _daily_rets.items() if v
    }

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

    # Gather live-trading hypotheses so the LLM can address real execution gaps
    hypothesis_text = ""
    try:
        from memory import get_open_hypotheses
        open_hyps = get_open_hypotheses(slot=slot, limit=5)
        if open_hyps:
            hyp_lines = []
            for h in open_hyps:
                hyp_lines.append(
                    f"  [{h['hypothesis_type']}] {h['description']} "
                    f"→ {h['suggested_action'] or 'no action suggested'}"
                )
            hypothesis_text = (
                "\n\nLIVE TRADING FEEDBACK (from real execution):\n"
                + "\n".join(hyp_lines)
                + "\nConsider these observations when improving the strategy."
            )
    except Exception as _hyp_err:
        logger.debug(f"[Slot {slot:02d}] Hypothesis fetch failed: {_hyp_err}")

    # Check for selector directive (meta-learning pivot)
    directive = state.pop("selector_directive", None)
    directive_text = ""
    if directive:
        target = directive.get("target_strategy_type", "")
        dreason = directive.get("reason", "")
        directive_text = (
            f"\n\nSELECTOR DIRECTIVE: The meta-learning selector has requested this slot "
            f"pivot to a '{target}' strategy. Reason: {dreason}\n"
            f"You MUST produce a strategy of type '{target}'. Use the current code as "
            f"a starting point but fundamentally change the approach to match this type."
        )
        proposal_temp = min(1.0, proposal_temp + 0.15)  # higher temp for pivots
        logger.info(
            f"[Slot {slot:02d}] Selector directive: pivot to {target} (temp={proposal_temp:.2f})"
        )

    logger.info(
        f"[Slot {slot:02d}] Proposing new strategy "
        f"(temp={proposal_temp:.2f}, failures={consecutive_failures})..."
    )
    async with _LLM_SEMAPHORE:
        new_source = await _llm_propose_strategy(
            current_source, analysis + directive_text + hypothesis_text, history,
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
        f"pf={new_train_agg['profit_factor']:.2f} "
        f"ci=[{new_train_agg.get('ci95_low', 0):+.4f},{new_train_agg.get('ci95_high', 0):+.4f}] "
        f"conf={new_train_agg.get('confidence_score', 0):.2f}"
    )

    # ── 9. Evaluate proposed on TEST ────────────────────────────
    new_test_agg, new_test_pd, new_test_res = await asyncio.to_thread(
        _evaluate_strategy, new_source, test_universe,
    )
    new_test_fitness = _compute_fitness(new_test_agg, slot=slot)
    logger.info(
        f"[Slot {slot:02d}] Proposed test:  signals={new_test_agg['total_signals']} "
        f"hit={new_test_agg['hit_rate']:.0f}% exp={new_test_agg['expectancy']:.4f} "
        f"{_format_fitness_metrics(new_test_agg)} "
        f"conf={new_test_agg.get('confidence_score', 0):.2f} "
        f"cond_conf={new_test_agg.get('condition_confidence', 0):.2f} "
        f"luck={new_test_agg.get('luck_pressure', 1):.2f}"
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

    # ── 10b. Options promotion repricing (if strategy uses options legs) ──
    # Run historical leg-level repricing for signals that have legs_json.
    # This is the promotion-grade check that replaces Greek approximations.
    promotion_score: dict | None = None
    options_signals = [s for s in new_test_res if s.get("legs_json") and isinstance(s.get("legs_json"), dict)]
    if options_signals and FEATURE_FLAGS.get("options_promotion_enabled", True):
        try:
            promoter = OptionsPromoter(get_data_provider())
            repriced = []
            # Group by trade date (eval_date is YYYY-MM-DD from _evaluate_strategy)
            date_groups: dict[str, list[dict]] = {}
            for sig in options_signals:
                d = sig.get("eval_date", "unknown")
                date_groups.setdefault(d, []).append(sig)

            for trade_date, day_sigs in date_groups.items():
                if trade_date == "unknown":
                    continue
                day_repriced = await asyncio.to_thread(
                    promoter.reprice_signals, day_sigs, trade_date
                )
                repriced.extend(day_repriced)

            if repriced:
                promotion_score = score_repriced_signals(repriced)
                n = promotion_score['repriced_count']
                total = promotion_score['total_count']
                cov = promotion_score['options_coverage_pct']
                if n == 0:
                    breakdown = promotion_score.get('missing_breakdown_text', 'no repriced contracts')
                    logger.info(
                        f"[Slot {slot:02d}] Promotion repricing: {total} options signals — "
                        f"0 repriced ({breakdown}, not penalised)"
                    )
                else:
                    logger.info(
                        f"[Slot {slot:02d}] Promotion repricing: "
                        f"{n}/{total} repriced, coverage={cov:.0f}%, "
                        f"fitness={promotion_score['promotion_fitness']:.4f}"
                    )
        except Exception as _promo_exc:
            logger.debug(f"[Slot {slot:02d}] Promotion repricing skipped: {_promo_exc}")

    # ── 10c. Replay gating (deterministic historical replay) ─────
    replay_episode = None
    if FEATURE_FLAGS.get("replay_gating", False) and new_test_res:
        try:
            from research.replay import ReplayHarness, store_replay_episode
            harness = ReplayHarness(
                slot=slot,
                session_date=date.today().isoformat(),
                min_pnl=0.0,
                min_signals=1,
            )
            replay_episode = await harness.run(new_test_res)
            store_replay_episode(replay_episode)
            logger.info(
                f"[Slot {slot:02d}] Replay episode: outcome={replay_episode.outcome} "
                f"fills={replay_episode.total_fills} pnl={replay_episode.total_pnl:.4f}"
            )
        except Exception as _replay_exc:
            logger.debug(f"[Slot {slot:02d}] Replay gating skipped: {_replay_exc}")

    # ── 11. Keep or discard — single composite score ───────────
    # Karpathy-style: ONE composite fitness decides everything.
    # The simulator's search_fitness already bakes in expectancy, profit
    # factor, sample size, drawdown, stability, confidence, and condition
    # confidence.  Only mandate compliance remains as a structural gate.
    best_fitness = state["best_test_fitness"]

    # Mandate compliance: only structural gate (slot direction must match)
    mandate_violation = _check_mandate_compliance(slot, new_source, new_test_res)
    mandate_blocked = mandate_violation is not None
    if mandate_blocked:
        logger.info(f"[Slot {slot:02d}] MANDATE GATE: {mandate_violation}")

    # Simple composite comparison
    strict_improved = new_test_fitness > best_fitness and not mandate_blocked

    # Annealing: after consecutive failures, accept near-matches
    annealing_accept = (
        not strict_improved
        and not mandate_blocked
        and consecutive_failures >= 5
        and best_fitness > 0
        and new_test_fitness >= best_fitness * 0.85
    )

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
        # Record promotion run (only when promotion actually ran)
        if promotion_score is not None:
            _store_promotion_run(
                slot=slot, strategy_id=new_id, eval_type="fast_search",
                search_fitness=new_test_fitness,
                promotion_score=promotion_score, promoted=True,
            )
        if not dry_run:
            strat_path.write_text(new_source, encoding="utf-8")
            logger.info(f"[Slot {slot:02d}] Strategy updated (id={new_id})")
        best_id = new_id
        state["best_test_fitness"] = new_test_fitness
        state["consecutive_failures"] = 0
        state["last_expectancy"] = new_train_agg.get("expectancy", 0)
        state["last_total_signals"] = new_train_agg.get("total_signals", 0)
        state["last_kept"] = True
        state["last_rejection_reasons"] = []
    else:
        reasons = []
        if mandate_blocked:
            reasons.append(mandate_violation)
        if new_test_fitness <= best_fitness:
            reasons.append(f"lower fitness ({new_test_fitness:.4f} vs {best_fitness:.4f})")

        logger.info(f"[Slot {slot:02d}] DISCARDING: {', '.join(reasons)}")
        new_id = _store_strategy(
            new_source, new_train_agg, new_train_pd, new_train_res, new_analysis,
            kept=False, parent_id=current_id, slot=slot,
        )
        if promotion_score is not None:
            _store_promotion_run(
                slot=slot, strategy_id=new_id, eval_type="fast_search",
                search_fitness=new_test_fitness,
                promotion_score=promotion_score, promoted=False,
                rejection_reason=", ".join(reasons),
            )
        state["consecutive_failures"] += 1
        state["last_kept"] = False
        state["last_rejection_reasons"] = reasons

    # ── 12. Live scan (gate on positive fitness + regime match) ──
    active_regime = _get_active_regime(env_snapshot)
    slot_regime_slots = REGIME_TO_SLOTS.get(active_regime, [])
    regime_match = slot in slot_regime_slots
    if best_id is not None and live_candles and state["best_test_fitness"] > 0:
        if regime_match:
            best_source = strat_path.read_text(encoding="utf-8") if not dry_run else current_source
            await asyncio.to_thread(_run_live_scan, best_id, best_source, slot, live_candles)
        else:
            mandate = SLOT_MANDATES.get(slot, {})
            logger.info(
                f"[Slot {slot:02d}] Live scan SKIPPED: slot regime "
                f"'{mandate.get('regime', '?')}' != active '{active_regime}'"
            )

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
    train_universe: dict,
    test_universe: dict,
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
            "darwinian_weight": state.get("darwinian_weight", 1.0),
            # Composite: raw fitness weighted by environment fit and Darwinian weight
            "env_adjusted_fitness": (
                state["best_test_fitness"]
                * (0.5 + 0.5 * env_fit)
                * state.get("darwinian_weight", 1.0)
            ),
        })

    rankings.sort(key=lambda x: x["env_adjusted_fitness"], reverse=True)

    # ── Diversity scoring ───────────────────────────────────────
    # Detect intra-regime type duplication and penalise converged slots.
    regime_type_counts: dict[str, dict[str, list[int]]] = {}
    for r in rankings:
        m = SLOT_MANDATES.get(r["slot"], {})
        regime = m.get("regime", "unknown")
        stype = r["strategy_type"]
        regime_type_counts.setdefault(regime, {}).setdefault(stype, []).append(r["slot"])

    diversity_issues: list[str] = []
    for regime, type_map in regime_type_counts.items():
        for stype, slots in type_map.items():
            if len(slots) > 1:
                diversity_issues.append(
                    f"{regime} regime: slots {slots} share type '{stype}'"
                )
                # Apply a 10% fitness penalty to non-best duplicates
                best_dup = max(slots, key=lambda s: next(
                    r["env_adjusted_fitness"] for r in rankings if r["slot"] == s
                ))
                for s in slots:
                    if s != best_dup:
                        for r in rankings:
                            if r["slot"] == s:
                                r["env_adjusted_fitness"] *= 0.90
                                r["diversity_penalty"] = True
                                break

    if diversity_issues:
        logger.info(f"  Diversity issues: {'; '.join(diversity_issues)}")
    rankings.sort(key=lambda x: x["env_adjusted_fitness"], reverse=True)

    # Log rankings
    for i, r in enumerate(rankings):
        marker = " ★" if i == 0 else " ✗" if i >= len(rankings) - 2 else ""
        dp = " [dup]" if r.get("diversity_penalty") else ""
        logger.info(
            f"  #{i+1} Slot {r['slot']:02d}: fitness={r['fitness']:.4f} "
            f"env_fit={r['env_fit']:.2f} adj={r['env_adjusted_fitness']:.4f} "
            f"iters={r['iterations']} type={r['strategy_type']}{marker}{dp}"
        )

    # ── Get historical and feedback context ─────────────────────
    env_slot_history = _get_env_slot_history()
    trade_feedback = _get_trade_feedback()
    environment_text = format_environment_for_prompt(env_snapshot)

    # ── Build prompt for meta-learning selector ─────────────────
    # Build slot ranking text with correlation data
    corr_data = _compute_slot_correlations(slot_states)
    slot_ranking_text = "\n".join(
        f"  Slot {r['slot']:02d}: fitness={r['fitness']:.4f} "
        f"weight={r['darwinian_weight']:.2f} "
        f"env_fit={r['env_fit']:.2f} adj_fitness={r['env_adjusted_fitness']:.4f} "
        f"iters={r['iterations']} failures={r['failures']} type={r['strategy_type']}"
        for r in rankings
    )
    if corr_data["redundant_pairs"]:
        slot_ranking_text += "\n\n  REDUNDANT PAIRS (correlation > 0.7):"
        for _s1, _s2, _r in corr_data["redundant_pairs"]:
            slot_ranking_text += f"\n    Slots {_s1:02d} & {_s2:02d}: r={_r:.2f}"

    if diversity_issues:
        slot_ranking_text += "\n\n  DIVERSITY VIOLATIONS (same strategy type within regime):"
        for issue in diversity_issues:
            slot_ranking_text += f"\n    {issue}"
        slot_ranking_text += (
            "\n  ACTION: Consider issuing directives to duplicate slots to pivot to "
            "their mandated order_class. Diversity across order types within a regime "
            "is critical for portfolio resilience."
        )

    # Build mandate summary for selector
    mandate_lines = []
    for s in range(1, NUM_SLOTS + 1):
        m = SLOT_MANDATES.get(s)
        if m:
            allowed = ", ".join(sorted(m["allowed_order_types"]))
            mandate_lines.append(
                f"  Slot {s:02d}: regime={m['regime']} dir={m['direction']} "
                f"class={m['order_class']} allowed=[{allowed}]"
            )
    mandate_text = "\n".join(mandate_lines)

    prompt = SELECTOR_PROMPT.format(
        num_slots=NUM_SLOTS,
        slot_rankings=slot_ranking_text,
        slot_mandates=mandate_text,
        environment_context=environment_text,
        env_slot_history=env_slot_history,
        trade_feedback=trade_feedback,
        max_replacements=MAX_SELECTOR_REPLACEMENTS,
    )

    llm = get_grok_llm()
    async with _LLM_SEMAPHORE:
        messages = [sdk_user(prompt)]
        response = await _call_llm(
            messages,
            temperature=0.7,
            max_tokens=8192,
            purpose="meta_selector",
        )

    # ── Parse response ──────────────────────────────────────────
    text = response.strip()  # _call_llm already returns str

    # Try full text first, then extract outermost { ... }
    decision = None
    try:
        decision = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    if not isinstance(decision, dict):
        logger.warning("Selector: Could not parse JSON response — skipping")
        return

    analysis = decision.get("analysis", "")
    actions = decision.get("actions", [])
    env_learning = decision.get("environment_learning", "")
    recommended_focus = decision.get("recommended_focus", [])

    logger.info(f"Selector analysis: {analysis[:200]}...")
    logger.info(f"Environment learning: {env_learning}")
    logger.info(f"Recommended focus: {recommended_focus}")

    # ── Execute actions (set directives — no direct code replacement) ──
    directives_set = 0
    for action_item in actions:
        if directives_set >= MAX_SELECTOR_REPLACEMENTS:
            logger.info(f"Selector: hit max directives ({MAX_SELECTOR_REPLACEMENTS})")
            break

        slot_num = action_item.get("slot")
        action = action_item.get("action", "keep")
        reason = action_item.get("reason", "")
        target_type = action_item.get("target_strategy_type", "")

        if not slot_num or slot_num < 1 or slot_num > NUM_SLOTS:
            continue

        logger.info(f"Selector: Slot {slot_num:02d} → {action} — {reason}")

        if action == "keep":
            continue

        if action in ("replace", "mutate"):
            # Set a directive — the next _run_slot iteration will use it
            # to guide the LLM proposal through normal eval pipeline
            seed_from = action_item.get("seed_from_slot")
            directive = {
                "action": action,
                "target_strategy_type": target_type,
                "reason": reason,
            }

            # Validate directive against slot mandate before setting
            mandate = SLOT_MANDATES.get(slot_num)
            if mandate:
                # If the LLM's target type is clearly an order_type, check it
                if target_type and target_type in (
                    "bracket", "trailing_stop_exit", "oca_exit", "limit",
                    "vertical_spread", "diagonal_spread", "calendar_spread",
                    "iron_condor", "butterfly", "strangle", "straddle",
                    "midprice", "moc", "stock_momentum", "stock_bracket",
                    "options_directional", "options_premium",
                    "stock_mean_reversion", "stock_range",
                ) and target_type not in mandate["allowed_order_types"] and target_type != mandate["order_class"]:
                    logger.info(
                        f"Selector: Slot {slot_num:02d} directive BLOCKED — "
                        f"target '{target_type}' violates mandate (class={mandate['order_class']})"
                    )
                    continue

            # If cloning from another slot, only allow within same regime group
            if seed_from and 1 <= seed_from <= NUM_SLOTS and seed_from != slot_num:
                donor_mandate = SLOT_MANDATES.get(seed_from, {})
                target_mandate = SLOT_MANDATES.get(slot_num, {})
                if donor_mandate.get("regime") != target_mandate.get("regime"):
                    logger.info(
                        f"Selector: Slot {slot_num:02d} clone from {seed_from:02d} BLOCKED — "
                        f"cross-regime ({donor_mandate.get('regime')} → {target_mandate.get('regime')})"
                    )
                    seed_from = None  # skip clone, still allow directive

            if seed_from:
                try:
                    donor_source = _strategy_path(seed_from).read_text(encoding="utf-8")
                    validation_err = _validate_strategy_source(donor_source)
                    if not validation_err:
                        # Evaluate donor code against current data before replacing
                        donor_agg, _, _ = await asyncio.to_thread(
                            _evaluate_strategy, donor_source, test_universe,
                        )
                        donor_fitness = _compute_fitness(donor_agg, slot=slot_num)
                        current_fitness = slot_states[slot_num]["best_test_fitness"]

                        if donor_fitness > current_fitness:
                            strat_path = _strategy_path(slot_num)
                            strat_path.write_text(donor_source, encoding="utf-8")
                            slot_states[slot_num]["best_test_fitness"] = donor_fitness
                            slot_states[slot_num]["consecutive_failures"] = 0
                            logger.info(
                                f"Selector: Slot {slot_num:02d} cloned from Slot {seed_from:02d} "
                                f"(fitness {current_fitness:.4f} → {donor_fitness:.4f})"
                            )
                            await _git_commit(
                                slot_states[slot_num]["iteration"], True,
                                donor_fitness, None, slot=slot_num,
                            )
                        else:
                            logger.info(
                                f"Selector: Slot {slot_num:02d} skip clone — donor fitness "
                                f"{donor_fitness:.4f} <= current {current_fitness:.4f}"
                            )
                    else:
                        logger.warning(f"Selector: donor slot {seed_from} failed validation")
                except FileNotFoundError:
                    logger.warning(f"Selector: seed slot {seed_from} not found")

            # Always set the directive so the next iteration knows which
            # strategy type to evolve toward
            slot_states[slot_num]["selector_directive"] = directive
            directives_set += 1
            logger.info(
                f"Selector: Slot {slot_num:02d} directive set → pivot to {target_type}"
            )

    logger.info(f"Selector complete: {directives_set} directives set")


# ═══════════════════════════════════════════════════════════════
# AUTO-TUNING
# ═══════════════════════════════════════════════════════════════

# Tracks consecutive rounds of 0 kept — resets when any slot is kept.
_consecutive_zero_kept_rounds = 0


def _auto_tune_config(
    cfg: dict[str, float],
    slot_states: dict[int, dict],
    slot_ids: list[int],
    round_num: int,
    kept_count: int,
) -> None:
    """Adjust tunable thresholds based on round outcomes.

    Conservative rules — small steps, bounded ranges, logged changes.
    Only fires when there is sustained evidence of over- or under-selectivity.
    """
    global _consecutive_zero_kept_rounds

    if kept_count == 0:
        _consecutive_zero_kept_rounds += 1
    else:
        _consecutive_zero_kept_rounds = 0

    # ── Rule 1: relax gates after 3+ consecutive all-discard rounds ──
    # If the agent can't keep anything for 3 rounds straight, the
    # thresholds are too tight for the current market environment.
    if _consecutive_zero_kept_rounds >= 3:
        # Relax min_keep_test_signals (floor: 5)
        cur = cfg["min_keep_test_signals"]
        new = max(5.0, cur - 2.0)
        if new < cur:
            set_research_config(
                "min_keep_test_signals", new,
                f"auto-tune: 0 kept for {_consecutive_zero_kept_rounds} rounds, "
                f"relaxing {cur:.0f}→{new:.0f}",
            )

        # Relax min_keep_condition_trades (floor: 1)
        cur = cfg["min_keep_condition_trades"]
        new = max(1.0, cur - 1.0)
        if new < cur:
            set_research_config(
                "min_keep_condition_trades", new,
                f"auto-tune: 0 kept for {_consecutive_zero_kept_rounds} rounds, "
                f"relaxing {cur:.0f}→{new:.0f}",
            )

        # Relax min_keep_confidence_score (floor: 0.10)
        cur = cfg["min_keep_confidence_score"]
        new = max(0.10, round(cur - 0.05, 2))
        if new < cur:
            set_research_config(
                "min_keep_confidence_score", new,
                f"auto-tune: 0 kept for {_consecutive_zero_kept_rounds} rounds, "
                f"relaxing {cur:.2f}→{new:.2f}",
            )

    # ── Rule 2: tighten gates when most slots are kept ───────────
    # If ≥75% of slots are kept, these thresholds might be too loose.
    if kept_count >= len(slot_ids) * 0.75 and round_num >= 3:
        # Tighten min_keep_test_signals (ceiling: 30)
        cur = cfg["min_keep_test_signals"]
        new = min(30.0, cur + 1.0)
        if new > cur:
            set_research_config(
                "min_keep_test_signals", new,
                f"auto-tune: {kept_count}/{len(slot_ids)} kept, "
                f"tightening {cur:.0f}→{new:.0f}",
            )

        # Tighten min_keep_confidence_score (ceiling: 0.50)
        cur = cfg["min_keep_confidence_score"]
        new = min(0.50, round(cur + 0.02, 2))
        if new > cur:
            set_research_config(
                "min_keep_confidence_score", new,
                f"auto-tune: {kept_count}/{len(slot_ids)} kept, "
                f"tightening {cur:.2f}→{new:.2f}",
            )

    # ── Rule 3: expand eval window after many low-signal rounds ──
    # If most slots fail the min_test_signals gate, we need more data.
    low_signal_slots = sum(
        1 for s in slot_ids
        if "too few test signals" in " ".join(slot_states[s].get("last_rejection_reasons", []))
    )
    if low_signal_slots >= len(slot_ids) * 0.5:
        cur = cfg["eval_days_back"]
        new = min(20.0, cur + 2.0)
        if new > cur:
            set_research_config(
                "eval_days_back", new,
                f"auto-tune: {low_signal_slots}/{len(slot_ids)} slots signal-starved, "
                f"expanding {cur:.0f}→{new:.0f} days",
            )


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

async def run_research(
    *, verbose: bool = False, dry_run: bool = False,
    slot_filter: list[int] | None = None,
):
    """Main research loop — all slots evolve concurrently, meta-learner optimizes."""
    global _LLM_SEMAPHORE
    _LLM_SEMAPHORE = asyncio.Semaphore(4)  # max 4 concurrent LLM calls

    from memory import init_db
    init_db()

    all_slot_ids = list(range(1, NUM_SLOTS + 1))
    if slot_filter:
        invalid = [s for s in slot_filter if s not in all_slot_ids]
        if invalid:
            raise ValueError(f"Invalid slot numbers: {invalid}. Valid range: 1-{NUM_SLOTS}")
        slot_ids = [s for s in all_slot_ids if s in slot_filter]
        logger.info(f"Slot filter active — running slots: {slot_ids}")
    else:
        slot_ids = all_slot_ids

    logger.info("=" * 60)
    logger.info("RESEARCH AGENT — Population-Based Strategy Evolution")
    logger.info(f"Slots: {len(slot_ids)}/{NUM_SLOTS} active")
    logger.info(f"Universe: {len(RESEARCH_UNIVERSE)} symbols")
    logger.info(f"Feature flags: {FEATURE_FLAGS}")
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
            "last_kept": False,
            "last_rejection_reasons": [],
            "darwinian_weight": 1.0,
            "_daily_returns": {},
            "fitness_history": [],  # Track fitness across rounds for decay detection
        }

    round_num = 0
    consecutive_all_fail_rounds = 0
    _cached_universe = None  # Re-use across rounds when market is closed

    while True:
        round_num += 1
        round_started_at = datetime.now(timezone.utc).isoformat()

        # ── Load tunable config (DB overrides > defaults) ────────
        cfg = _load_tunable_config()
        eval_days = int(cfg["eval_days_back"])
        selector_every = int(cfg["selector_every_n_rounds"])
        max_sel_repl = int(cfg["max_selector_replacements"])

        db_overrides = get_all_research_config()
        if db_overrides:
            logger.info(f"Tunable config overrides: {db_overrides}")

        logger.info(f"{'='*20} Round {round_num} {'='*20}")
        logger.info(f"Eval window: last {eval_days} trading days, {CANDLE_RESOLUTION}-min bars")
        logger.info(f"Selector runs every {selector_every} rounds")

        # Fetch candles once per round (shared across all slots).
        # When market is closed the historical data doesn't change, so re-use
        # the previous fetch to avoid redundant API calls / timeout crashes.
        from data.market_hours import get_market_hours_provider
        market_open = get_market_hours_provider().is_market_open()
        if _cached_universe is not None and not market_open:
            logger.info("Reusing cached candle data (market closed)")
            universe = _cached_universe
        else:
            logger.info("Fetching candle data...")
            universe = await _fetch_universe_candles(eval_days, exclude_today=True)
            _cached_universe = universe
        unique_dates = sorted({d for days in universe.values() for d in days})
        logger.info(f"Got data for {len(universe)} symbols across {len(unique_dates)} trading days ({len(universe) * len(unique_dates)} symbol-days)")

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

        # Walk-forward: use the last fold for primary evaluation (most recent OOS day)
        wf_splits = walk_forward_splits(universe, train_days=7, test_days=1)
        if len(wf_splits) > 1:
            # Use last fold: trains on most recent 7 days, tests on most recent day
            train_universe, test_universe = wf_splits[-1]
            logger.info(f"Walk-forward: {len(wf_splits)} folds, using last fold for evaluation")
        train_dates = sorted({d for days in train_universe.values() for d in days})
        test_dates = sorted({d for days in test_universe.values() for d in days})
        logger.info(f"Split: train={len(train_dates)} days ({train_dates[0]}..{train_dates[-1]}), test={len(test_dates)} days ({test_dates[0]}..{test_dates[-1]})")

        # Fetch today's candles once for all live scans
        logger.info("Fetching live candles (today)...")
        live_candles = await _fetch_live_candles()

        # Run slots in batches (skip frozen slots)
        active_slot_ids = []
        for s in slot_ids:
            if slot_states[s]["consecutive_failures"] >= CIRCUIT_BREAKER_SLOT_MAX_FAILURES:
                logger.warning(
                    f"[Slot {s:02d}] FROZEN — {slot_states[s]['consecutive_failures']} "
                    f"consecutive failures (limit {CIRCUIT_BREAKER_SLOT_MAX_FAILURES})"
                )
                continue
            active_slot_ids.append(s)
            slot_states[s]["iteration"] += 1

        if not active_slot_ids:
            logger.critical("All slots frozen — triggering selector to unfreeze")
            # Force selector run to clone/replace frozen slots
            try:
                await _run_selector(
                    slot_states, env_snapshot, env_snapshot_id,
                    train_universe, test_universe,
                )
            except Exception as e:
                logger.error(f"Selector failed: {e}")
            # Check if any slots were unfrozen by selector
            still_frozen = all(
                slot_states[s]["consecutive_failures"] >= CIRCUIT_BREAKER_SLOT_MAX_FAILURES
                for s in slot_ids
            )
            if still_frozen:
                logger.critical("All slots still frozen after selector — waiting 5 min")
                await asyncio.sleep(300)
            continue
        batches = [
            active_slot_ids[i:i + BATCH_SIZE]
            for i in range(0, len(active_slot_ids), BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(batches, 1):
            logger.info(
                f"Batch {batch_num}/{len(batches)}: "
                f"slots {[f'{s:02d}' for s in batch]}"
            )
            tasks = [
                _run_slot(
                    slot=s,
                    iteration=slot_states[s]["iteration"],
                    train_universe=train_universe,
                    test_universe=test_universe,
                    state=slot_states[s],
                    dry_run=dry_run,
                    environment_text=env_text,
                    env_snapshot=env_snapshot,
                    live_candles=live_candles,
                    cfg=cfg,
                )
                for s in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for s, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"[Slot {s:02d}] Failed: {result}")
                elif isinstance(result, dict):
                    slot_states[s] = result

        # ── Darwinian weight update ──────────────────────────────
        _update_darwinian_weights(slot_states, slot_ids, cfg)

        # ── Cross-slot correlation ───────────────────────────────
        corr_data = _compute_slot_correlations(slot_states, cfg)
        if corr_data["redundant_pairs"]:
            for _s1, _s2, _r in corr_data["redundant_pairs"]:
                logger.info(
                    f"  \u26a0 Redundant: Slot {_s1:02d} & {_s2:02d} corr={_r:.2f}"
                )
        if corr_data["pairs"]:
            logger.info(f"  Avg pairwise correlation: {corr_data['avg_correlation']:.2f}")

        # Meta-learning selector agent
        if round_num % selector_every == 0:
            try:
                await _run_selector(
                    slot_states, env_snapshot, env_snapshot_id,
                    train_universe, test_universe,
                )
            except Exception as e:
                logger.error(f"Selector failed: {e}")

        # Summary
        logger.info(f"Round {round_num} complete. Slot fitness summary:")
        for s in slot_ids:
            st = slot_states[s]
            w = st.get("darwinian_weight", 1.0)

            # Record fitness for decay tracking
            st.setdefault("fitness_history", []).append(st["best_test_fitness"])

            # Detect decay: fitness dropped >40% over last 3 rounds
            decay_marker = ""
            fh = st.get("fitness_history", [])
            if len(fh) >= 3:
                peak_recent = max(fh[-3:])
                if peak_recent > 0 and fh[-1] < peak_recent * 0.6:
                    decay_marker = " ⚠ DECAYING"
                    st["consecutive_failures"] = max(
                        st["consecutive_failures"],
                        CIRCUIT_BREAKER_SLOT_MAX_FAILURES - 1,
                    )

            logger.info(
                f"  Slot {s:02d}: fitness={st['best_test_fitness']:.4f} "
                f"weight={w:.2f} "
                f"iter={st['iteration']} failures={st['consecutive_failures']}"
                f"{decay_marker}"
            )

        rejection_counts = Counter()
        kept_count = 0
        for s in slot_ids:
            st = slot_states[s]
            if st.get("last_kept"):
                kept_count += 1
            for reason in st.get("last_rejection_reasons", []):
                rejection_counts[reason] += 1

        if rejection_counts or kept_count:
            logger.info(
                f"  Round decisions: kept={kept_count} discarded={len(slot_ids) - kept_count}"
            )
            for reason, count in rejection_counts.most_common():
                logger.info(f"    reject[{count}]: {reason}")

        # Structured promotion fitness snapshot (search vs promotion comparison)
        try:
            from memory import get_db as _get_db
            _db = _get_db()
            _slot_placeholders = ", ".join("?" for _ in slot_ids)
            _promo_rows = _db.execute(
                f"""SELECT slot, search_fitness, promotion_fitness, options_coverage_pct, promoted
                   FROM promotion_runs
                   WHERE ts >= ?
                     AND slot IN ({_slot_placeholders})
                   ORDER BY ts DESC"""
                ,
                (round_started_at, *slot_ids),
            ).fetchall()
            if _promo_rows:
                logger.info("  Promotion metrics (this round):")
                for _r in _promo_rows:
                    _pf = f"{_r['promotion_fitness']:.4f}" if _r['promotion_fitness'] else "n/a"
                    _cov = f"{_r['options_coverage_pct']:.0f}%" if _r['options_coverage_pct'] else "n/a"
                    _kept = "KEPT" if _r['promoted'] else "disc"
                    logger.info(
                        f"    Slot {_r['slot']:02d}: search={_r['search_fitness']:.4f} "
                        f"promo={_pf} cov={_cov} [{_kept}]"
                    )
        except Exception:
            pass

        # ── Auto-tune thresholds based on round outcomes ─────────
        _auto_tune_config(cfg, slot_states, slot_ids, round_num, kept_count)

        # ── All-fail circuit breaker ─────────────────────────────
        if kept_count == 0 and active_slot_ids:
            consecutive_all_fail_rounds += 1
            logger.warning(
                f"All-fail round {consecutive_all_fail_rounds}/"
                f"{CIRCUIT_BREAKER_ALL_FAIL_ROUNDS}"
            )
            if consecutive_all_fail_rounds >= CIRCUIT_BREAKER_ALL_FAIL_ROUNDS:
                logger.critical(
                    f"CIRCUIT BREAKER: {CIRCUIT_BREAKER_ALL_FAIL_ROUNDS} consecutive "
                    f"all-fail rounds — cooling down {CIRCUIT_BREAKER_COOLDOWN_SECS}s"
                )
                await asyncio.sleep(CIRCUIT_BREAKER_COOLDOWN_SECS)
                consecutive_all_fail_rounds = 0
        else:
            consecutive_all_fail_rounds = 0

        # ── Daily LLM budget check ────────────────────────────────────
        try:
            from data.cost_tracker import get_cost_tracker
            budget = get_cost_tracker().get_budget_summary()
            today_spend = budget.today_llm_cost
            if today_spend >= RESEARCH_DAILY_LLM_BUDGET:
                logger.warning(
                    f"Daily LLM budget reached: ${today_spend:.2f} >= "
                    f"${RESEARCH_DAILY_LLM_BUDGET:.2f} — stopping research"
                )
                break
            logger.info(f"Daily LLM spend: ${today_spend:.2f} / ${RESEARCH_DAILY_LLM_BUDGET:.2f}")
        except Exception as e:
            logger.debug(f"Budget check failed (continuing): {e}")

        # ── Inter-round pacing delay ──────────────────────────────────
        if ROUND_DELAY_SECS > 0:
            logger.info(f"Pacing: waiting {ROUND_DELAY_SECS}s before next round...")
            await asyncio.sleep(ROUND_DELAY_SECS)

        logger.info("Starting next round...")
