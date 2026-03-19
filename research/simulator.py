"""
Trade Simulator — walks 1-min candles bar-by-bar to evaluate signals.

Fast search path: deterministic bar-by-bar simulation for all order types and
options strategies.  Slippage is parameterised so downstream callers (promotion
engine) can substitute harsher assumptions without duplicating the walk logic.

No LLM involved — pure math.
"""

import logging
import math
import statistics
from typing import Any

import pandas as pd

from research.config import FORCE_EXIT_MINUTE, SLIPPAGE_BPS, validate_legs_json

logger = logging.getLogger(__name__)


# Two-tailed 95% t-critical values for df 1..29; df >= 30 uses z = 1.96.
_T_CRIT_95 = [
    0.0,     # df=0 placeholder
    12.706, 4.303, 3.182, 2.776, 2.571,   # df 1-5
    2.447, 2.365, 2.306, 2.262, 2.228,    # df 6-10
    2.201, 2.179, 2.160, 2.145, 2.131,    # df 11-15
    2.120, 2.110, 2.101, 2.093, 2.086,    # df 16-20
    2.080, 2.074, 2.069, 2.064, 2.060,    # df 21-25
    2.056, 2.052, 2.048, 2.045,           # df 26-29
]


def _t_critical(n: int) -> float:
    """Return two-tailed 95% critical value for n observations (df = n-1)."""
    df = n - 1
    if df < 1:
        return 1.96
    if df < len(_T_CRIT_95):
        return _T_CRIT_95[df]
    return 1.96


def compute_sample_confidence(samples: list[float]) -> dict[str, float | bool | int]:
    """Estimate how much of an observed edge may still be luck/noise."""
    n = len(samples)
    if n == 0:
        return {
            "sample_mean": 0.0,
            "sample_count": 0,
            "return_std": 0.0,
            "stderr": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "t_stat": 0.0,
            "confidence_score": 0.0,
            "luck_pressure": 1.0,
            "ci_excludes_zero": False,
        }

    mean_r = statistics.mean(samples)
    if n < 2:
        return {
            "sample_mean": mean_r,
            "sample_count": n,
            "return_std": 0.0,
            "stderr": 0.0,
            "ci95_low": mean_r,
            "ci95_high": mean_r,
            "t_stat": 0.0,
            "confidence_score": 0.0,
            "luck_pressure": 1.0,
            "ci_excludes_zero": False,
        }

    std_r = statistics.stdev(samples)
    stderr = std_r / math.sqrt(n) if n > 0 else 0.0
    t_crit = _t_critical(n)
    ci_half = t_crit * stderr
    ci_low = mean_r - ci_half
    ci_high = mean_r + ci_half
    t_stat = mean_r / stderr if stderr > 0 else 0.0

    sample_score = min(1.0, n / 30.0)
    t_score = min(1.0, abs(t_stat) / 3.0)
    confidence_score = sample_score * t_score

    return {
        "sample_mean": mean_r,
        "sample_count": n,
        "return_std": std_r,
        "stderr": stderr,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_stat": t_stat,
        "confidence_score": confidence_score,
        "luck_pressure": 1.0 - confidence_score,
        "ci_excludes_zero": ci_low > 0 or ci_high < 0,
    }


def _confidence_metrics(returns: list[float]) -> dict[str, float | bool | int]:
    """Backward-compatible alias for confidence summaries on return samples."""
    return compute_sample_confidence(returns)


def _condition_metrics(results: list[dict]) -> dict[str, Any]:
    """Score how believable the edge is inside each environment bucket."""
    grouped: dict[str, list[float]] = {}
    for r in results:
        key = r.get("env_key") or "env=unknown"
        grouped.setdefault(key, []).append(r.get("return_pct", 0.0))

    if not grouped:
        return {
            "condition_confidence": 0.0,
            "condition_count": 0,
            "positive_condition_rate": 0.0,
            "condition_metrics": [],
        }

    rows: list[dict[str, Any]] = []
    total = max(sum(len(v) for v in grouped.values()), 1)
    weighted_conf = 0.0
    positive_conditions = 0
    for key, vals in grouped.items():
        conf = _confidence_metrics(vals)
        expectancy = statistics.mean(vals) if vals else 0.0
        if expectancy > 0:
            positive_conditions += 1
        row = {
            "env_key": key,
            "trades": len(vals),
            "expectancy": round(expectancy, 4),
            "ci95_low": round(float(conf["ci95_low"]), 4),
            "ci95_high": round(float(conf["ci95_high"]), 4),
            "confidence_score": round(float(conf["confidence_score"]), 4),
            "luck_pressure": round(float(conf["luck_pressure"]), 4),
            "ci_excludes_zero": bool(conf["ci_excludes_zero"]),
        }
        rows.append(row)
        weighted_conf += float(conf["confidence_score"]) * len(vals)

    rows.sort(key=lambda x: (-x["trades"], -x["confidence_score"], -x["expectancy"]))
    return {
        "condition_confidence": weighted_conf / total,
        "condition_count": len(rows),
        "positive_condition_rate": positive_conditions / len(rows) if rows else 0.0,
        "condition_metrics": rows,
    }

# ── Slippage presets ─────────────────────────────────────────────
# Each preset maps order_type to (entry_slippage_bps, exit_slippage_bps).
# 'fast' is used for the inner research loop (throughput-optimised).
# 'strict' is used for promotion-grade stock evaluation (harsher).
_SLIPPAGE_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "fast": {
        "market":               (SLIPPAGE_BPS, SLIPPAGE_BPS),
        "limit":                (0, SLIPPAGE_BPS),
        "stop_entry":           (SLIPPAGE_BPS * 1.5, SLIPPAGE_BPS),
        "bracket":              (SLIPPAGE_BPS, SLIPPAGE_BPS),
        "trailing_stop_exit":   (SLIPPAGE_BPS, SLIPPAGE_BPS),
        "oca_exit":             (SLIPPAGE_BPS, SLIPPAGE_BPS),
        "midprice":             (SLIPPAGE_BPS * 0.5, SLIPPAGE_BPS * 0.5),
        "vwap":                 (SLIPPAGE_BPS * 0.5, SLIPPAGE_BPS * 0.5),
        "moc":                  (SLIPPAGE_BPS, SLIPPAGE_BPS * 2),
        "moo":                  (SLIPPAGE_BPS * 2, SLIPPAGE_BPS),
        "_options":             (0, 0),   # options: handled by separate pricers
    },
    "strict": {
        "market":               (SLIPPAGE_BPS * 3, SLIPPAGE_BPS * 3),
        "limit":                (0, SLIPPAGE_BPS * 2),
        "stop_entry":           (SLIPPAGE_BPS * 4, SLIPPAGE_BPS * 2),
        "bracket":              (SLIPPAGE_BPS * 3, SLIPPAGE_BPS * 3),
        "trailing_stop_exit":   (SLIPPAGE_BPS * 3, SLIPPAGE_BPS * 3),
        "oca_exit":             (SLIPPAGE_BPS * 3, SLIPPAGE_BPS * 3),
        "midprice":             (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 2),
        "vwap":                 (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 2),
        "moc":                  (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 4),
        "moo":                  (SLIPPAGE_BPS * 4, SLIPPAGE_BPS * 2),
        "_options":             (0, 0),
    },
}

# Time-of-day slippage multiplier: first and last 15 minutes are more expensive
_TOD_SLIPPAGE: list[tuple[int, int, float]] = [
    # (start_minute_inclusive, end_minute_exclusive, multiplier)
    (570, 585,  2.0),   # 09:30–09:45 open auction
    (945, 960,  1.5),   # 15:45–16:00 close approach
]


def _tod_multiplier(bar_minute: int | None) -> float:
    if bar_minute is None:
        return 1.0
    for start, end, mult in _TOD_SLIPPAGE:
        if start <= bar_minute < end:
            return mult
    return 1.0


def simulate(
    signals: list[dict],
    candles: pd.DataFrame,
    slippage_preset: str = "fast",
) -> list[dict]:
    """
    Simulate trades from signals against 1-min candle data.

    Each signal specifies entry conditions, stop, target, max hold, and order type.
    The simulator walks forward bar-by-bar and records outcomes mechanically.

    Args:
        signals: list of signal dicts from strategy.scan()
        candles: DataFrame with columns [ts, open, high, low, close, volume]
        slippage_preset: 'fast' (default, inner loop) or 'strict' (promotion grade)

    Returns:
        list of result dicts with outcome fields added
    """
    if not signals or candles.empty:
        return []

    results = []
    for sig in signals:
        result = _simulate_one(sig, candles, slippage_preset=slippage_preset)
        if result is not None:
            results.append(result)
    return results


def _simulate_one(sig: dict, candles: pd.DataFrame, slippage_preset: str = "fast") -> dict | None:
    """Simulate a single signal to completion."""
    entry_bar = sig.get("entry_bar")
    if entry_bar is None or entry_bar < 0 or entry_bar >= len(candles):
        return None

    direction = sig.get("direction", "long")
    order_type = sig.get("order_type", "market")
    entry_price = sig.get("entry_price")
    target_price = sig.get("target_price")
    stop_price = sig.get("stop_price")
    max_hold_bars = sig.get("max_hold_bars", 60)

    if entry_price is None or target_price is None or stop_price is None:
        return None

    # Validate legs_json schema early — reject invalid payloads rather than propagating
    # garbage P&L all the way through the promotion pipeline
    legs_early = sig.get("legs_json")
    if legs_early and isinstance(legs_early, dict):
        err = validate_legs_json(legs_early)
        if err:
            logger.debug(f"Skipping signal with invalid legs_json: {err}")
            return None

    # Determine fill bar and fill price based on order type
    fill_bar, fill_price = _get_fill(order_type, entry_bar, entry_price, candles, direction)
    if fill_bar is None:
        return None

    # Recalculate stop/target relative to actual fill price (preserves % distance)
    if entry_price and entry_price > 0:
        target_dist = (target_price - entry_price) / entry_price
        stop_dist = (stop_price - entry_price) / entry_price
        target_price = fill_price * (1 + target_dist)
        stop_price = fill_price * (1 + stop_dist)

    # Walk forward from fill bar
    is_long = direction == "long"
    trail_stop = stop_price
    trail_pct = sig.get("trail_percent", 0)

    exit_bar = None
    exit_price = None
    hit_target = 0
    hit_stop = 0
    timed_out = 0

    for i in range(fill_bar + 1, len(candles)):
        bar = candles.iloc[i]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        # Force exit at 15:55 ET
        bar_minute = _minute_of_day(bar["ts"])
        if bar_minute is not None and bar_minute >= FORCE_EXIT_MINUTE:
            exit_bar = i
            exit_price = bar_close
            timed_out = 1
            break

        # Max hold check
        if i - fill_bar >= max_hold_bars:
            exit_bar = i
            exit_price = bar_close
            timed_out = 1
            break

        # Trailing stop update
        if trail_pct > 0 and is_long and bar_high > fill_price:
            new_trail = bar_high * (1 - trail_pct / 100)
            if new_trail > trail_stop:
                trail_stop = new_trail
        elif trail_pct > 0 and not is_long and bar_low < fill_price:
            new_trail = bar_low * (1 + trail_pct / 100)
            if new_trail < trail_stop:
                trail_stop = new_trail

        active_stop = trail_stop if trail_pct > 0 else stop_price

        if is_long:
            stop_hit = bar_low <= active_stop
            tgt_hit = bar_high >= target_price
            if stop_hit and tgt_hit:
                # Both hit on same bar — use open to infer which came first
                bar_open = bar["open"]
                if bar_open >= fill_price:  # opened in favorable direction
                    exit_bar, exit_price, hit_target = i, target_price, 1
                else:
                    exit_bar, exit_price, hit_stop = i, active_stop, 1
                break
            elif stop_hit:
                exit_bar, exit_price, hit_stop = i, active_stop, 1
                break
            elif tgt_hit:
                exit_bar, exit_price, hit_target = i, target_price, 1
                break
        else:
            stop_hit = bar_high >= active_stop
            tgt_hit = bar_low <= target_price
            if stop_hit and tgt_hit:
                bar_open = bar["open"]
                if bar_open <= fill_price:
                    exit_bar, exit_price, hit_target = i, target_price, 1
                else:
                    exit_bar, exit_price, hit_stop = i, active_stop, 1
                break
            elif stop_hit:
                exit_bar, exit_price, hit_stop = i, active_stop, 1
                break
            elif tgt_hit:
                exit_bar, exit_price, hit_target = i, target_price, 1
                break

    # If we never exited (ran out of bars), close at last bar
    if exit_bar is None:
        exit_bar = len(candles) - 1
        exit_price = candles.iloc[-1]["close"]
        timed_out = 1

    # Compute return (subtract slippage + commission based on preset + time-of-day)
    if is_long:
        return_pct = (exit_price - fill_price) / fill_price * 100
    else:
        return_pct = (fill_price - exit_price) / fill_price * 100

    preset = _SLIPPAGE_PRESETS.get(slippage_preset, _SLIPPAGE_PRESETS["fast"])
    is_options = order_type in (
        "vertical_spread", "iron_condor", "straddle", "strangle",
        "calendar_spread", "diagonal_spread", "butterfly",
    )
    slip_key = "_options" if is_options else order_type
    entry_slip_bps, exit_slip_bps = preset.get(slip_key, (SLIPPAGE_BPS, SLIPPAGE_BPS))
    fill_bar_minute = _minute_of_day(candles.iloc[fill_bar]["ts"]) if fill_bar < len(candles) else None
    entry_mult = _tod_multiplier(fill_bar_minute)
    exit_bar_minute = _minute_of_day(candles.iloc[exit_bar]["ts"]) if exit_bar < len(candles) else None
    exit_mult = _tod_multiplier(exit_bar_minute)
    total_slip = (entry_slip_bps * entry_mult + exit_slip_bps * exit_mult) / 100
    if not is_options:
        return_pct -= total_slip

    # Handle options legs if present
    legs = sig.get("legs_json")
    if legs and isinstance(legs, dict):
        return_pct = _simulate_options_return(legs, fill_price, exit_price, is_long,
                                               fill_bar, exit_bar, candles)

    return {
        **sig,
        "fill_bar": fill_bar,
        "fill_price": fill_price,
        "exit_bar": exit_bar,
        "exit_ts": str(candles.iloc[exit_bar]["ts"]) if exit_bar < len(candles) else None,
        "exit_price": exit_price,
        "hit_target": hit_target,
        "hit_stop": hit_stop,
        "timed_out": timed_out,
        "return_pct": round(return_pct, 4),
        "hold_bars": exit_bar - fill_bar,
    }


def _get_fill(order_type: str, entry_bar: int, entry_price: float,
              candles: pd.DataFrame, direction: str) -> tuple[int | None, float | None]:
    """Determine fill bar and price based on order type."""

    if entry_bar + 1 >= len(candles):
        return None, None

    next_bar = candles.iloc[entry_bar + 1]

    if order_type == "market":
        return entry_bar + 1, next_bar["open"]

    elif order_type == "limit":
        # Scan forward for a bar where price reaches limit
        is_buy = direction == "long"
        for i in range(entry_bar + 1, min(entry_bar + 11, len(candles))):
            bar = candles.iloc[i]
            if is_buy and bar["low"] <= entry_price:
                return i, entry_price
            elif not is_buy and bar["high"] >= entry_price:
                return i, entry_price
        return None, None  # limit not reached within 10 bars

    elif order_type == "stop_entry":
        # Breakout entry: fill when price reaches trigger
        is_buy = direction == "long"
        for i in range(entry_bar + 1, min(entry_bar + 11, len(candles))):
            bar = candles.iloc[i]
            if is_buy and bar["high"] >= entry_price:
                return i, entry_price
            elif not is_buy and bar["low"] <= entry_price:
                return i, entry_price
        return None, None

    elif order_type in ("bracket", "oca_exit"):
        # Bracket/OCA: entry as limit, stop+target managed in the walk loop
        return entry_bar + 1, next_bar["open"]

    elif order_type == "trailing_stop_exit":
        # Enter at market, trailing stop managed in walk loop
        return entry_bar + 1, next_bar["open"]

    elif order_type == "midprice":
        # Approximate midpoint as (open + close) / 2 of entry bar
        mid = (next_bar["open"] + next_bar["close"]) / 2
        return entry_bar + 1, mid

    elif order_type == "vwap":
        # Approximate VWAP over next 5 bars
        end = min(entry_bar + 6, len(candles))
        chunk = candles.iloc[entry_bar + 1:end]
        if chunk.empty:
            return None, None
        vwap = (chunk["close"] * chunk["volume"]).sum() / chunk["volume"].sum()
        return entry_bar + 5 if end > entry_bar + 5 else end - 1, vwap

    elif order_type == "moc":
        # Market on close — fill at last bar's close
        last = len(candles) - 1
        return last, candles.iloc[last]["close"]

    elif order_type == "moo":
        # Market on open — fill at first bar's open
        return 0, candles.iloc[0]["open"]

    # Options strategies: enter at market for simplicity
    elif order_type in ("vertical_spread", "iron_condor", "straddle", "strangle",
                        "calendar_spread", "diagonal_spread", "butterfly"):
        return entry_bar + 1, next_bar["open"]

    else:
        # Unknown order type — default to market
        return entry_bar + 1, next_bar["open"]


def _simulate_options_return(legs: dict, underlying_entry: float, underlying_exit: float,
                              is_long: bool, fill_bar: int, exit_bar: int,
                              candles: pd.DataFrame) -> float:
    """
    Approximate options P&L using delta/theta estimation.

    This is simplified — we estimate P&L as:
      delta_pnl = delta * (price_change) * 100
      theta_cost = theta * (bars_held / 390)  # 390 bars = 1 day
      return_pct = (delta_pnl - theta_cost) / max_risk * 100

    For defined-risk strategies, max_risk = spread width or debit paid.
    """
    strategy = legs.get("strategy", "")
    price_move = underlying_exit - underlying_entry
    bars_held = exit_bar - fill_bar
    days_held = bars_held / 390.0

    if strategy == "vertical_spread":
        long_strike = legs.get("long_strike", underlying_entry)
        short_strike = legs.get("short_strike", underlying_entry)
        width = abs(short_strike - long_strike)
        right = legs.get("right", "C")
        is_call = right.upper() == "C"

        # Estimate debit as ~40% of width for ATM spreads
        debit = width * 0.4
        if debit <= 0:
            return 0.0

        # Delta approximation: ~0.3 net delta for vertical
        net_delta = 0.3 if is_call else -0.3
        delta_pnl = net_delta * price_move
        # Theta cost: ~2% of debit per day for near-term spreads
        theta_cost = debit * 0.02 * days_held
        pnl = delta_pnl - theta_cost
        return (pnl / debit) * 100

    elif strategy == "iron_condor":
        # Credit received ~30% of wing width, max loss = wing - credit
        put_long = legs.get("put_long_strike", underlying_entry * 0.95)
        put_short = legs.get("put_short_strike", underlying_entry * 0.97)
        call_short = legs.get("call_short_strike", underlying_entry * 1.03)
        call_long = legs.get("call_long_strike", underlying_entry * 1.05)
        wing_width = max(abs(put_short - put_long), abs(call_long - call_short), 1.0)
        credit = wing_width * 0.3
        max_loss = wing_width - credit

        if put_short <= underlying_exit <= call_short:
            # Price in range — profit is theta decay
            pnl = credit * min(days_held * 0.1, 1.0)  # decay toward full credit
        else:
            # Breach — loss proportional to how far OTM
            breach = max(put_short - underlying_exit, underlying_exit - call_short, 0)
            pnl = -(min(breach, max_loss))
        return (pnl / max_loss) * 100

    elif strategy in ("straddle", "strangle"):
        # Long vol: cost is total premium, profit from large moves
        premium = abs(underlying_entry) * 0.04  # ~4% premium estimate
        abs_move = abs(price_move)
        delta_pnl = abs_move * 0.8  # combined delta ~0.8 at-move
        theta_cost = premium * 0.03 * days_held
        pnl = delta_pnl - theta_cost - premium * 0.1  # subtract ~10% of premium as IV crush
        return (pnl / premium) * 100

    elif strategy in ("calendar_spread", "diagonal_spread"):
        # Calendar: profits from theta differential
        debit = abs(underlying_entry) * 0.02  # ~2% of underlying
        theta_income = debit * 0.05 * days_held  # front leg decays faster
        delta_risk = abs(price_move) * 0.1  # low delta exposure
        pnl = theta_income - delta_risk
        return (pnl / debit) * 100

    elif strategy == "butterfly":
        low = legs.get("low_strike", underlying_entry - 5)
        mid = legs.get("mid_strike", underlying_entry)
        high = legs.get("high_strike", underlying_entry + 5)
        width = max(abs(high - low) / 2, 0.5)
        debit = width * 0.2
        dist_from_center = abs(underlying_exit - mid)
        if dist_from_center <= width * 0.3:
            pnl = (width - debit) * (1 - dist_from_center / (width * 0.3)) * 0.5
        else:
            pnl = -debit * 0.8
        return (pnl / max(debit, 0.01)) * 100

    return 0.0


def _minute_of_day(ts: Any) -> int | None:
    """Extract minute-of-day (ET) from a timestamp. Returns None if unparseable."""
    try:
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        if isinstance(ts, (int, float)):
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(_et)
            return dt.hour * 60 + dt.minute
        elif isinstance(ts, str):
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(_et)
            return dt.hour * 60 + dt.minute
        elif hasattr(ts, "hour"):
            return ts.hour * 60 + ts.minute
    except Exception:
        pass
    return None


def compute_expectancy(results: list[dict]) -> dict:
    """
    Compute aggregate statistics plus search fitness from simulation results.

    Returns dict with: hit_rate, avg_win, avg_loss, expectancy,
    profit_factor, max_drawdown, sharpe_approx, total_signals,
    stability_score, search_fitness.
    """
    if not results:
        return {
            "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "sharpe_approx": 0.0,
            "total_signals": 0,
            "stability_score": 0.0,
            "return_std": 0.0,
            "stderr": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "t_stat": 0.0,
            "confidence_score": 0.0,
            "luck_pressure": 1.0,
            "condition_confidence": 0.0,
            "condition_count": 0,
            "positive_condition_rate": 0.0,
            "signed_edge_score": 0.0,
            "raw_search_fitness": 0.0,
            "condition_metrics": [],
            "search_fitness": 0.0,
        }

    returns = [r["return_pct"] for r in results]
    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r < 0]

    total = len(returns)
    hit_rate = len(winners) / total * 100 if total > 0 else 0.0
    avg_win = sum(winners) / len(winners) if winners else 0.0
    avg_loss = sum(losers) / len(losers) if losers else 0.0

    # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    expectancy = (hit_rate / 100 * avg_win) + ((1 - hit_rate / 100) * avg_loss)

    # Profit factor = gross_wins / gross_losses
    gross_wins = sum(winners) if winners else 0.0
    gross_losses = abs(sum(losers)) if losers else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    # Max drawdown (cumulative)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Sharpe approximation (annualized, assumes uniform sampling)
    if len(returns) >= 2:
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    confidence = _confidence_metrics(returns)
    condition = _condition_metrics(results)

    # Average R:R
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf") if avg_win > 0 else 0.0

    # ── Stability score ───────────────────────────────────────────
    # Measures how consistently the strategy performs across symbols and days.
    # 1.0 = perfectly consistent, 0.0 = all results on one day/symbol only.
    symbols_seen = {r.get("symbol", "unknown") for r in results}
    dates_seen: set[str] = set()
    for r in results:
        ts = r.get("entry_ts") or r.get("exit_ts")
        if ts:
            dates_seen.add(str(ts)[:10])
    n_symbols = max(len(symbols_seen), 1)
    n_days = max(len(dates_seen), 1)
    stability_score = min(1.0, math.log1p(n_symbols) * math.log1p(n_days) / 10.0)

    # ── Search fitness: composite fast-search score ───────────────
    # Penalizes:
    #   - too few signals (< 5 not trustworthy)
    #   - large drawdown relative to expectancy
    #   - low symbol/day diversity (concentration penalty)
    # Rewards:
    #   - positive expectancy
    #   - high profit factor
    #   - cross-symbol stability
    sample_penalty = min(1.0, total / 10.0)  # ramp from 0→1 as signals grow 0→10
    drawdown_penalty = max(0.0, 1.0 - max_dd / 20.0)  # dd>20% kills score quickly
    fitness_profit_factor = profit_factor if math.isfinite(profit_factor) else 5.0
    cc = float(condition["condition_confidence"])
    signed_edge_score = (
        expectancy
        * sample_penalty
        * drawdown_penalty
        * stability_score
        * float(confidence["confidence_score"])
        * (cc if cc > 0 else 0.25)
    )
    fitness_raw = (
        expectancy
        * fitness_profit_factor
        * sample_penalty
        * drawdown_penalty
        * stability_score
        * float(confidence["confidence_score"])
        * (cc if cc > 0 else 0.25)
    )
    signed_edge_score = round(signed_edge_score, 6)
    raw_search_fitness = round(fitness_raw, 6)
    search_fitness = round(max(fitness_raw, 0.0), 6)

    return {
        "hit_rate": round(hit_rate, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "avg_rr": round(avg_rr, 2),
        "expectancy": round(expectancy, 4),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_approx": round(sharpe, 2),
        "total_signals": total,
        "stability_score": round(stability_score, 4),
        "return_std": round(float(confidence["return_std"]), 4),
        "stderr": round(float(confidence["stderr"]), 4),
        "ci95_low": round(float(confidence["ci95_low"]), 4),
        "ci95_high": round(float(confidence["ci95_high"]), 4),
        "t_stat": round(float(confidence["t_stat"]), 4),
        "confidence_score": round(float(confidence["confidence_score"]), 4),
        "luck_pressure": round(float(confidence["luck_pressure"]), 4),
        "condition_confidence": round(float(condition["condition_confidence"]), 4),
        "condition_count": int(condition["condition_count"]),
        "positive_condition_rate": round(float(condition["positive_condition_rate"]), 4),
        "signed_edge_score": signed_edge_score,
        "raw_search_fitness": raw_search_fitness,
        "condition_metrics": condition["condition_metrics"][:6],
        "search_fitness": search_fitness,
    }


def format_confidence_line(
    label: str, values: list[float], kind: str = "raw"
) -> str | None:
    """Format a confidence summary line using compute_sample_confidence.

    Used in trade feedback and research briefing for consistent reporting
    of means, CI95, confidence_score, and luck_pressure.
    """
    samples = [float(v) for v in values if v is not None]
    if not samples:
        return None

    summary = compute_sample_confidence(samples)

    def _fmt(value: float) -> str:
        if kind == "pct":
            return f"{value:+.2f}%"
        if kind == "usd":
            return f"${value:+.2f}"
        return f"{value:+.2f}"

    return (
        f"  {label}: mean={_fmt(float(summary['sample_mean']))} "
        f"ci95=[{_fmt(float(summary['ci95_low']))}, {_fmt(float(summary['ci95_high']))}] "
        f"conf={float(summary['confidence_score']):.2f} "
        f"luck={float(summary['luck_pressure']):.2f} "
        f"(n={int(summary['sample_count'])})"
    )
