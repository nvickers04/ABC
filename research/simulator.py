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
from datetime import datetime, date, timezone
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

# ── Calibrated slippage from live execution data ─────────────────
# Cached with a version check — invalidated when memory.upsert_calibrated_slippage bumps the counter.
_calibrated_cache: dict[tuple[str, str, str], float] | None = None
_calibrated_version: int = -1


def _load_calibrated() -> dict[tuple[str, str, str], float]:
    global _calibrated_cache, _calibrated_version
    try:
        from memory import get_calibrated_slippage, get_calibration_version
        current_ver = get_calibration_version()
        if _calibrated_cache is None or current_ver != _calibrated_version:
            _calibrated_cache = get_calibrated_slippage()
            _calibrated_version = current_ver
    except Exception:
        logger.warning("Calibrated slippage load failed — falling back to defaults")
        if _calibrated_cache is None:
            _calibrated_cache = {}
    return _calibrated_cache


def _minute_to_time_bucket(bar_minute: int | None) -> str:
    """Map a minute-of-day (ET) to the same bucket labels used by execution snapshots."""
    if bar_minute is None:
        return "unknown"
    if 570 <= bar_minute < 585:
        return "open"
    elif 585 <= bar_minute < 720:
        return "morning"
    elif 720 <= bar_minute < 945:
        return "midday"
    elif 945 <= bar_minute < 960:
        return "close"
    return "extended"


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
        "adaptive":             (SLIPPAGE_BPS * 0.7, SLIPPAGE_BPS * 0.7),
        "vwap":                 (SLIPPAGE_BPS * 0.5, SLIPPAGE_BPS * 0.5),
        "twap":                 (SLIPPAGE_BPS * 0.5, SLIPPAGE_BPS * 0.5),
        "relative":             (SLIPPAGE_BPS * 0.5, SLIPPAGE_BPS * 0.5),
        "snap_mid":             (SLIPPAGE_BPS * 0.5, SLIPPAGE_BPS * 0.5),
        "moc":                  (SLIPPAGE_BPS, SLIPPAGE_BPS * 2),
        "moo":                  (SLIPPAGE_BPS * 2, SLIPPAGE_BPS),
        "loc":                  (0, SLIPPAGE_BPS * 2),
        "loo":                  (0, SLIPPAGE_BPS),
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
        "adaptive":             (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 2),
        "vwap":                 (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 2),
        "twap":                 (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 2),
        "relative":             (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 2),
        "snap_mid":             (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 2),
        "moc":                  (SLIPPAGE_BPS * 2, SLIPPAGE_BPS * 4),
        "moo":                  (SLIPPAGE_BPS * 4, SLIPPAGE_BPS * 2),
        "loc":                  (0, SLIPPAGE_BPS * 4),
        "loo":                  (0, SLIPPAGE_BPS * 2),
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
    exit_bar_minute = _minute_of_day(candles.iloc[exit_bar]["ts"]) if exit_bar < len(candles) else None

    # Use calibrated entry slippage from live execution data when available (strict only).
    # Calibrated values already incorporate time-of-day effects, so skip tod_multiplier.
    calibrated_entry = False
    if slippage_preset == "strict" and not is_options:
        cal = _load_calibrated()
        if cal:
            tb = _minute_to_time_bucket(fill_bar_minute)
            # Prefer: time-specific (order_type, time_bucket, "all") if it exists
            # Fallback: broad (order_type, "all", "all") — most data, most stable
            broad_key = (order_type, "all", "all")
            time_key = (order_type, tb, "all")
            cal_key = time_key if time_key in cal else broad_key
            if cal_key in cal:
                entry_slip_bps = cal[cal_key]
                calibrated_entry = True

    entry_mult = 1.0 if calibrated_entry else _tod_multiplier(fill_bar_minute)
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
        # Fill probability: depends on how far price penetrates through the limit.
        # If limit is just touched (wicked), fill probability < 100%.
        import random
        is_buy = direction == "long"
        for i in range(entry_bar + 1, min(entry_bar + 11, len(candles))):
            bar = candles.iloc[i]
            if is_buy and bar["low"] <= entry_price:
                # How deep did price go through the limit?
                bar_range = bar["high"] - bar["low"]
                if bar_range > 0:
                    penetration = (entry_price - bar["low"]) / bar_range
                else:
                    penetration = 1.0
                # Fill probability: 50% at touch, 100% at full penetration
                fill_prob = min(1.0, 0.5 + 0.5 * penetration)
                if random.random() < fill_prob:
                    return i, entry_price
                # Not filled this bar — keep scanning
            elif not is_buy and bar["high"] >= entry_price:
                bar_range = bar["high"] - bar["low"]
                if bar_range > 0:
                    penetration = (bar["high"] - entry_price) / bar_range
                else:
                    penetration = 1.0
                fill_prob = min(1.0, 0.5 + 0.5 * penetration)
                if random.random() < fill_prob:
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


# ═══════════════════════════════════════════════════════════════
# BLACK-SCHOLES OPTION PRICING
# ═══════════════════════════════════════════════════════════════

_RISK_FREE_RATE = 0.05  # Approximate; not a sensitive parameter for short-DTE math
_MIN_VOL = 0.10         # Floor so options never collapse to intrinsic


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price.  T in years, sigma annualised."""
    try:
        if not (math.isfinite(S) and math.isfinite(K) and S > 0 and K > 0 and T > 0 and sigma > 0):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    d1 = (math.log(S / K) + (_RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-_RISK_FREE_RATE * T) * _norm_cdf(d2)


def _bs_put(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes put price via put-call parity."""
    try:
        if not (math.isfinite(S) and math.isfinite(K) and S > 0 and K > 0 and T > 0 and sigma > 0):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    d1 = (math.log(S / K) + (_RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-_RISK_FREE_RATE * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_price(S: float, K: float, T: float, sigma: float, is_call: bool) -> float:
    return _bs_call(S, K, T, sigma) if is_call else _bs_put(S, K, T, sigma)


def _realised_vol(candles: pd.DataFrame, fill_bar: int) -> float:
    """Estimate annualised volatility from close-to-close returns preceding fill_bar."""
    lookback = min(fill_bar, 60)  # Up to 60 bars before entry
    if lookback < 5:
        return 0.25  # Fallback
    closes = candles["close"].iloc[max(0, fill_bar - lookback):fill_bar].values
    if len(closes) < 5:
        return 0.25
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    if not rets:
        return 0.25
    std_1m = statistics.stdev(rets)
    annualised = std_1m * math.sqrt(390 * 252)  # 390 1-min bars/day, 252 days/yr
    return max(annualised, _MIN_VOL)


def _skew_adjusted_vol(base_sigma: float, S: float, K: float, is_call: bool) -> float:
    """Apply IV skew based on moneyness.

    OTM puts get higher implied vol (the "skew smile"), OTM calls slightly lower.
    Uses a simple quadratic smile: vol = base * (1 + skew_coeff * (moneyness - 1)^2 + tilt)
    """
    if S <= 0 or K <= 0:
        return base_sigma
    moneyness = K / S
    # Quadratic smile: vol increases for strikes far from ATM
    smile = 0.8 * (moneyness - 1.0) ** 2
    # Negative tilt: puts (low moneyness) get extra vol, calls (high moneyness) get less
    tilt = -0.15 * (moneyness - 1.0)
    adjusted = base_sigma * (1.0 + smile + tilt)
    return max(adjusted, _MIN_VOL)


def _dte_years(legs: dict, bars_held: int, entry_date: date | None = None) -> tuple[float, float]:
    """Return (T_entry, T_exit) in years.

    If expiration is parseable, compute real calendar DTE from entry_date.
    Otherwise fall back to a conservative 7-day estimate.
    """
    try:
        exp = datetime.strptime(str(legs["expiration"]), "%Y-%m-%d").date()
        ref = entry_date or date.today()
        dte_entry = max((exp - ref).days, 1)
    except Exception:
        dte_entry = 7  # Fallback for unparseable dates
    days_held = bars_held / 390.0
    T_entry = dte_entry / 365.0
    T_exit = max((dte_entry - days_held) / 365.0, 0.0001)
    return T_entry, T_exit


def _simulate_options_return(legs: dict, underlying_entry: float, underlying_exit: float,
                              is_long: bool, fill_bar: int, exit_bar: int,
                              candles: pd.DataFrame) -> float:
    """
    Option P&L via Black-Scholes repricing.

    Uses realised volatility from the candle data as an IV proxy, then prices
    each leg at entry and exit using the actual underlying prices.  This replaces
    the previous hardcoded-delta/theta approximation with physics-based math.

    Returns return_pct based on max risk (debit paid or margin requirement).
    """
    strategy = legs.get("strategy", "")
    bars_held = exit_bar - fill_bar
    sigma = _realised_vol(candles, fill_bar)
    entry_date = _entry_date_from_candle(candles, fill_bar)

    if not (math.isfinite(underlying_entry) and math.isfinite(underlying_exit)
            and underlying_entry > 0 and underlying_exit > 0):
        return 0.0

    if strategy == "vertical_spread":
        long_strike = legs.get("long_strike", underlying_entry)
        short_strike = legs.get("short_strike", underlying_entry)
        right = legs.get("right", "C")
        is_call = right.upper() in ("C", "CALL")
        T_entry, T_exit = _dte_years(legs, bars_held, entry_date)

        sigma_long = _skew_adjusted_vol(sigma, underlying_entry, long_strike, is_call)
        sigma_short = _skew_adjusted_vol(sigma, underlying_entry, short_strike, is_call)
        long_entry = _bs_price(underlying_entry, long_strike, T_entry, sigma_long, is_call)
        short_entry = _bs_price(underlying_entry, short_strike, T_entry, sigma_short, is_call)
        long_exit = _bs_price(underlying_exit, long_strike, T_exit, sigma_long, is_call)
        short_exit = _bs_price(underlying_exit, short_strike, T_exit, sigma_short, is_call)

        # Debit spread: pay for long, receive for short
        debit = long_entry - short_entry
        exit_value = long_exit - short_exit
        if abs(debit) < 0.001:
            return 0.0
        pnl = exit_value - debit
        return (pnl / abs(debit)) * 100

    elif strategy == "iron_condor":
        put_long = legs.get("put_long_strike", underlying_entry * 0.95)
        put_short = legs.get("put_short_strike", underlying_entry * 0.97)
        call_short = legs.get("call_short_strike", underlying_entry * 1.03)
        call_long = legs.get("call_long_strike", underlying_entry * 1.05)
        T_entry, T_exit = _dte_years(legs, bars_held, entry_date)

        # Credit from short legs minus debit for long wings (skew-adjusted per strike)
        sig_pl = _skew_adjusted_vol(sigma, underlying_entry, put_long, False)
        sig_ps = _skew_adjusted_vol(sigma, underlying_entry, put_short, False)
        sig_cs = _skew_adjusted_vol(sigma, underlying_entry, call_short, True)
        sig_cl = _skew_adjusted_vol(sigma, underlying_entry, call_long, True)
        credit = (
            _bs_put(underlying_entry, put_short, T_entry, sig_ps)
            - _bs_put(underlying_entry, put_long, T_entry, sig_pl)
            + _bs_call(underlying_entry, call_short, T_entry, sig_cs)
            - _bs_call(underlying_entry, call_long, T_entry, sig_cl)
        )
        exit_cost = (
            _bs_put(underlying_exit, put_short, T_exit, sig_ps)
            - _bs_put(underlying_exit, put_long, T_exit, sig_pl)
            + _bs_call(underlying_exit, call_short, T_exit, sig_cs)
            - _bs_call(underlying_exit, call_long, T_exit, sig_cl)
        )
        wing_width = max(abs(put_short - put_long), abs(call_long - call_short), 0.5)
        max_risk = wing_width - abs(credit)
        if max_risk <= 0:
            max_risk = wing_width
        pnl = credit - exit_cost
        return (pnl / max_risk) * 100

    elif strategy in ("straddle", "strangle"):
        if strategy == "straddle":
            strike_c = strike_p = legs.get("strike", underlying_entry)
        else:
            strike_c = legs.get("call_strike", underlying_entry * 1.02)
            strike_p = legs.get("put_strike", underlying_entry * 0.98)
        T_entry, T_exit = _dte_years(legs, bars_held, entry_date)

        sig_c = _skew_adjusted_vol(sigma, underlying_entry, strike_c, True)
        sig_p = _skew_adjusted_vol(sigma, underlying_entry, strike_p, False)
        call_entry = _bs_call(underlying_entry, strike_c, T_entry, sig_c)
        put_entry = _bs_put(underlying_entry, strike_p, T_entry, sig_p)
        call_exit = _bs_call(underlying_exit, strike_c, T_exit, sig_c)
        put_exit = _bs_put(underlying_exit, strike_p, T_exit, sig_p)

        debit = call_entry + put_entry
        exit_value = call_exit + put_exit
        if debit < 0.001:
            return 0.0
        pnl = exit_value - debit
        return (pnl / debit) * 100

    elif strategy in ("calendar_spread", "diagonal_spread"):
        right = legs.get("right", "C")
        is_call = right.upper() in ("C", "CALL")
        if strategy == "calendar_spread":
            near_strike = far_strike = legs.get("strike", underlying_entry)
        else:
            near_strike = legs.get("near_strike", underlying_entry)
            far_strike = legs.get("far_strike", underlying_entry)

        # Parse both expirations
        try:
            near_exp = datetime.strptime(str(legs["near_expiration"]), "%Y-%m-%d").date()
            far_exp = datetime.strptime(str(legs["far_expiration"]), "%Y-%m-%d").date()
            ref = entry_date or date.today()
            near_dte = max((near_exp - ref).days, 1)
            far_dte = max((far_exp - ref).days, 1)
        except Exception:
            near_dte, far_dte = 7, 21

        days_held_cal = bars_held / 390.0
        T_near_entry = near_dte / 365.0
        T_far_entry = far_dte / 365.0
        T_near_exit = max((near_dte - days_held_cal) / 365.0, 0.0001)
        T_far_exit = max((far_dte - days_held_cal) / 365.0, 0.0001)

        # Sell near-term, buy far-term (skew-adjusted per strike)
        sig_near = _skew_adjusted_vol(sigma, underlying_entry, near_strike, is_call)
        sig_far = _skew_adjusted_vol(sigma, underlying_entry, far_strike, is_call)
        near_entry = _bs_price(underlying_entry, near_strike, T_near_entry, sig_near, is_call)
        far_entry = _bs_price(underlying_entry, far_strike, T_far_entry, sig_far, is_call)
        near_exit = _bs_price(underlying_exit, near_strike, T_near_exit, sig_near, is_call)
        far_exit = _bs_price(underlying_exit, far_strike, T_far_exit, sig_far, is_call)

        debit = far_entry - near_entry  # Pay for far, receive for near
        exit_value = far_exit - near_exit
        if abs(debit) < 0.001:
            return 0.0
        pnl = exit_value - debit
        return (pnl / abs(debit)) * 100

    elif strategy == "butterfly":
        low = legs.get("low_strike", underlying_entry - 5)
        mid = legs.get("mid_strike", underlying_entry)
        high = legs.get("high_strike", underlying_entry + 5)
        right = legs.get("right", "C")
        is_call = right.upper() in ("C", "CALL")
        T_entry, T_exit = _dte_years(legs, bars_held, entry_date)

        # Buy 1 low + 1 high, sell 2 mid (skew-adjusted per strike)
        sig_low = _skew_adjusted_vol(sigma, underlying_entry, low, is_call)
        sig_mid = _skew_adjusted_vol(sigma, underlying_entry, mid, is_call)
        sig_high = _skew_adjusted_vol(sigma, underlying_entry, high, is_call)
        low_entry = _bs_price(underlying_entry, low, T_entry, sig_low, is_call)
        mid_entry = _bs_price(underlying_entry, mid, T_entry, sig_mid, is_call)
        high_entry = _bs_price(underlying_entry, high, T_entry, sig_high, is_call)
        low_exit = _bs_price(underlying_exit, low, T_exit, sig_low, is_call)
        mid_exit = _bs_price(underlying_exit, mid, T_exit, sig_mid, is_call)
        high_exit = _bs_price(underlying_exit, high, T_exit, sig_high, is_call)

        debit = low_entry + high_entry - 2 * mid_entry
        exit_value = low_exit + high_exit - 2 * mid_exit
        if abs(debit) < 0.001:
            return 0.0
        pnl = exit_value - debit
        return (pnl / abs(debit)) * 100

    return 0.0


def _minute_of_day(ts: Any) -> int | None:
    """Extract minute-of-day (ET) from a timestamp. Returns None if unparseable."""
    try:
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(_et)
            return dt.hour * 60 + dt.minute
        elif isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(_et)
            return dt.hour * 60 + dt.minute
        elif hasattr(ts, "hour"):
            return ts.hour * 60 + ts.minute
    except Exception:
        pass
    return None


def _entry_date_from_candle(candles: pd.DataFrame, fill_bar: int) -> date | None:
    """Extract the calendar date from a candle's timestamp."""
    if fill_bar >= len(candles):
        return None
    ts = candles.iloc[fill_bar]["ts"]
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
        elif isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
        elif hasattr(ts, "date"):
            return ts.date()
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
    fitness_profit_factor = min(3.0, profit_factor if math.isfinite(profit_factor) else 3.0)
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
    # Cap at 5.0 to prevent inflated benchmarks from overfitting artifacts
    search_fitness = round(min(max(fitness_raw, 0.0), 5.0), 6)

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
