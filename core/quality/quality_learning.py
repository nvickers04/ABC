"""Historical learning for QualityMatrix scoring weights (bounded vs profile base)."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.loop_config import LoopConfig, get_loop_config, install_loop_config
from core.memory_config import get_memory_config
from core.profit_config_state import get_active_profile_label

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATE_PATH = _REPO_ROOT / "data" / "quality_matrix_learned.json"
_LOCK = threading.RLock()

LEARNABLE_WEIGHT_KEYS: tuple[str, ...] = (
    "symbol_exec_quality_base",
    "symbol_exec_quality_gap_coeff",
    "global_exec_formula_base",
    "global_exec_formula_gap_coeff",
    "symbol_exec_poor_threshold",
    "global_exec_degraded_threshold",
)

# Direction: +1 means higher learned value when reward is positive
_REWARD_DIRECTION: dict[str, float] = {
    "symbol_exec_quality_gap_coeff": -1.0,
    "global_exec_formula_gap_coeff": -1.0,
    "symbol_exec_quality_base": 1.0,
    "global_exec_formula_base": 1.0,
    "symbol_exec_poor_threshold": -1.0,
    "global_exec_degraded_threshold": -1.0,
}

# True when a larger numeric value makes QualityMatrix posture more aggressive (higher risk).
_RISKIER_IF_HIGHER: dict[str, bool] = {
    "symbol_exec_quality_base": True,
    "symbol_exec_quality_gap_coeff": False,
    "global_exec_formula_base": True,
    "global_exec_formula_gap_coeff": False,
    "symbol_exec_poor_threshold": False,
    "global_exec_degraded_threshold": False,
}

_DDL = """
CREATE TABLE IF NOT EXISTS quality_matrix_trade_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_date TEXT,
    symbol TEXT,
    profit_profile TEXT,
    won INTEGER NOT NULL DEFAULT 0,
    realized_rr REAL,
    pnl_usd REAL,
    source TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_qm_trade_outcomes_ts
    ON quality_matrix_trade_outcomes (ts DESC);
CREATE INDEX IF NOT EXISTS idx_qm_trade_outcomes_profile
    ON quality_matrix_trade_outcomes (profit_profile, ts DESC);
"""


@dataclass
class LearnedWeightState:
    profile_label: str = "balanced"
    base_weights: dict[str, float] = field(default_factory=dict)
    learned_weights: dict[str, float] = field(default_factory=dict)
    trades_since_refit: int = 0
    total_trades: int = 0
    last_refit_ts: str = ""
    last_reward: float = 0.0
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_label": self.profile_label,
            "base_weights": dict(self.base_weights),
            "learned_weights": dict(self.learned_weights),
            "trades_since_refit": self.trades_since_refit,
            "total_trades": self.total_trades,
            "last_refit_ts": self.last_refit_ts,
            "last_reward": self.last_reward,
            "recent_outcomes": list(self.recent_outcomes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LearnedWeightState:
        recent = raw.get("recent_outcomes") or []
        return cls(
            profile_label=str(raw.get("profile_label") or "balanced"),
            base_weights={k: float(v) for k, v in (raw.get("base_weights") or {}).items()},
            learned_weights={k: float(v) for k, v in (raw.get("learned_weights") or {}).items()},
            trades_since_refit=int(raw.get("trades_since_refit") or 0),
            total_trades=int(raw.get("total_trades") or 0),
            last_refit_ts=str(raw.get("last_refit_ts") or ""),
            last_reward=float(raw.get("last_reward") or 0.0),
            recent_outcomes=[r for r in recent if isinstance(r, dict)],
        )


def learning_enabled() -> bool:
    try:
        from core.central_profit_config import get_profit_config

        return bool(get_profit_config().learn_from_history)
    except Exception:
        mem = get_memory_config()
        return bool(mem.quality_matrix_learn_from_history)


def max_drift_fraction() -> float:
    mem = get_memory_config()
    pct = float(mem.quality_matrix_learn_max_drift_pct)
    return max(0.0, min(0.5, pct / 100.0))


def refit_interval() -> int:
    mem = get_memory_config()
    return max(1, int(mem.quality_matrix_learn_refit_interval))


def profile_baseline_loop_config(profile_label: str | None = None) -> LoopConfig:
    """LoopConfig for the named ProfitConfig profile without QualityMatrix learned overrides."""
    from core.loop_config import reload_loop_config
    from core.profit_profiles import (
        VALID_PROFILES,
        apply_loop_profile,
        is_evolved_profile,
        load_evolved_profile_registry,
    )

    label = (profile_label or get_active_profile_label()).strip().lower()
    lc = reload_loop_config()
    if label in VALID_PROFILES:
        return apply_loop_profile(lc, label)  # type: ignore[arg-type]
    if is_evolved_profile(label):
        entry = load_evolved_profile_registry()[label]
        lc = apply_loop_profile(lc, entry.base_profile)  # type: ignore[arg-type]
        loop_patch = entry.patches.get("loop") or {}
        if loop_patch:
            return lc.model_copy(update=loop_patch)
        return lc
    try:
        from core.central_profit_config import get_profit_config

        return get_profit_config().loop
    except Exception:
        return get_loop_config()


def capture_base_weights(profile_label: str | None = None) -> dict[str, float]:
    """Snapshot ProfitConfig profile loop scoring fields (not learned overrides)."""
    lc = profile_baseline_loop_config(profile_label)
    return {k: float(getattr(lc, k)) for k in LEARNABLE_WEIGHT_KEYS}


def clamp_not_riskier_than_profile(key: str, profile_value: float, proposed: float) -> float:
    """Clamp one learned weight so it cannot be more aggressive than the profile baseline."""
    if _RISKIER_IF_HIGHER.get(key, True):
        return min(proposed, profile_value)
    return max(proposed, profile_value)


def guard_learned_weights_against_profile(
    profile_bases: dict[str, float],
    learned: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Ensure learned weights are not riskier than ``profile_bases`` (ProfitConfig profile)."""
    out: dict[str, float] = {}
    notes: list[str] = []
    for key, val in learned.items():
        base = profile_bases.get(key)
        if base is None:
            out[key] = float(val)
            continue
        clamped = clamp_not_riskier_than_profile(key, float(base), float(val))
        if clamped != val:
            notes.append(f"{key}: {val} -> {clamped} (profile risk guard)")
        out[key] = round(clamped, 6)
    return out, notes


def _ensure_tables() -> None:
    try:
        from memory import get_db

        db = get_db()
        db.executescript(_DDL)
    except Exception as e:
        logger.debug("quality_matrix_trade_outcomes DDL skipped: %s", e)


def load_state() -> LearnedWeightState:
    with _LOCK:
        if not _STATE_PATH.is_file():
            return LearnedWeightState()
        try:
            raw = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return LearnedWeightState.from_dict(raw)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("quality_matrix learned state unreadable: %s", e)
        return LearnedWeightState()


def save_state(state: LearnedWeightState) -> None:
    with _LOCK:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, _STATE_PATH)


def sync_bases_for_current_profile(force: bool = False) -> LearnedWeightState:
    """Align base weights with active profile; reset learned to base when profile changes."""
    label = get_active_profile_label()
    state = load_state()
    if force or state.profile_label != label or not state.base_weights:
        state.profile_label = label
        state.base_weights = capture_base_weights(label)
        state.learned_weights = dict(state.base_weights)
        state.trades_since_refit = 0
        save_state(state)
    return state


def clamp_learned_value(base: float, proposed: float) -> float:
    """Clamp ``proposed`` to ±max_drift_pct of ``base``."""
    if base == 0:
        return proposed
    drift = max_drift_fraction()
    lo = base * (1.0 - drift)
    hi = base * (1.0 + drift)
    return max(lo, min(hi, proposed))


def _append_recent_outcome(state: LearnedWeightState, norm: dict[str, Any]) -> None:
    mem = get_memory_config()
    limit = max(1, int(mem.quality_matrix_learn_history_limit))
    row = {
        "won": norm["won"],
        "realized_rr": norm["realized_rr"],
        "profit_profile": norm["profit_profile"],
        "symbol": norm["symbol"],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    state.recent_outcomes.insert(0, row)
    del state.recent_outcomes[limit:]


def _load_history_rows(state: LearnedWeightState, *, lookback: int | None = None) -> list[dict[str, Any]]:
    mem = get_memory_config()
    limit = lookback or int(mem.quality_matrix_learn_history_limit)
    rows: list[dict[str, Any]] = []
    try:
        from memory import get_db

        db = get_db()
        cur = db.execute(
            """
            SELECT won, realized_rr, profit_profile, symbol, ts
            FROM quality_matrix_trade_outcomes
            WHERE profit_profile = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (state.profile_label, limit),
        ).fetchall()
        for r in cur:
            rows.append(dict(r) if hasattr(r, "keys") else {"won": r[0], "realized_rr": r[1]})
    except Exception as e:
        logger.debug("quality_matrix history from DB skipped: %s", e)
    if rows:
        return rows
    filtered = [
        r
        for r in state.recent_outcomes
        if str(r.get("profit_profile") or state.profile_label).lower() == state.profile_label.lower()
    ]
    return filtered[:limit]


def _compute_reward(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    wins = sum(1 for r in rows if r.get("won"))
    n = len(rows)
    win_rate = wins / n
    rr_vals = [float(r["realized_rr"]) for r in rows if r.get("realized_rr") is not None]
    avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0
    reward = (win_rate - 0.5) * 2.0 + math.tanh(avg_rr) * 0.5
    return max(-1.0, min(1.0, reward))


def refit_weights_from_history(*, lookback: int | None = None) -> dict[str, Any]:
    """Recompute learned weights from stored trade outcomes (bounded vs base)."""
    if not learning_enabled():
        return {"skipped": True, "reason": "learning_disabled"}

    _ensure_tables()
    state = sync_bases_for_current_profile()
    mem = get_memory_config()
    limit = lookback or int(mem.quality_matrix_learn_history_limit)

    rows = _load_history_rows(state, lookback=limit)
    if not rows:
        return {"skipped": True, "reason": "no_history"}

    reward = _compute_reward(rows)
    lr = float(mem.quality_matrix_learn_rate)
    new_learned: dict[str, float] = {}
    for key, base in state.base_weights.items():
        direction = _REWARD_DIRECTION.get(key, 1.0)
        delta = reward * lr * direction
        proposed = float(base) * (1.0 + max(-max_drift_fraction(), min(max_drift_fraction(), delta)))
        new_learned[key] = round(clamp_learned_value(float(base), proposed), 6)

    new_learned, risk_notes = guard_learned_weights_against_profile(state.base_weights, new_learned)
    state.learned_weights = new_learned
    state.trades_since_refit = 0
    state.last_refit_ts = datetime.now(timezone.utc).isoformat()
    state.last_reward = round(reward, 4)
    state.total_trades += 0  # unchanged
    save_state(state)
    apply_learned_weights_to_loop_config()

    summary = {
        "refitted": True,
        "profile": state.profile_label,
        "reward": state.last_reward,
        "samples": len(rows),
        "learned_weights": dict(new_learned),
        "max_drift_pct": mem.quality_matrix_learn_max_drift_pct,
    }
    if risk_notes:
        summary["risk_guard_clamps"] = risk_notes
        logger.info(
            "QualityMatrix risk guard applied %d clamp(s): %s",
            len(risk_notes),
            "; ".join(risk_notes[:5]),
        )
    logger.info(
        "QualityMatrix weights refit: profile=%s reward=%.3f samples=%d",
        state.profile_label,
        reward,
        len(rows),
    )
    return summary


def apply_learned_weights_to_loop_config() -> None:
    """Install LoopConfig override with bounded learned scoring weights."""
    state = load_state()
    if not state.learned_weights:
        return
    bases = state.base_weights or capture_base_weights(state.profile_label)
    safe, _notes = guard_learned_weights_against_profile(bases, state.learned_weights)
    if safe != state.learned_weights:
        state.learned_weights = safe
        save_state(state)
    lc = get_loop_config()
    updates = {k: safe[k] for k in LEARNABLE_WEIGHT_KEYS if k in safe}
    if not updates:
        return
    install_loop_config(lc.model_copy(update=updates))


def get_scoring_loop_config() -> LoopConfig:
    """LoopConfig used by QualityMatrix.populate (includes learned overrides when enabled)."""
    if not learning_enabled():
        return get_loop_config()
    state = load_state()
    if not state.learned_weights:
        sync_bases_for_current_profile()
        state = load_state()
    if not state.learned_weights:
        return get_loop_config()
    bases = state.base_weights or capture_base_weights(state.profile_label)
    safe, _notes = guard_learned_weights_against_profile(bases, state.learned_weights)
    lc = get_loop_config()
    updates = {k: safe[k] for k in LEARNABLE_WEIGHT_KEYS if k in safe}
    return lc.model_copy(update=updates)


def _normalize_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    won = outcome.get("won")
    if won is None:
        won = outcome.get("win")
    if won is None:
        pnl = float(outcome.get("pnl_usd") or outcome.get("pnl") or outcome.get("realized_pnl") or 0)
        won = pnl > 0
    rr = outcome.get("realized_rr")
    if rr is None:
        rr = outcome.get("rr")
    try:
        rr_f = float(rr) if rr is not None else None
    except (TypeError, ValueError):
        rr_f = None
    return {
        "symbol": str(outcome.get("symbol") or "").upper() or None,
        "profit_profile": str(
            outcome.get("profit_profile") or outcome.get("profile") or get_active_profile_label()
        ).strip().lower(),
        "won": 1 if bool(won) else 0,
        "realized_rr": rr_f,
        "pnl_usd": float(outcome.get("pnl_usd") or outcome.get("pnl") or 0),
        "source": str(outcome.get("source") or "live"),
        "session_date": str(outcome.get("session_date") or "")[:10] or None,
        "payload": outcome,
    }


def store_trade_outcome(outcome: dict[str, Any]) -> int | None:
    """Persist one trade outcome row; return row id when available."""
    norm = _normalize_outcome(outcome)
    state = load_state()
    _append_recent_outcome(state, norm)
    save_state(state)
    _ensure_tables()
    try:
        from memory import get_db

        db = get_db()
        row = db.execute(
            """
            INSERT INTO quality_matrix_trade_outcomes
                (ts, session_date, symbol, profit_profile, won, realized_rr, pnl_usd, source, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                norm["session_date"],
                norm["symbol"],
                norm["profit_profile"],
                norm["won"],
                norm["realized_rr"],
                norm["pnl_usd"],
                norm["source"],
                json.dumps(norm["payload"], default=str),
            ),
        ).fetchone()
        db.commit()
        if row:
            rid = row["id"] if hasattr(row, "keys") else row[0]
            return int(rid) if rid is not None else None
    except Exception as e:
        logger.debug("quality_matrix_trade_outcomes insert (returning) failed: %s", e)
        try:
            from memory import get_db

            db = get_db()
            db.execute(
                """
                INSERT INTO quality_matrix_trade_outcomes
                    (ts, session_date, symbol, profit_profile, won, realized_rr, pnl_usd, source, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    norm["session_date"],
                    norm["symbol"],
                    norm["profit_profile"],
                    norm["won"],
                    norm["realized_rr"],
                    norm["pnl_usd"],
                    norm["source"],
                    json.dumps(norm["payload"], default=str),
                ),
            )
            db.commit()
        except Exception as e2:
            logger.warning("quality_matrix_trade_outcomes insert failed: %s", e2)
            return None
    return None


def record_trade_outcome_and_maybe_refit(outcome: dict[str, Any], *, force_refit: bool = False) -> dict[str, Any]:
    """Store outcome and refit weights every ``refit_interval`` trades."""
    if not learning_enabled():
        return {"skipped": True, "reason": "learning_disabled"}

    sync_bases_for_current_profile()
    store_trade_outcome(outcome)

    state = load_state()
    state.trades_since_refit += 1
    state.total_trades += 1
    save_state(state)

    if force_refit or state.trades_since_refit >= refit_interval():
        return refit_weights_from_history()

    return {
        "stored": True,
        "trades_since_refit": state.trades_since_refit,
        "refit_in": refit_interval() - state.trades_since_refit,
    }


_CYCLE_LOG_SKIP_ACTIONS = frozenset(
    {"", "quality_status", "briefing", "market_hours", "market_hours_check"}
)


def trade_outcomes_from_cycle_logs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract trade-like outcomes from profit cycle log entries (dashboard logger)."""
    outcomes: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda e: str(e.get("ts") or "")):
        trade = entry.get("trade_outcome") or {}
        action = str(trade.get("action") or "").strip().lower()
        if action in _CYCLE_LOG_SKIP_ACTIONS:
            continue
        pnl = float(entry.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
        symbol = str(trade.get("symbol") or "").strip().upper()
        order_id = str(trade.get("order_id") or "").strip()
        if not action:
            continue
        if pnl == 0.0 and not order_id:
            continue
        outcomes.append(
            {
                "symbol": symbol or None,
                "won": pnl > 0,
                "pnl_usd": pnl,
                "realized_rr": None,
                "profit_profile": str(entry.get("profit_profile") or get_active_profile_label()),
                "session_date": str(entry.get("session_date") or "")[:10] or None,
                "source": "profit_cycle_log",
                "action": action,
                "ts": entry.get("ts"),
                "filled": trade.get("filled"),
                "order_id": order_id or None,
            }
        )
    return outcomes


def train_from_trade_outcomes(
    outcomes: list[dict[str, Any]],
    *,
    refit_at_end: bool = True,
) -> dict[str, Any]:
    """Replay trade outcomes through learning; optional final refit."""
    if not learning_enabled():
        return {"skipped": True, "reason": "learning_disabled", "ingested": 0}

    sync_bases_for_current_profile(force=True)
    ingested = 0
    last: dict[str, Any] = {}
    for outcome in outcomes:
        last = record_trade_outcome_and_maybe_refit(outcome, force_refit=False)
        ingested += 1
    if refit_at_end and ingested:
        last = refit_weights_from_history()
    summary = dict(last)
    summary["ingested"] = ingested
    return summary


def persist_learned_weights_to_active_profile(
    *,
    profile_label: str | None = None,
) -> dict[str, Any]:
    """Write learned loop weights into ``data/evolved_profiles.json`` for the active profile."""
    from core.profit_profiles import (
        VALID_PROFILES,
        EvolvedProfileEntry,
        is_evolved_profile,
        load_evolved_profile_registry,
        save_evolved_profile,
    )

    state = load_state()
    label = (profile_label or state.profile_label or get_active_profile_label()).strip().lower()
    if not state.learned_weights:
        return {"skipped": True, "reason": "no_learned_weights", "profile": label}

    bases = state.base_weights or capture_base_weights(label)
    safe, risk_notes = guard_learned_weights_against_profile(bases, state.learned_weights)
    loop_patches = {k: safe[k] for k in LEARNABLE_WEIGHT_KEYS if k in safe}
    qm_metrics = {
        "last_persist_ts": datetime.now(timezone.utc).isoformat(),
        "last_reward": state.last_reward,
        "loop_patches": dict(loop_patches),
    }

    if is_evolved_profile(label):
        entry = load_evolved_profile_registry()[label]
        patches = dict(entry.patches)
        loop = dict(patches.get("loop") or {})
        loop.update(loop_patches)
        patches["loop"] = loop
        updated = EvolvedProfileEntry(
            base_profile=entry.base_profile,
            patches=patches,
            note=entry.note or "QualityMatrix learned loop weights",
            genes=entry.genes,
            metrics={**entry.metrics, "quality_matrix_learning": qm_metrics},
        )
        path = save_evolved_profile(label, updated)
        target = label
    elif label in VALID_PROFILES:
        target = f"{label}_qm_learned"
        updated = EvolvedProfileEntry(
            base_profile=label,  # type: ignore[arg-type]
            patches={"loop": loop_patches},
            note=f"QualityMatrix weights from cycle-log training (base={label})",
            metrics={"quality_matrix_learning": qm_metrics},
        )
        path = save_evolved_profile(target, updated)
    else:
        return {"skipped": True, "reason": "unknown_profile", "profile": label}

    apply_learned_weights_to_loop_config()
    result = {
        "saved": True,
        "profile": target,
        "path": str(path),
        "loop_patches": loop_patches,
        "promote_env": f"{target}" if target != label else None,
    }
    if risk_notes:
        result["risk_guard_clamps"] = risk_notes
    return result


def ingest_backtest_trades_and_refit(result: Any, profile_label: str) -> dict[str, Any]:
    """Feed simulation trade log into learning and refit once (offline training)."""
    if not learning_enabled():
        return {"skipped": True, "reason": "learning_disabled"}

    sync_bases_for_current_profile(force=True)
    count = 0
    for row in getattr(result, "trade_log", None) or []:
        d = row.__dict__ if hasattr(row, "__dict__") else row
        if not isinstance(d, dict):
            d = {
                "symbol": getattr(row, "symbol", ""),
                "realized_rr": getattr(row, "realized_rr", None),
                "pnl_usd": getattr(row, "realized_pnl", None),
                "profit_profile": getattr(row, "profit_profile", profile_label),
                "session_date": getattr(row, "session_date", None),
            }
        record_trade_outcome_and_maybe_refit(
            {
                **d,
                "profit_profile": d.get("profit_profile") or profile_label,
                "won": float(d.get("realized_pnl") or d.get("pnl_usd") or 0) > 0,
                "source": "simulation",
            },
            force_refit=False,
        )
        count += 1
    summary = refit_weights_from_history()
    summary["ingested_trades"] = count
    return summary


def reset_learning_for_tests() -> None:
    """Clear learned state file and loop override (unit tests)."""
    with _LOCK:
        if _STATE_PATH.is_file():
            _STATE_PATH.unlink(missing_ok=True)
    install_loop_config(None)


__all__ = [
    "LEARNABLE_WEIGHT_KEYS",
    "apply_learned_weights_to_loop_config",
    "capture_base_weights",
    "clamp_not_riskier_than_profile",
    "get_scoring_loop_config",
    "guard_learned_weights_against_profile",
    "profile_baseline_loop_config",
    "ingest_backtest_trades_and_refit",
    "learning_enabled",
    "persist_learned_weights_to_active_profile",
    "record_trade_outcome_and_maybe_refit",
    "refit_weights_from_history",
    "reset_learning_for_tests",
    "sync_bases_for_current_profile",
    "trade_outcomes_from_cycle_logs",
    "train_from_trade_outcomes",
]
