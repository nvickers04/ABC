"""Daily profitability cycle logs (JSON file + optional Postgres)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from typing import TYPE_CHECKING

from core.profit_config_state import get_active_profile_label, log_active_profit_config

if TYPE_CHECKING:
    from core.central_profit_config import ProfitConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = _REPO_ROOT / "logs"
_DDL = """
CREATE TABLE IF NOT EXISTS profit_cycle_logs (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_date DATE NOT NULL,
    cycle_id INTEGER NOT NULL,
    profit_profile TEXT,
    outcome TEXT NOT NULL,
    payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profit_cycle_logs_session
    ON profit_cycle_logs (session_date DESC, cycle_id DESC);
"""


@dataclass
class CycleTradeOutcome:
    """Last tool action in the cycle (if any)."""

    action: str = ""
    symbol: str = ""
    success: bool | None = None
    error: str = ""
    order_id: str = ""
    filled: bool | None = None


@dataclass
class CyclePnlMetrics:
    """Realized P&L and spend metrics for one cycle."""

    cycle_realized_pnl_usd: float = 0.0
    cumulative_realized_pnl_usd: float = 0.0
    llm_cost_usd: float = 0.0
    intraday_drawdown_pct: float = 0.0
    daily_loss_pct: float | None = None
    net_liquidation: float | None = None
    session_high_water: float | None = None


@dataclass
class CycleQualityMetrics:
    overall_quality: str = "unknown"
    risk_multiplier: float = 0.0
    execution_quality: float = 0.0
    working_memory_completeness: float = 0.0


@dataclass
class ProfitCycleRecord:
    ts: str
    session_date: str
    cycle_id: int
    profit_profile: str
    trading_mode: str
    outcome: str
    cooldown_seconds: int
    session: str
    cycle_summary: str = ""
    cycle_actions: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    pnl: CyclePnlMetrics = field(default_factory=CyclePnlMetrics)
    quality: CycleQualityMetrics = field(default_factory=CycleQualityMetrics)
    trade_outcome: CycleTradeOutcome = field(default_factory=CycleTradeOutcome)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def snapshot_profit_config(
    cfg: "ProfitConfig",
    *,
    profile_label: str | None = None,
) -> dict[str, Any]:
    """Serialize full ProfitConfig levers for the cycle log."""
    profile = profile_label or get_active_profile_label()
    tools = cfg.tools.tools
    return {
        "profit_profile": profile,
        "trading_mode": cfg.trading_mode,
        "risk": _model_dump(cfg.risk),
        "loop": _model_dump(cfg.loop),
        "memory": _model_dump(cfg.memory),
        "prompt": _model_dump(cfg.prompt),
        "tools": {
            "registered": len(tools),
            "enabled": sum(1 for t in tools.values() if t.enabled),
            "mutating_broker": sum(1 for t in tools.values() if t.mutates_broker),
        },
    }


def _daily_json_path(session_date: str | None = None) -> Path:
    d = session_date or date.today().isoformat()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"profit_cycles_{d}.json"


def _postgres_enabled() -> bool:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith(("postgresql://", "postgres://")):
        return True
    return all(os.getenv(k) for k in ("PGHOST", "PGDATABASE", "PGUSER"))


def _ensure_pg_table() -> None:
    try:
        from memory import get_db

        db = get_db()
        db.executescript(_DDL)
    except Exception as e:
        logger.debug("profit_cycle_logs DDL skipped: %s", e)


def _append_json(record: ProfitCycleRecord) -> Path:
    """Atomically append one entry (write temp file then replace)."""
    import tempfile

    path = _daily_json_path(record.session_date)
    payload: dict[str, Any]
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "profit_cycle_json corrupt; starting fresh file: %s",
                exc,
            )
            payload = {"date": record.session_date, "entries": []}
    else:
        payload = {"date": record.session_date, "entries": []}
    payload.setdefault("entries", []).append(record.to_dict())
    text = json.dumps(payload, indent=2, default=str)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def _append_postgres(record: ProfitCycleRecord) -> None:
    try:
        from memory import get_db

        _ensure_pg_table()
        db = get_db()
        db.execute(
            """
            INSERT INTO profit_cycle_logs
                (ts, session_date, cycle_id, profit_profile, outcome, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.ts,
                record.session_date,
                record.cycle_id,
                record.profit_profile,
                record.outcome,
                json.dumps(record.to_dict(), default=str),
            ),
        )
        db.commit()
    except Exception as e:
        logger.warning(
            "profit_cycle_logs postgres append failed (JSON log retained): %s | "
            "cycle_id=%s session_date=%s profile=%s",
            e,
            record.cycle_id,
            record.session_date,
            record.profit_profile,
        )


def collect_cycle_metrics(agent: Any | None) -> tuple[CyclePnlMetrics, CycleQualityMetrics, CycleTradeOutcome]:
    """Gather P&L, QualityMatrix, and last-tool outcome from a live agent."""
    pnl = CyclePnlMetrics()
    quality = CycleQualityMetrics()
    trade = CycleTradeOutcome()

    try:
        from data.cost_tracker import get_cost_tracker

        bs = get_cost_tracker().get_budget_summary()
        pnl.llm_cost_usd = float(bs.today_llm_cost)
        pnl.cumulative_realized_pnl_usd = float(bs.today_realized_pnl)
        prev = getattr(agent, "_profit_log_prev_realized_pnl", None) if agent else None
        if prev is not None:
            pnl.cycle_realized_pnl_usd = pnl.cumulative_realized_pnl_usd - float(prev)
        if agent is not None:
            agent._profit_log_prev_realized_pnl = pnl.cumulative_realized_pnl_usd
    except Exception as e:
        logger.debug("profit log cost_tracker: %s", e)

    if agent is not None:
        gw = getattr(agent, "gateway", None)
        if gw is not None:
            try:
                pnl.net_liquidation = float(getattr(gw, "net_liquidation", 0) or 0)
            except (TypeError, ValueError):
                pass
        shw = getattr(agent, "_session_high_water", None)
        if shw and pnl.net_liquidation is not None and shw > 0:
            pnl.session_high_water = float(shw)
            pnl.intraday_drawdown_pct = max(
                0.0, (float(shw) - pnl.net_liquidation) / float(shw) * 100.0
            )
        try:
            loss = agent._check_daily_loss()
            if loss is not None:
                pnl.daily_loss_pct = float(loss)
        except Exception:
            pass

        last = getattr(agent, "_profit_last_tool", None) or {}
        if isinstance(last, dict):
            trade.action = str(last.get("action", ""))
            trade.symbol = str(last.get("symbol", ""))
            trade.success = last.get("success")
            trade.error = str(last.get("error", ""))[:200]
            trade.order_id = str(last.get("order_id", ""))
            trade.filled = last.get("filled")

        try:
            ctx = agent._operating_context
            qm = getattr(ctx, "quality_matrix", None) or getattr(ctx, "quality", None)
            if qm is not None:
                quality.working_memory_completeness = float(
                    getattr(qm, "working_memory_completeness", 0) or 0
                )
        except Exception:
            pass

    try:
        from core.quality.quality_matrix import get_quality_matrix_service

        m = get_quality_matrix_service().get_matrix()
        quality.overall_quality = str(m.overall_quality)
        quality.risk_multiplier = float(m.risk_multiplier)
        quality.execution_quality = float(getattr(m, "execution_quality", 0) or 0)
    except Exception as e:
        logger.debug("profit log quality_matrix: %s", e)

    return pnl, quality, trade


def append_profit_cycle_log(
    cfg: "ProfitConfig",
    *,
    cycle_id: int,
    outcome: str,
    cooldown_seconds: int,
    session: str = "unknown",
    cycle_summary: str = "",
    cycle_actions: list[str] | None = None,
    agent: Any | None = None,
    profile_label: str | None = None,
) -> ProfitCycleRecord:
    """Persist one cycle record to daily JSON and Postgres (when configured)."""
    now = datetime.now(timezone.utc)
    session_date = now.strftime("%Y-%m-%d")
    pnl, quality, trade = collect_cycle_metrics(agent)
    label = profile_label or get_active_profile_label()
    snap = snapshot_profit_config(cfg, profile_label=label)
    record = ProfitCycleRecord(
        ts=now.isoformat(),
        session_date=session_date,
        cycle_id=cycle_id,
        profit_profile=label,
        trading_mode=str(snap.get("trading_mode", "")),
        outcome=outcome,
        cooldown_seconds=int(cooldown_seconds),
        session=session,
        cycle_summary=(cycle_summary or "")[:500],
        cycle_actions=list(cycle_actions or []),
        config=snap,
        pnl=pnl,
        quality=quality,
        trade_outcome=trade,
    )
    _append_json(record)
    if _postgres_enabled():
        _append_postgres(record)
    log_config_digest(cfg, context=f"cycle_{cycle_id}")
    return record


def load_daily_entries(session_date: str | None = None) -> list[dict[str, Any]]:
    """Load cycle entries for a day from JSON (Postgres fallback if file missing)."""
    d = session_date or date.today().isoformat()
    path = _daily_json_path(d)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return list(data.get("entries") or [])
        except (json.JSONDecodeError, OSError):
            return []
    if _postgres_enabled():
        try:
            from memory import get_db

            rows = get_db().execute(
                """
                SELECT payload FROM profit_cycle_logs
                WHERE session_date = ?
                ORDER BY cycle_id ASC
                """,
                (d,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                raw = row["payload"] if isinstance(row, dict) else row[0]
                if isinstance(raw, str):
                    out.append(json.loads(raw))
                elif isinstance(raw, dict):
                    out.append(raw)
            return out
        except Exception as e:
            logger.debug("load_daily_entries postgres: %s", e)
    return []


def _parse_entry_ts(entry: dict[str, Any]) -> datetime | None:
    raw = entry.get("ts")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def load_entries_since(since: datetime) -> list[dict[str, Any]]:
    """Load cycle log entries with ``ts`` >= ``since`` (UTC) from JSON files and Postgres."""
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    entries: list[dict[str, Any]] = []

    if LOG_DIR.is_dir():
        for path in sorted(LOG_DIR.glob("profit_cycles_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for row in data.get("entries") or []:
                    if isinstance(row, dict):
                        entries.append(row)
            except (json.JSONDecodeError, OSError):
                continue

    if _postgres_enabled():
        try:
            from memory import get_db

            rows = get_db().execute(
                """
                SELECT payload FROM profit_cycle_logs
                WHERE ts >= ?
                ORDER BY ts ASC
                """,
                (since.isoformat(),),
            ).fetchall()
            for row in rows:
                raw = row["payload"] if isinstance(row, dict) else row[0]
                if isinstance(raw, str):
                    entries.append(json.loads(raw))
                elif isinstance(raw, dict):
                    entries.append(raw)
        except Exception as e:
            logger.debug("load_entries_since postgres: %s", e)

    # Dedupe by (ts, cycle_id, profile) — prefer later source (postgres wins if both)
    seen: set[tuple[str, int, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in sorted(entries, key=lambda e: str(e.get("ts") or "")):
        key = (
            str(row.get("ts") or ""),
            int(row.get("cycle_id") or 0),
            str(row.get("profit_profile") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        ts = _parse_entry_ts(row)
        if ts is not None and ts >= since:
            unique.append(row)
    return unique


def find_latest_log_file() -> Path | None:
    """Return the newest ``logs/profit_cycles_*.json`` by mtime."""
    if not LOG_DIR.is_dir():
        return None
    files = sorted(LOG_DIR.glob("profit_cycles_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def build_daily_summary(session_date: str | None = None) -> dict[str, Any]:
    """Aggregate today's (or ``session_date``) profit cycle log entries."""
    from core.profit_summary import aggregate_entries

    entries = load_daily_entries(session_date)
    summary = aggregate_entries(entries)
    summary["session_date"] = session_date or date.today().isoformat()
    summary["log_file"] = str(_daily_json_path(summary["session_date"]))
    return summary


def log_config_digest(cfg: "ProfitConfig", *, context: str = "cycle") -> None:
    """Emit a compact ProfitConfig digest at DEBUG (ops / post-mortems)."""
    try:
        snap = snapshot_profit_config(cfg)
        risk = snap.get("risk") or {}
        logger.debug(
            "profit_config_digest context=%s profile=%s mode=%s "
            "loss_cap=%s%% dd_cap=%s%% llm_cap=$%s cooldown=%ss",
            context,
            snap.get("profit_profile"),
            snap.get("trading_mode"),
            risk.get("max_daily_loss_pct"),
            risk.get("intraday_drawdown_pct"),
            risk.get("max_daily_llm_cost"),
            (snap.get("loop") or {}).get("cooldown_seconds"),
        )
    except Exception as e:
        logger.debug("profit_config_digest skipped: %s", e)
