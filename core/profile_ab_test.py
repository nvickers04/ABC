"""Live ProfitConfig A/B testing on paper trading (rotation, comparative cycle logs)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.live_profile_optimize import score_profile_entries
from core.profit_cycle_logger import load_entries_since
from core.profit_profiles import PROFIT_PROFILE_ENV, normalize_profit_profile

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STATE_PATH = _REPO_ROOT / "data" / "profile_ab_test_state.json"
_LOG_DIR = _REPO_ROOT / "logs"

_ab_log_meta: ContextVar[dict[str, Any] | None] = ContextVar("ab_test_log_meta", default=None)
_ab_active: ContextVar[bool] = ContextVar("ab_test_active", default=False)

AB_MODES = ("rotation", "session")
DEFAULT_DURATION_DAYS = 30
MIN_CYCLES_FOR_WINNER = 3


@dataclass
class AbTestState:
    """Persisted A/B run metadata."""

    profiles: tuple[str, str]
    mode: str = "rotation"
    duration_days: int = DEFAULT_DURATION_DAYS
    run_id: str = ""
    started_at: str = ""
    ends_at: str = ""
    cycle_index: int = 0
    last_daily_report_date: str = ""
    last_session_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": list(self.profiles),
            "mode": self.mode,
            "duration_days": self.duration_days,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ends_at": self.ends_at,
            "cycle_index": self.cycle_index,
            "last_daily_report_date": self.last_daily_report_date,
            "last_session_date": self.last_session_date,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AbTestState:
        profs = raw.get("profiles") or []
        if len(profs) < 2:
            profs = ["balanced", "conservative"]
        return cls(
            profiles=(str(profs[0]).lower(), str(profs[1]).lower()),
            mode=str(raw.get("mode") or "rotation"),
            duration_days=int(raw.get("duration_days") or DEFAULT_DURATION_DAYS),
            run_id=str(raw.get("run_id") or ""),
            started_at=str(raw.get("started_at") or ""),
            ends_at=str(raw.get("ends_at") or ""),
            cycle_index=int(raw.get("cycle_index") or 0),
            last_daily_report_date=str(raw.get("last_daily_report_date") or ""),
            last_session_date=str(raw.get("last_session_date") or ""),
        )


def is_ab_test_active() -> bool:
    return bool(_ab_active.get())


def get_ab_log_meta() -> dict[str, Any] | None:
    return _ab_log_meta.get()


def parse_ab_profiles(raw: str) -> tuple[str, str]:
    """Parse ``--ab-test`` value into exactly two distinct profile names."""
    from core.profit_profiles import normalize_profit_profile

    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            "--ab-test requires exactly two comma-separated profiles, e.g. conservative,balanced"
        )
    a = str(normalize_profit_profile(parts[0]))
    b = str(normalize_profit_profile(parts[1]))
    if a == b:
        raise ValueError("--ab-test profiles must be distinct")
    return (a, b)


def validate_ab_test_startup(*, account: str, trading_mode: str | None) -> None:
    """Refuse live A/B; paper-only guardrail."""
    if str(account).lower() == "live":
        raise SystemExit("A/B test refuses --account live (paper only).")
    mode = (trading_mode or os.getenv("TRADING_MODE", "paper")).strip().lower()
    if mode == "live":
        raise SystemExit("A/B test refuses TRADING_MODE=live (paper or aggressive_paper only).")
    if os.getenv("IBKR_ACCOUNT_TYPE", "paper").strip().lower() == "live":
        raise SystemExit("A/B test refuses IBKR_ACCOUNT_TYPE=live.")


def load_state() -> AbTestState | None:
    if not _STATE_PATH.is_file():
        return None
    try:
        raw = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return AbTestState.from_dict(raw)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("profile_ab_test state unreadable: %s", e)
    return None


def save_state(state: AbTestState) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_PATH)


def init_ab_test_state(
    profiles: tuple[str, str],
    *,
    mode: str = "rotation",
    duration_days: int = DEFAULT_DURATION_DAYS,
) -> AbTestState:
    now = datetime.now(timezone.utc)
    ends = now + timedelta(days=max(1, duration_days))
    state = AbTestState(
        profiles=profiles,
        mode=mode if mode in AB_MODES else "rotation",
        duration_days=max(1, duration_days),
        run_id=uuid.uuid4().hex[:12],
        started_at=now.isoformat(),
        ends_at=ends.isoformat(),
    )
    save_state(state)
    return state


def duration_expired(state: AbTestState) -> bool:
    try:
        ends = datetime.fromisoformat(state.ends_at.replace("Z", "+00:00"))
        if ends.tzinfo is None:
            ends = ends.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= ends
    except ValueError:
        return False


def select_profile_for_cycle(
    state: AbTestState,
    *,
    session_date: str | None = None,
) -> str:
    """Pick active arm: per-cycle rotation or per-session day alternation."""
    a, b = state.profiles
    if state.mode == "session":
        day = session_date or date.today().isoformat()
        if state.last_session_date != day:
            state.last_session_date = day
        # Stable assignment per calendar day (even/odd day index from start)
        try:
            start = datetime.fromisoformat(state.started_at.replace("Z", "+00:00")).date()
            cur = date.fromisoformat(day[:10])
            day_num = (cur - start).days
        except ValueError:
            day_num = state.cycle_index
        return a if day_num % 2 == 0 else b
    return a if state.cycle_index % 2 == 0 else b


def activate_ab_profile(profile: str, *, state: AbTestState) -> Any:
    """Reload ProfitConfig singleton for one A/B arm (shared process, one IBKR connection)."""
    from core.central_profit_config import get_profit_config
    from core.profit_config_state import set_active_profile_label

    label = normalize_profit_profile(profile)
    os.environ[PROFIT_PROFILE_ENV] = label
    set_active_profile_label(label)
    _ab_active.set(True)
    _ab_log_meta.set(
        {
            "run_id": state.run_id,
            "profiles": list(state.profiles),
            "mode": state.mode,
            "arm": label,
            "cycle_index": state.cycle_index,
            "duration_days": state.duration_days,
            "ends_at": state.ends_at,
        }
    )
    return get_profit_config().reload(dotenv=False)


def _entries_for_ab_run(state: AbTestState) -> list[dict[str, Any]]:
    since = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    entries = load_entries_since(since)
    arms = set(state.profiles)
    return [e for e in entries if str(e.get("profit_profile") or "").lower() in arms]


def build_live_comparison(state: AbTestState) -> dict[str, Any]:
    """Real-time comparative P&L for both arms from cycle logs since run start."""
    entries = _entries_for_ab_run(state)
    by_arm: dict[str, list[dict[str, Any]]] = {p: [] for p in state.profiles}
    for e in entries:
        prof = str(e.get("profit_profile") or "").lower()
        if prof in by_arm:
            by_arm[prof].append(e)

    arms: dict[str, Any] = {}
    for prof in state.profiles:
        arm_entries = by_arm.get(prof) or []
        arms[prof] = score_profile_entries(arm_entries)

    leader = max(
        state.profiles,
        key=lambda p: (arms[p].get("composite_score", 0), arms[p].get("total_cycle_pnl_usd", 0)),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": state.run_id,
        "profiles": list(state.profiles),
        "mode": state.mode,
        "cycle_index": state.cycle_index,
        "entries_total": len(entries),
        "arms": arms,
        "leader": leader,
    }


def format_live_comparison(comp: dict[str, Any]) -> str:
    lines = [
        "  [A/B live]",
        f"    run={comp.get('run_id')}  mode={comp.get('mode')}  cycles_logged={comp.get('entries_total')}",
    ]
    for prof in comp.get("profiles") or []:
        m = (comp.get("arms") or {}).get(prof) or {}
        mark = " <-- leading" if prof == comp.get("leader") else ""
        lines.append(
            f"    {prof:14}  pnl=${m.get('total_cycle_pnl_usd', 0):+,.2f}  "
            f"composite={m.get('composite_score', 0):.3f}  cycles={m.get('cycles', 0)}{mark}"
        )
    return "\n".join(lines)


def build_daily_winner_report(
    state: AbTestState,
    *,
    session_date: str | None = None,
) -> dict[str, Any]:
    """Daily winner for one session date (default: today UTC)."""
    day = (session_date or date.today().isoformat())[:10]
    entries = [
        e
        for e in _entries_for_ab_run(state)
        if str(e.get("session_date") or "")[:10] == day
    ]
    by_arm: dict[str, list[dict[str, Any]]] = {p: [] for p in state.profiles}
    for e in entries:
        prof = str(e.get("profit_profile") or "").lower()
        if prof in by_arm:
            by_arm[prof].append(e)

    scored: dict[str, dict[str, Any]] = {}
    for prof in state.profiles:
        scored[prof] = score_profile_entries(by_arm.get(prof) or [])

    eligible = [p for p in state.profiles if scored[p].get("cycles", 0) >= MIN_CYCLES_FOR_WINNER]
    if eligible:
        winner = max(
            eligible,
            key=lambda p: (scored[p].get("composite_score", 0), scored[p].get("total_cycle_pnl_usd", 0)),
        )
        confidence = "high"
    elif any(scored[p].get("cycles", 0) > 0 for p in state.profiles):
        winner = max(
            state.profiles,
            key=lambda p: (scored[p].get("composite_score", 0), scored[p].get("total_cycle_pnl_usd", 0)),
        )
        confidence = "low"
    else:
        winner = state.profiles[0]
        confidence = "none"

    return {
        "session_date": day,
        "run_id": state.run_id,
        "profiles": list(state.profiles),
        "mode": state.mode,
        "winner": winner,
        "confidence": confidence,
        "arms": scored,
        "cycles_today": len(entries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def format_daily_winner_report(report: dict[str, Any]) -> str:
    lines = [
        "=" * 64,
        "  ProfitConfig A/B — daily winner",
        "=" * 64,
        f"  Session:     {report.get('session_date')}",
        f"  Run:         {report.get('run_id')}",
        f"  Mode:        {report.get('mode')}",
        f"  Winner:      {report.get('winner')}  [{report.get('confidence')} confidence]",
        "",
        "  Arms (today)",
    ]
    for prof in report.get("profiles") or []:
        m = (report.get("arms") or {}).get(prof) or {}
        flag = " *" if prof == report.get("winner") else ""
        lines.append(
            f"    {prof:14}  pnl=${m.get('total_cycle_pnl_usd', 0):+,.2f}  "
            f"composite={m.get('composite_score', 0):.3f}  cycles={m.get('cycles', 0)}{flag}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


def append_daily_report_json(report: dict[str, Any]) -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = str(report.get("session_date") or date.today().isoformat())
    path = _LOG_DIR / f"ab_test_daily_{day}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def build_final_report(state: AbTestState) -> dict[str, Any]:
    """End-of-run summary across the full A/B window."""
    comp = build_live_comparison(state)
    daily_reports: list[dict[str, Any]] = []
    if _LOG_DIR.is_dir():
        for p in sorted(_LOG_DIR.glob("ab_test_daily_*.json")):
            try:
                daily_reports.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
    winner_counts: dict[str, int] = {p: 0 for p in state.profiles}
    for rep in daily_reports:
        if rep.get("run_id") == state.run_id:
            w = str(rep.get("winner") or "")
            if w in winner_counts:
                winner_counts[w] += 1
    overall_winner = max(
        state.profiles,
        key=lambda p: (
            (comp.get("arms") or {}).get(p, {}).get("composite_score", 0),
            winner_counts.get(p, 0),
        ),
    )
    return {
        "final": True,
        "run_id": state.run_id,
        "profiles": list(state.profiles),
        "started_at": state.started_at,
        "ends_at": state.ends_at,
        "duration_days": state.duration_days,
        "cycles_completed": state.cycle_index,
        "overall_winner": overall_winner,
        "daily_winner_counts": winner_counts,
        "comparison": comp,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def format_final_report(report: dict[str, Any]) -> str:
    lines = [
        "=" * 64,
        "  ProfitConfig A/B — final report",
        "=" * 64,
        f"  Run:              {report.get('run_id')}",
        f"  Profiles:         {', '.join(report.get('profiles') or [])}",
        f"  Cycles (arms):    {report.get('cycles_completed')}",
        f"  Overall winner:   {report.get('overall_winner')}",
        f"  Daily wins:       {report.get('daily_winner_counts')}",
        "",
        format_live_comparison(report.get("comparison") or {}),
        "=" * 64,
    ]
    return "\n".join(lines)


class AbTestCycleScheduler:
    """Outer loop: rotate ProfitConfig profile each cycle, log comparative P&L."""

    def __init__(
        self,
        agent: Any,
        wake_bus: Any,
        market_hours_provider: Any,
        *,
        state: AbTestState,
    ) -> None:
        from core.runtime.scheduler import CycleScheduler

        self._inner = CycleScheduler(agent, wake_bus, market_hours_provider)
        self.state = state
        self.agent = agent

    async def _maybe_daily_report(self, session_date: str) -> None:
        day = session_date[:10]
        if self.state.last_daily_report_date == day:
            return
        report = build_daily_winner_report(self.state, session_date=day)
        print("\n" + format_daily_winner_report(report))
        path = append_daily_report_json(report)
        logger.info("A/B daily winner report written: %s", path)
        self.state.last_daily_report_date = day
        save_state(self.state)

    async def run(self) -> None:
        from core.runtime.scheduler import CycleScheduler

        sched = self._inner
        try:
            while True:
                if duration_expired(self.state):
                    logger.info("A/B test duration reached; stopping.")
                    break

                if self.agent._halted:
                    if await sched._handle_halted():
                        break
                    continue

                if self.agent._start_of_day_cash is None:
                    self.agent._capture_start_of_day_cash()

                if await sched._maybe_after_hours_sleep():
                    continue

                sched._cycle += 1
                session_date = date.today().isoformat()
                arm = select_profile_for_cycle(self.state, session_date=session_date)
                profit = activate_ab_profile(arm, state=self.state)

                logger.info(
                    "A/B cycle %d arm=%s mode=%s until=%s",
                    sched._cycle,
                    arm,
                    self.state.mode,
                    self.state.ends_at,
                )
                logger.info(
                    "ProfitConfig arm: loss_cap=%s%% llm_cap=$%s profile=%s",
                    profit.risk.max_daily_loss_pct,
                    profit.risk.max_daily_llm_cost,
                    arm,
                )

                await sched._run_one_cycle()

                self.state.cycle_index += 1
                save_state(self.state)

                comp = build_live_comparison(self.state)
                print(format_live_comparison(comp))
                logger.info("A/B comparison leader=%s", comp.get("leader"))

                await self._maybe_daily_report(session_date)

        except KeyboardInterrupt:
            logger.info("A/B test interrupted")
        finally:
            _ab_active.set(False)
            _ab_log_meta.set(None)
            final = build_final_report(self.state)
            print("\n" + format_final_report(final))
            out = _LOG_DIR / f"ab_test_final_{self.state.run_id}.json"
            out.write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
            logger.info("A/B final report: %s", out)


async def run_ab_agent(
    *,
    profiles: tuple[str, str],
    duration_days: int,
    mode: str = "rotation",
    verbose: bool = False,
) -> None:
    """Paper trader loop with profile rotation and comparative cycle logging."""
    import asyncio

    from core.agent import TradingAgent
    from core.async_utils import safe_sleep as _safe_sleep
    from core.config import validate_config
    from core.log_setup import configure_root_logging
    from core.wake_events import wake_bus
    from data.broker_gateway import BrokerConfigError, create_gateway
    from data.cost_tracker import get_cost_tracker
    from data.data_provider import get_data_provider
    from data.market_hours import get_market_hours_provider
    from tools.tools_executor import ToolExecutor

    configure_root_logging("agent.log", verbose=verbose)

    state = init_ab_test_state(profiles, mode=mode, duration_days=duration_days)
    activate_ab_profile(select_profile_for_cycle(state), state=state)

    errors: list[str] = list(validate_config())
    if errors:
        for err in errors:
            logger.error("Startup: %s", err)
        raise SystemExit(1)

    gateway = None
    for attempt in range(5):
        try:
            gateway = await create_gateway({})
            break
        except (BrokerConfigError, ConnectionError) as e:
            if attempt < 4:
                await _safe_sleep(2 ** (attempt + 1))
            else:
                raise SystemExit(f"Broker connection failed: {e}") from e

    tools = ToolExecutor(
        gateway,
        get_data_provider(),
        market_hours_provider=get_market_hours_provider(),
        cost_tracker=get_cost_tracker(),
    )
    agent = TradingAgent(gateway, tools)
    tools._state_builder = agent._build_state_context
    tools._agent = agent
    agent._capture_start_of_day_cash()

    print(
        f"\n=== ProfitConfig A/B test (paper) ===\n"
        f"  Arms:     {profiles[0]} vs {profiles[1]}\n"
        f"  Mode:     {mode}\n"
        f"  Duration: {duration_days} days (ends {state.ends_at[:10]})\n"
        f"  Run id:   {state.run_id}\n"
        f"  Logs:     profit_cycles_*.json + logs/ab_test_daily_*.json\n"
        f"Press Ctrl+C to stop.\n"
    )

    scheduler = AbTestCycleScheduler(
        agent,
        wake_bus,
        get_market_hours_provider(),
        state=state,
    )
    await scheduler.run()

    if gateway:
        try:
            await gateway.disconnect()
        except Exception:
            pass


def run_ab_test_cli(args: Any) -> int:
    """Entry for ``--ab-test`` (blocking asyncio paper loop)."""
    import asyncio

    profiles = parse_ab_profiles(str(getattr(args, "ab_test", "")))
    duration = max(1, int(getattr(args, "ab_duration_days", DEFAULT_DURATION_DAYS) or DEFAULT_DURATION_DAYS))
    mode = str(getattr(args, "ab_mode", "rotation") or "rotation").strip().lower()
    if mode not in AB_MODES:
        raise SystemExit(f"--ab-mode must be one of: {', '.join(AB_MODES)}")

    validate_ab_test_startup(
        account=str(getattr(args, "account", "paper")),
        trading_mode=getattr(args, "trading_mode", None),
    )

    asyncio.run(
        run_ab_agent(
            profiles=profiles,
            duration_days=duration,
            mode=mode,
            verbose=bool(getattr(args, "verbose", False)),
        )
    )
    return 0


__all__ = [
    "AbTestCycleScheduler",
    "AbTestState",
    "build_daily_winner_report",
    "build_final_report",
    "build_live_comparison",
    "format_daily_winner_report",
    "format_final_report",
    "format_live_comparison",
    "get_ab_log_meta",
    "init_ab_test_state",
    "is_ab_test_active",
    "parse_ab_profiles",
    "run_ab_agent",
    "run_ab_test_cli",
    "validate_ab_test_startup",
]
