"""Shared health-check helpers for scripts/health.py, verify_trader_db.py, smoke_tools.py."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class Check:
    name: str
    status: Status
    message: str
    detail: str = ""


# Exit codes (shared convention)
EXIT_OK = 0
EXIT_WARN = 1
EXIT_FAIL = 2


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def enable_windows_vt() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = getattr(ctypes, "windll").kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


class _Colors:
    RESET = "\033[0m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"


def _c(code: str, text: str, *, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{_Colors.RESET}"


def status_label(status: Status, *, use_color: bool) -> str:
    if status == Status.OK:
        return _c(_Colors.GREEN, "OK  ", use_color=use_color)
    if status == Status.WARN:
        return _c(_Colors.YELLOW, "WARN", use_color=use_color)
    if status == Status.FAIL:
        return _c(_Colors.RED, "FAIL", use_color=use_color)
    return _c(_Colors.DIM, "SKIP", use_color=use_color)


class Reporter:
    """Collect checks and render a colored summary."""

    def __init__(self, title: str, *, use_color: bool | None = None) -> None:
        if use_color is None:
            enable_windows_vt()
            use_color = supports_color()
        self.title = title
        self.use_color = use_color
        self.checks: list[Check] = []

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def ok(self, name: str, message: str, *, detail: str = "") -> None:
        self.add(Check(name, Status.OK, message, detail))

    def warn(self, name: str, message: str, *, detail: str = "") -> None:
        self.add(Check(name, Status.WARN, message, detail))

    def fail(self, name: str, message: str, *, detail: str = "") -> None:
        self.add(Check(name, Status.FAIL, message, detail))

    def skip(self, name: str, message: str, *, detail: str = "") -> None:
        self.add(Check(name, Status.SKIP, message, detail))

    def exit_code(self) -> int:
        if any(c.status == Status.FAIL for c in self.checks):
            return EXIT_FAIL
        if any(c.status == Status.WARN for c in self.checks):
            return EXIT_WARN
        return EXIT_OK

    def print_report(self) -> None:
        header = _c(_Colors.BOLD, self.title, use_color=self.use_color)
        print(f"\n{header}\n")
        width = max((len(c.name) for c in self.checks), default=10)
        for c in self.checks:
            label = status_label(c.status, use_color=self.use_color)
            line = f"  {label}  {c.name.ljust(width)}  {c.message}"
            print(line)
            if c.detail:
                detail = _c(_Colors.DIM, f"         {c.detail}", use_color=self.use_color)
                print(detail)

        code = self.exit_code()
        if code == EXIT_OK:
            overall = _c(_Colors.GREEN, "OVERALL: HEALTHY", use_color=self.use_color)
        elif code == EXIT_WARN:
            overall = _c(_Colors.YELLOW, "OVERALL: DEGRADED (warnings)", use_color=self.use_color)
        else:
            overall = _c(_Colors.RED, "OVERALL: UNHEALTHY", use_color=self.use_color)
        print(f"\n{overall}  (exit {code})\n")


def _today_utc_start_ts() -> float:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()


def check_database_url_config(rep: Reporter) -> bool:
    """Return False if DATABASE_URL / PG* is missing."""
    try:
        from core.config import get_database_dsn

        dsn = get_database_dsn()
        if not dsn or not str(dsn).strip():
            rep.fail("Postgres config", "DATABASE_URL / PG* not set")
            return False
        # Redact password in display
        safe = str(dsn)
        if "@" in safe:
            try:
                from urllib.parse import urlparse

                p = urlparse(safe)
                host = p.hostname or "?"
                port = f":{p.port}" if p.port else ""
                db = p.path or ""
                rep.ok("Postgres config", f"{p.scheme}://{host}{port}{db}")
            except Exception:
                rep.ok("Postgres config", "configured")
        else:
            rep.ok("Postgres config", "configured")
        return True
    except Exception as e:
        rep.fail("Postgres config", "could not read DSN", detail=str(e))
        return False


def check_postgres_ping(rep: Reporter) -> bool:
    try:
        from memory import get_db

        conn = get_db()
        conn.execute("SELECT 1").fetchone()
        rep.ok("Postgres ping", "SELECT 1 succeeded")
        return True
    except Exception as e:
        rep.fail("Postgres ping", "connection failed", detail=str(e))
        return False


def check_postgres_init(rep: Reporter) -> bool:
    try:
        import memory

        memory.reset_state()
        memory.init_db()
        rep.ok("Schema init", "memory.init_db() OK")
        return True
    except Exception as e:
        rep.fail("Schema init", "memory.init_db() failed", detail=str(e))
        return False


_REQUIRED_TABLES = (
    "research_config",
    "signal_scores",
    "working_memory",
    "trades",
)


def check_required_tables(rep: Reporter) -> None:
    try:
        from memory import get_db

        conn = get_db()
        missing: list[str] = []
        for table in _REQUIRED_TABLES:
            try:
                conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
            except Exception:
                missing.append(table)
        if missing:
            rep.fail("Core tables", f"missing or unreadable: {', '.join(missing)}")
        else:
            rep.ok("Core tables", f"{len(_REQUIRED_TABLES)} tables reachable")
    except Exception as e:
        rep.fail("Core tables", "probe failed", detail=str(e))


def check_research_heartbeat(rep: Reporter, *, role: str = "trader") -> None:
    """role: 'trader' expects fresh heartbeat; 'researcher' same but warns if self stale."""
    try:
        from core.runtime.heartbeat import heartbeat_age_s, is_research_host_alive, read_heartbeat

        last = read_heartbeat()
        alive = is_research_host_alive()
        age = heartbeat_age_s()
        if last <= 0:
            rep.fail(
                "Research heartbeat",
                "no heartbeat in research_config",
                detail="Start: python -m research",
            )
            return
        age_s = round(age, 1) if age < float("inf") else None
        ts = datetime.fromtimestamp(last, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        detail = f"last={ts}, age={age_s}s" if age_s is not None else f"last={ts}"
        if alive:
            rep.ok("Research heartbeat", "fresh", detail=detail)
        elif age_s is not None and age_s <= 600:
            rep.warn(
                "Research heartbeat",
                "stale (scorer may be between rounds)",
                detail=detail,
            )
        else:
            msg = "stale or missing — trader may run independent mode"
            if role == "researcher":
                msg = "stale — research host may be down"
            rep.fail("Research heartbeat", msg, detail=detail)
    except Exception as e:
        rep.fail("Research heartbeat", "read failed", detail=str(e))


def check_token_cap(rep: Reporter) -> None:
    try:
        from core.config import RESEARCHER_DAILY_TOKEN_CAP
        from memory import get_research_config

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"researcher_daily_usage_{today}"
        usage = float(get_research_config(key, 0.0))
        cap = int(RESEARCHER_DAILY_TOKEN_CAP)
        if cap <= 0:
            rep.skip("Researcher token cap", "cap disabled (0)")
            return
        pct = (usage / cap) * 100.0
        detail = f"{usage:,.0f} / {cap:,} ({pct:.1f}%) UTC {today}"
        if usage >= cap:
            rep.fail("Researcher token cap", "CAP EXCEEDED", detail=detail)
        elif pct >= 85.0:
            rep.warn("Researcher token cap", "approaching cap", detail=detail)
        else:
            rep.ok("Researcher token cap", "within budget", detail=detail)
    except Exception as e:
        rep.warn("Researcher token cap", "could not read usage", detail=str(e))


def check_mda(rep: Reporter, *, symbol: str = "SPY") -> None:
    try:
        from core.config import RESEARCHER_MDA_HEALTH_CHECK_ENABLED

        if not RESEARCHER_MDA_HEALTH_CHECK_ENABLED:
            rep.skip("MDA quotes", "RESEARCHER_MDA_HEALTH_CHECK_ENABLED=0")
            return
    except Exception:
        pass

    try:
        from data.data_provider import get_data_provider

        dp = get_data_provider()
        quote = dp.get_quote(symbol)
        last = quote.get("last") if isinstance(quote, dict) else None
        if last is None:
            rep.fail("MDA quotes", f"{symbol} probe returned no last price")
            return
        rep.ok("MDA quotes", f"{symbol} last={last}", detail=str(quote.get("source", ""))[:80])
    except Exception as e:
        rep.fail("MDA quotes", f"{symbol} probe failed", detail=str(e))


async def check_ibkr_async(rep: Reporter, *, client_id: int | None) -> None:
    if client_id is None:
        rep.skip("IBKR gateway", "pass --client-id or --ibkr-client-id to test")
        return
    os.environ["IBKR_CLIENT_ID"] = str(client_id)
    gateway = None
    try:
        from data.broker_gateway import create_gateway

        gateway = await create_gateway({})
        acct = getattr(gateway, "account_id", None) or "connected"
        rep.ok("IBKR gateway", f"connected (client_id={client_id})", detail=str(acct))
    except Exception as e:
        rep.fail("IBKR gateway", "connect failed", detail=str(e))
    finally:
        if gateway is not None:
            try:
                await gateway.disconnect()
            except Exception:
                pass


def check_scoring_activity(rep: Reporter) -> None:
    try:
        from memory import get_db

        conn = get_db()
        start_ts = _today_utc_start_ts()
        row = conn.execute(
            "SELECT MAX(ts) AS last_ts, COUNT(*) AS count FROM signal_scores WHERE ts >= ?",
            (start_ts,),
        ).fetchone()
        if row and row["last_ts"]:
            last = datetime.fromtimestamp(float(row["last_ts"]), tz=timezone.utc)
            count = int(row["count"] or 0)
            rep.ok(
                "Scoring today",
                f"{count} score row(s)",
                detail=f"last={last.strftime('%H:%M:%S UTC')}",
            )
        else:
            rep.warn("Scoring today", "no signal_scores rows yet today (UTC)")
    except Exception as e:
        rep.warn("Scoring today", "query failed", detail=str(e))


def check_template_evolution(rep: Reporter) -> None:
    try:
        from memory import get_research_config

        last_evo = float(get_research_config("last_template_evolution_round", 0.0))
        if last_evo > 0:
            ts = datetime.fromtimestamp(last_evo, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            rep.ok("Template evolution", f"last round {ts}")
        else:
            rep.warn("Template evolution", "no last_template_evolution_round in config")
    except Exception as e:
        rep.warn("Template evolution", "could not read", detail=str(e))


def check_trader_operating_context(rep: Reporter) -> None:
    try:
        from core.runtime.heartbeat import heartbeat_age_s, is_research_host_alive
        from core.runtime.local_memory_fallback import LOCAL_MEMORY_FILE
        from core.runtime.operating_context import get_operating_context

        ctx = get_operating_context()
        try:
            researcher_alive = is_research_host_alive()
            hb_age = heartbeat_age_s()
        except Exception:
            researcher_alive = False
            hb_age = float("inf")

        mode = "INDEPENDENT" if (not researcher_alive or ctx.is_independent_mode) else "FULL"
        detail = (
            f"mode={mode} quality={ctx.quality.overall_quality} "
            f"risk_mult={ctx.risk_multiplier:.2f} wm={ctx.quality.working_memory_completeness * 100:.0f}%"
        )
        if mode == "INDEPENDENT":
            rep.warn("Trader operating mode", "independent / conservative", detail=detail)
        else:
            rep.ok("Trader operating mode", "full researcher access", detail=detail)

        if LOCAL_MEMORY_FILE.exists():
            try:
                import json

                data = json.loads(LOCAL_MEMORY_FILE.read_text(encoding="utf-8"))
                total = sum(len(v) for v in data.values() if isinstance(v, list))
                rep.ok("Local WM fallback", f"{total} cached entries", detail=str(LOCAL_MEMORY_FILE))
            except Exception as e:
                rep.warn("Local WM fallback", "file present but unreadable", detail=str(e))
        else:
            rep.ok("Local WM fallback", "not in use (no local file)")

        if hb_age < float("inf"):
            rep.ok("Heartbeat age (trader view)", f"{round(hb_age, 1)}s ago")
    except Exception as e:
        rep.fail("Trader operating mode", "context read failed", detail=str(e))


def check_profit_api(rep: Reporter, *, base_url: str | None = None) -> None:
    """GET /profit_summary — 24h stats and active ProfitConfig (optional sidecar)."""
    import httpx

    root = (base_url or os.getenv("PROFIT_API_URL", "http://127.0.0.1:8787")).rstrip("/")
    url = f"{root}/profit_summary"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            rep.fail("Profit API", f"HTTP {resp.status_code}", detail=url)
            return
        data = resp.json()
        stats = data.get("stats_24h") or {}
        active = data.get("active_config") or {}
        profile = active.get("profit_profile", "?")
        mode = active.get("trading_mode", "?")
        cycles = int(stats.get("cycles") or 0)
        pnl = float(stats.get("total_cycle_pnl_usd") or 0)
        llm = float(stats.get("llm_cost_usd") or 0)
        detail = (
            f"profile={profile} mode={mode} cycles_24h={cycles} "
            f"pnl_24h=${pnl:+.2f} llm=${llm:.4f}"
        )
        if cycles == 0:
            rep.warn("Profit API", "reachable but no cycles in window", detail=detail)
        else:
            rep.ok("Profit API", url, detail=detail)
    except httpx.ConnectError:
        rep.skip(
            "Profit API",
            "not running",
            detail=f"start: python -m api  then {url}",
        )
    except httpx.TimeoutException:
        rep.fail("Profit API", "timeout", detail=url)
    except Exception as e:
        rep.fail("Profit API", "request failed", detail=str(e))


def apply_health_report_to_reporter(rep: Reporter, report: dict[str, Any]) -> None:
    """Map :func:`core.observability.health_report.build_health_report` into checks."""
    status = report.get("overall_status", "healthy")
    if status == "healthy":
        rep.ok("Observability", f"overall={status}")
    elif status == "degraded":
        rep.warn("Observability", f"overall={status}")
    else:
        rep.fail("Observability", f"overall={status}")

    profile = report.get("active_profile", "?")
    levers = (report.get("profit_config") or {}).get("key_levers") or {}
    rep.ok(
        "ProfitConfig",
        f"profile={profile} mode={levers.get('trading_mode')}",
        detail=(
            f"loss_cap={levers.get('max_daily_loss_pct')}% "
            f"dd_cap={levers.get('intraday_drawdown_pct')}% "
            f"llm_cap=${levers.get('max_daily_llm_cost')}"
        ),
    )

    hb = report.get("research_heartbeat") or {}
    if hb.get("operational"):
        rep.ok("Research heartbeat (report)", "operational", detail=f"age_s={hb.get('age_s')}")
    elif hb.get("last_ts"):
        rep.warn("Research heartbeat (report)", "not operational", detail=str(hb))
    else:
        rep.fail("Research heartbeat (report)", "missing", detail=str(hb))

    daily = report.get("daily_summary") or {}
    rep.ok(
        "Daily cycle summary",
        f"{daily.get('cycles', 0)} cycles today",
        detail=f"pnl=${daily.get('total_cycle_pnl_usd', 0):+.2f} llm=${daily.get('llm_cost_usd', 0):.4f}",
    )

    sim = report.get("simulation") or {}
    if sim.get("latest_export_csv"):
        rep.ok(
            "Simulation export",
            sim["latest_export_csv"].get("path", "?"),
            detail=sim["latest_export_csv"].get("mtime_utc", ""),
        )

    skip_info = {
        "profit_cycles_missing_today",
        "profit_cycles_missing_window",
        "simulation_mode_active",
    }
    for alert in report.get("alerts") or []:
        code = alert.get("code", "alert")
        sev = alert.get("severity", "info")
        msg = alert.get("message", "")
        detail = alert.get("detail", "")
        if sev == "critical":
            rep.fail(f"Alert:{code}", msg, detail=detail)
        elif sev == "warn":
            rep.warn(f"Alert:{code}", msg, detail=detail)
        elif code not in skip_info:
            rep.ok(f"Alert:{code}", msg, detail=detail)


def check_observability_report(rep: Reporter, *, role: str = "trader") -> dict[str, Any]:
    """Build central health report and add Reporter checks."""
    from core.observability.health_report import build_health_report

    report = build_health_report(role=role)
    apply_health_report_to_reporter(rep, report)
    return report


def run_platform_checks(
    rep: Reporter,
    *,
    role: str,
    include_mda: bool = True,
    include_scoring: bool = False,
    include_evolution: bool = False,
    include_trader_context: bool = False,
    include_profit_api: bool = True,
    profit_api_url: str | None = None,
    include_observability: bool = True,
) -> bool:
    """Postgres + heartbeat + token. Returns True if Postgres config present."""
    if not check_database_url_config(rep):
        return False
    if not check_postgres_ping(rep):
        return False
    check_research_heartbeat(rep, role=role)
    check_token_cap(rep)
    if include_mda:
        check_mda(rep)
    if include_scoring:
        check_scoring_activity(rep)
    if include_evolution:
        check_template_evolution(rep)
    if include_trader_context:
        check_trader_operating_context(rep)
    if include_profit_api:
        check_profit_api(rep, base_url=profit_api_url)
    if include_observability:
        try:
            check_observability_report(rep, role=role)
        except Exception as e:
            rep.warn("Observability report", "build failed", detail=str(e))
    return True


def load_dotenv_repo(root: Any) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=True)
    except Exception:
        pass
