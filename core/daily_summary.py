"""Daily profitability summary: dashboard, optimizer, tomorrow's ProfitConfig recommendation."""

from __future__ import annotations

import json
import os
import smtplib
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from core.live_profile_optimize import format_live_optimize_report, run_live_optimize
from core.profit_profiles import PROFIT_PROFILE_ENV, VALID_PROFILES, profile_note

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_JSON_DIR = _REPO_ROOT / "data"


def _tomorrow_session() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def _sim_suggested_profile(sim_payload: dict[str, Any] | None) -> str | None:
    if not sim_payload:
        return None
    best = sim_payload.get("best") or {}
    base = str(best.get("base_profile") or "").strip().lower()
    if base in VALID_PROFILES:
        return base
    cid = str(best.get("candidate_id") or "").strip().lower()
    if cid in VALID_PROFILES:
        return cid
  # Perturbed candidates: use base_profile when set
    if base:
        return base
    for part in cid.replace("-", "_").split("_"):
        if part in VALID_PROFILES:
            return part
    return "balanced"


def merge_tomorrow_recommendation(
    live: dict[str, Any],
    sim: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine live cycle-log ranking with simulation optimizer best candidate."""
    live_prof = str(live.get("suggested_profile") or "balanced")
    sim_prof = _sim_suggested_profile(sim)
    tomorrow = _tomorrow_session()

    if sim_prof is None:
        return {
            "session_date": tomorrow,
            "recommended_profile": live_prof,
            "confidence": live.get("confidence", "none"),
            "source": "live_cycle_logs",
            "rationale": live.get("rationale", ""),
            "live_suggested": live_prof,
            "sim_suggested": None,
            "agreement": None,
        }

    agreement = live_prof == sim_prof
    live_conf = str(live.get("confidence") or "none")

    if agreement:
        confidence = "high"
        rationale = (
            f"Live cycle logs and {sim.get('mode', 'simulation')} optimizer both rank "
            f"{live_prof!r} highest for the lookback window."
        )
        recommended = live_prof
    elif live_conf == "none":
        confidence = "medium"
        rationale = (
            f"Insufficient live cycle data; simulation best base profile is {sim_prof!r} "
            f"(composite={((sim.get('best') or {}).get('metrics') or {}).get('composite_score', '?')})."
        )
        recommended = sim_prof
    elif live_conf == "high":
        confidence = "medium"
        rationale = (
            f"Live logs strongly favor {live_prof!r}; simulation favors {sim_prof!r}. "
            f"Defaulting to live for tomorrow; re-run with longer sim window if you prefer backtest-led."
        )
        recommended = live_prof
    else:
        confidence = "low"
        rationale = (
            f"Mixed signals: live suggests {live_prof!r} ({live_conf}), "
            f"simulation suggests {sim_prof!r}. Review rankings before trading."
        )
        recommended = live_prof

    return {
        "session_date": tomorrow,
        "recommended_profile": recommended,
        "confidence": confidence,
        "source": "live+simulation" if sim else "live_cycle_logs",
        "rationale": rationale,
        "live_suggested": live_prof,
        "sim_suggested": sim_prof,
        "agreement": agreement,
    }


def build_dashboard_section(
    *,
    session_date: str | None = None,
    days: int = 1,
) -> dict[str, Any]:
    """Dashboard aggregates from profit cycle logs (same data as scripts/dashboard.py)."""
    from scripts.dashboard import (
        _aggregate,
        _load_entries_window,
        _window_label,
        format_dashboard,
    )

    entries = _load_entries_window(session_date=session_date, days=days)
    agg = _aggregate(entries)
    label = _window_label(session_date, days)
    return {
        "window_label": label,
        "days": days,
        "entries": len(entries),
        "aggregate": agg,
        "console_text": format_dashboard(agg, window_label=label),
    }


def run_sim_optimizer(
    *,
    days: int = 7,
    quick: bool = True,
    baseline: str = "balanced",
    cycles_per_day: int = 1,
) -> dict[str, Any]:
    """Historical grid optimization (scripts/optimize_profiles grid path)."""
    from scripts.optimize_profiles import _parse_args, run_grid_optimization

    argv = [
        "--days",
        str(max(1, days)),
        "--baseline",
        baseline,
        "--cycles-per-day",
        str(max(1, min(4, cycles_per_day))),
    ]
    if quick:
        argv.append("--quick")
    args = _parse_args(argv)
    return run_grid_optimization(args)


def run_daily_summary(
    *,
    dashboard_days: int = 1,
    dashboard_date: str | None = None,
    live_lookback_days: int = 7,
    sim_days: int = 7,
    run_sim: bool = True,
    sim_quick: bool = True,
    sim_baseline: str = "balanced",
) -> dict[str, Any]:
    """Run dashboard + live optimize + optional sim optimizer; return full report dict."""
    now = datetime.now(timezone.utc)
    dashboard = build_dashboard_section(session_date=dashboard_date, days=dashboard_days)
    live = run_live_optimize(days=live_lookback_days)

    sim: dict[str, Any] | None = None
    sim_error: str | None = None
    if run_sim:
        try:
            sim = run_sim_optimizer(days=sim_days, quick=sim_quick, baseline=sim_baseline)
        except Exception as e:
            sim_error = str(e)

    recommendation = merge_tomorrow_recommendation(live, sim)
    prof = recommendation["recommended_profile"]
    try:
        from core.central_profit_config import get_profit_config

        active = get_profit_config().reload(dotenv=False)
        from core.profit_config_state import get_active_profile_label
        from core.profit_cycle_logger import snapshot_profit_config

        config_snap = snapshot_profit_config(active, profile_label=get_active_profile_label())
    except Exception as e:
        config_snap = {"error": str(e)}

    try:
        _DEFAULT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        (_DEFAULT_JSON_DIR / "live_profile_suggestion.json").write_text(
            json.dumps(
                {
                    "generated_at": now.isoformat(),
                    "source": "daily_summary",
                    "suggested_profile": prof,
                    "confidence": recommendation.get("confidence"),
                    "days_analyzed": live_lookback_days,
                    "rationale": recommendation.get("rationale"),
                    "live_suggested": recommendation.get("live_suggested"),
                    "sim_suggested": recommendation.get("sim_suggested"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass

    return {
        "generated_at": now.isoformat(),
        "report_type": "daily_summary",
        "dashboard": dashboard,
        "live_optimization": live,
        "simulation_optimization": sim,
        "simulation_error": sim_error,
        "tomorrow": recommendation,
        "active_profit_config": {
            "profile_env": os.getenv(PROFIT_PROFILE_ENV, "balanced"),
            "snapshot": config_snap,
        },
        "action": {
            "set_env": f"{PROFIT_PROFILE_ENV}={prof}",
            "cli_trader": f"python __main__.py --profit-profile {prof}",
            "cli_research": f"python -m research --profile {prof}",
            "profile_note": profile_note(prof) if prof in VALID_PROFILES else "",
        },
    }


def format_daily_summary_report(report: dict[str, Any]) -> str:
    """Multi-section console report."""
    lines: list[str] = [
        "",
        "=" * 70,
        "  ABC DAILY SUMMARY — ProfitConfig recommendation for tomorrow",
        "=" * 70,
        f"  Generated:  {report.get('generated_at', '')}",
    ]
    tomorrow = (report.get("tomorrow") or {})
    lines.extend(
        [
            f"  Tomorrow:   {tomorrow.get('session_date', _tomorrow_session())}",
            f"  Recommend:  {tomorrow.get('recommended_profile', '?')}  "
            f"[{tomorrow.get('confidence', '?')} confidence]",
            f"  Source:     {tomorrow.get('source', '?')}",
            f"  Rationale:  {tomorrow.get('rationale', '')}",
            "",
            "  Apply",
            f"    {report.get('action', {}).get('set_env', '')}",
            f"    {report.get('action', {}).get('cli_trader', '')}",
            "",
        ]
    )
    if tomorrow.get("live_suggested") != tomorrow.get("sim_suggested") and tomorrow.get("sim_suggested"):
        lines.append(
            f"  Note: live={tomorrow.get('live_suggested')}  sim={tomorrow.get('sim_suggested')}  "
            f"agree={tomorrow.get('agreement')}"
        )
        lines.append("")

    dash_text = (report.get("dashboard") or {}).get("console_text", "")
    if dash_text:
        lines.append(dash_text)
        lines.append("")

    live = report.get("live_optimization")
    if live:
        lines.append(format_live_optimize_report(live))
        lines.append("")

    sim = report.get("simulation_optimization")
    if sim:
        best = sim.get("best") or {}
        metrics = best.get("metrics") or {}
        lines.extend(
            [
                "-" * 70,
                "  Simulation optimizer (historical backtest)",
                "-" * 70,
                f"  Window:     {sim.get('start_date')} -> {sim.get('end_date')}",
                f"  Best:       {best.get('candidate_id')} (base={best.get('base_profile')})",
                f"  Composite:  {metrics.get('composite_score', '?')}",
                f"  Profit:     ${metrics.get('total_profit_usd', 0):+,.2f}",
                f"  Sharpe:     {metrics.get('sharpe_ratio', '?')}",
                "",
            ]
        )
    elif report.get("simulation_error"):
        lines.extend(
            [
                "-" * 70,
                f"  Simulation optimizer skipped: {report['simulation_error']}",
                "",
            ]
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def save_daily_summary_json(report: dict[str, Any], path: Path | None = None) -> Path:
    """Write report JSON under ``data/daily_summary_YYYY-MM-DD.json``."""
    d = date.today().isoformat()
    target = path or (_DEFAULT_JSON_DIR / f"daily_summary_{d}.json")
    if not target.is_absolute():
        target = _REPO_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return target


def format_slack_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Slack-compatible webhook body."""
    tomorrow = report.get("tomorrow") or {}
    dash = (report.get("dashboard") or {}).get("aggregate") or {}
    text = (
        f"*ABC daily summary* — tomorrow `{tomorrow.get('recommended_profile', '?')}` "
        f"({tomorrow.get('confidence', '?')} confidence)\n"
        f"{tomorrow.get('rationale', '')}\n"
        f"Today: {dash.get('cycles', 0)} cycles, "
        f"pnl=${dash.get('total_cycle_pnl', 0):+.2f}, "
        f"llm=${dash.get('llm_cost', 0):.4f}\n"
        f"`{report.get('action', {}).get('set_env', '')}`"
    )
    return {"text": text}


def send_slack(webhook_url: str, report: dict[str, Any]) -> None:
    import httpx

    payload = format_slack_payload(report)
    with httpx.Client(timeout=15.0) as client:
        client.post(webhook_url, json=payload)


def send_email(
    report: dict[str, Any],
    *,
    to_addrs: list[str],
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_addr: str,
    use_tls: bool = True,
) -> None:
    """Send plain-text daily summary via SMTP."""
    subject = (
        f"ABC daily summary — tomorrow: "
        f"{(report.get('tomorrow') or {}).get('recommended_profile', '?')}"
    )
    body = format_daily_summary_report(report)
    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        if smtp_user:
            smtp.login(smtp_user, smtp_password)
        smtp.sendmail(from_addr, to_addrs, msg.as_string())


def deliver_notifications(report: dict[str, Any]) -> list[str]:
    """Send optional Slack and email; return list of delivery notes."""
    notes: list[str] = []
    slack = (
        os.getenv("DAILY_SUMMARY_SLACK_WEBHOOK", "").strip()
        or os.getenv("ALERT_WEBHOOK_URL", "").strip()
    )
    if slack:
        try:
            send_slack(slack, report)
            notes.append("slack:ok")
        except Exception as e:
            notes.append(f"slack:failed:{e}")

    to_raw = os.getenv("DAILY_SUMMARY_EMAIL_TO", "").strip()
    if to_raw:
        host = os.getenv("DAILY_SUMMARY_SMTP_HOST", "").strip()
        if not host:
            notes.append("email:skipped:no SMTP host")
        else:
            try:
                send_email(
                    report,
                    to_addrs=[a.strip() for a in to_raw.split(",") if a.strip()],
                    smtp_host=host,
                    smtp_port=int(os.getenv("DAILY_SUMMARY_SMTP_PORT", "587")),
                    smtp_user=os.getenv("DAILY_SUMMARY_SMTP_USER", ""),
                    smtp_password=os.getenv("DAILY_SUMMARY_SMTP_PASSWORD", ""),
                    from_addr=os.getenv(
                        "DAILY_SUMMARY_EMAIL_FROM",
                        os.getenv("DAILY_SUMMARY_SMTP_USER", "abc-daily@localhost"),
                    ),
                    use_tls=os.getenv("DAILY_SUMMARY_SMTP_TLS", "1").strip().lower()
                    not in ("0", "false", "no"),
                )
                notes.append("email:ok")
            except Exception as e:
                notes.append(f"email:failed:{e}")
    return notes


__all__ = [
    "build_dashboard_section",
    "deliver_notifications",
    "format_daily_summary_report",
    "merge_tomorrow_recommendation",
    "run_daily_summary",
    "run_sim_optimizer",
    "save_daily_summary_json",
    "send_email",
    "send_slack",
]
