"""FastAPI status API — full JSON health report from central ProfitConfig."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from infra.status_api.dashboard import router as dashboard_router

app = FastAPI(
    title="ABC Status API",
    description="Production health and observability (ProfitConfig, safety, alerts).",
    version="1.0.0",
)

app.include_router(dashboard_router)


def _default_role() -> str:
    return os.getenv("ABC_HEALTH_ROLE", "trader").strip().lower() or "trader"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready() -> JSONResponse:
    """Return 503 when overall_status is unhealthy (for Docker / load balancers)."""
    from core.observability.health_report import build_health_report

    role = _default_role()
    report = build_health_report(role=role)
    status = report.get("overall_status", "healthy")
    code = 200 if status != "unhealthy" else 503
    return JSONResponse(
        content={
            "status": "ready" if code == 200 else "unhealthy",
            "overall_status": status,
            "active_profile": report.get("active_profile"),
            "alert_counts": report.get("alert_counts"),
        },
        status_code=code,
    )


@app.get("/status")
def status(
    role: str | None = Query(None, description="trader | researcher | all"),
    window_hours: int | None = Query(None, ge=1, le=168),
) -> JSONResponse:
    """Full JSON health report including active ProfitConfig profile and alerts."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    from core.observability.health_report import build_health_report

    resolved_role = (role or _default_role()).strip().lower()
    if resolved_role not in ("trader", "researcher", "all"):
        resolved_role = _default_role()

    body: dict[str, Any] = build_health_report(
        role=resolved_role,
        window_hours=window_hours,
    )
    return JSONResponse(content=body)


@app.get("/status/text")
def status_text(
    role: str | None = Query(None),
) -> PlainTextResponse:
    """Human-readable one-page summary for operators."""
    from core.observability.health_report import build_health_report

    report = build_health_report(role=(role or _default_role()))
    lines = [
        f"ABC Status — {report.get('overall_status', '?').upper()}",
        f"Profile: {report.get('active_profile')}",
        f"Generated: {report.get('generated_at')}",
        "",
        "Alerts:",
    ]
    alerts = report.get("alerts") or []
    if not alerts:
        lines.append("  (none)")
    else:
        for a in alerts:
            lines.append(f"  [{a.get('severity')}] {a.get('code')}: {a.get('message')}")
    daily = report.get("daily_summary") or {}
    lines.extend(
        [
            "",
            f"Today: {daily.get('cycles', 0)} cycles, "
            f"pnl=${daily.get('total_cycle_pnl_usd', 0):+.2f}, "
            f"llm=${daily.get('llm_cost_usd', 0):.4f}",
        ]
    )
    return PlainTextResponse("\n".join(lines))
