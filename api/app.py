"""FastAPI app — profitability and ops endpoints."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="ABC Profit API",
    description="Lightweight ops API for profitability cycle logs and config.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status(role: str = "trader") -> JSONResponse:
    """Full health report (same payload as ``infra.status_api`` ``GET /status``)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass
    from core.observability.health_report import build_health_report

    body = build_health_report(role=role.strip().lower() or "trader")
    return JSONResponse(content=body)


@app.get("/profit_summary")
def profit_summary() -> JSONResponse:
    """Last 24h profitability stats and active :class:`ProfitConfig` snapshot."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    from core.profit_summary import build_profit_summary

    hours = int(os.getenv("PROFIT_SUMMARY_WINDOW_HOURS", "24"))
    body: dict[str, Any] = build_profit_summary(window_hours=hours)
    return JSONResponse(content=body)
