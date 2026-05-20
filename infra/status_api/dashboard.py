"""Password-protected HTML dashboard for ABC operations."""

from __future__ import annotations

import json
import os
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

router = APIRouter(tags=["dashboard"])
_security = HTTPBasic(auto_error=False)


def _dashboard_password() -> str | None:
    raw = os.getenv("DASHBOARD_PASSWORD", "").strip()
    return raw or None


def require_dashboard_auth(
    credentials: HTTPBasicCredentials | None = Depends(_security),
) -> None:
    """HTTP Basic auth when ``DASHBOARD_PASSWORD`` is set."""
    expected = _dashboard_password()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard disabled: set DASHBOARD_PASSWORD in environment",
        )
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic realm=ABC Dashboard"},
        )
    user_ok = secrets.compare_digest(credentials.username.encode(), b"admin") or secrets.compare_digest(
        credentials.username.encode(), b"dashboard"
    )
    pass_ok = secrets.compare_digest(credentials.password.encode(), expected.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic realm=ABC Dashboard"},
        )


def _status_class(overall: str) -> str:
    return {
        "healthy": "ok",
        "degraded": "warn",
        "unhealthy": "bad",
    }.get(str(overall).lower(), "warn")


def _render_dashboard_html(data: dict[str, Any], *, refresh_sec: int = 60) -> str:
    payload = json.dumps(data, default=str)
    meta_refresh = f'<meta http-equiv="refresh" content="{refresh_sec}">' if refresh_sec > 0 else ""
    overall = data.get("overall_status", "unknown")
    status_cls = _status_class(overall)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_refresh}
  <title>ABC Profit Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f1419; --card: #1a2332; --text: #e7ecf3; --muted: #8b9cb3;
      --ok: #3dd68c; --warn: #f5c542; --bad: #f87171; --accent: #60a5fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
      margin: 0; padding: 1rem 1.25rem; line-height: 1.45; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.25rem; }}
    .sub {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }}
    .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .card {{ background: var(--card); border-radius: 10px; padding: 1rem; }}
    .card h2 {{ font-size: 0.95rem; margin: 0 0 0.75rem; color: var(--accent); }}
    .badge {{ display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
      font-size: 0.8rem; font-weight: 600; }}
    .badge.ok {{ background: rgba(61,214,140,0.2); color: var(--ok); }}
    .badge.warn {{ background: rgba(245,197,66,0.2); color: var(--warn); }}
    .badge.bad {{ background: rgba(248,113,113,0.2); color: var(--bad); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    td, th {{ text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #2a3548; }}
    .metric {{ font-size: 1.4rem; font-weight: 600; }}
    .charts {{ display: grid; gap: 1rem; grid-template-columns: 1fr; margin-top: 1rem; }}
    @media (min-width: 900px) {{ .charts {{ grid-template-columns: 1fr 1fr; }} }}
    canvas {{ max-height: 260px; }}
    ul.alerts {{ margin: 0; padding-left: 1.1rem; font-size: 0.85rem; }}
    ul.alerts li {{ margin: 0.25rem 0; }}
  </style>
</head>
<body>
  <h1>ABC Profit Dashboard</h1>
  <p class="sub">Profile <strong id="profile">—</strong> ·
    <span class="badge {status_cls}" id="status">{overall}</span> ·
    <span id="generated">—</span></p>

  <div class="grid">
    <div class="card">
      <h2>ProfitConfig</h2>
      <div id="levers"></div>
    </div>
    <div class="card">
      <h2>Safety &amp; P&amp;L (today)</h2>
      <div id="safety"></div>
    </div>
    <div class="card">
      <h2>Research heartbeat</h2>
      <div id="heartbeat"></div>
    </div>
    <div class="card">
      <h2>QualityMatrix</h2>
      <div id="qm"></div>
    </div>
  </div>

  <div class="charts">
    <div class="card">
      <h2>Live P&amp;L (cycle logs)</h2>
      <canvas id="pnlChart"></canvas>
    </div>
    <div class="card">
      <h2>Simulation vs live (by profile)</h2>
      <canvas id="compareChart"></canvas>
    </div>
  </div>

  <div class="card" style="margin-top:1rem">
    <h2>Alerts</h2>
    <ul class="alerts" id="alerts"></ul>
  </div>

  <script>
    const DATA = {payload};
    document.getElementById('profile').textContent = DATA.active_profile || '—';
    document.getElementById('generated').textContent = DATA.generated_at || '—';

    const levers = DATA.key_levers || {{}};
    document.getElementById('levers').innerHTML = `
      <table>
        <tr><th>Trading mode</th><td>${{levers.trading_mode ?? '—'}}</td></tr>
        <tr><th>Risk / trade</th><td>${{levers.risk_per_trade_pct ?? '—'}}%</td></tr>
        <tr><th>Max daily loss</th><td>${{levers.max_daily_loss_pct ?? '—'}}%</td></tr>
        <tr><th>Drawdown cap</th><td>${{levers.intraday_drawdown_pct ?? '—'}}%</td></tr>
        <tr><th>LLM cap</th><td>$${{levers.max_daily_llm_cost ?? '—'}}</td></tr>
      </table>`;

    const s = DATA.safety || {{}};
    const th = s.thresholds || {{}};
    document.getElementById('safety').innerHTML = `
      <div class="metric">$${{(s.today_realized_pnl_usd ?? 0).toFixed(2)}}</div>
      <p class="sub">Realized P&amp;L today · LLM $${{(s.today_llm_cost_usd ?? 0).toFixed(4)}} /
        $${{th.max_daily_llm_cost_usd ?? '—'}}</p>
      <table>
        <tr><th>Drawdown</th><td>${{s.intraday_drawdown_pct ?? 0}}% / ${{th.intraday_drawdown_pct ?? '—'}}%</td></tr>
        <tr><th>Daily loss</th><td>${{s.daily_loss_pct ?? '—'}}%</td></tr>
      </table>`;

    const hb = DATA.research_heartbeat || {{}};
    document.getElementById('heartbeat').innerHTML = `
      <table>
        <tr><th>Operational</th><td>${{hb.operational ? 'yes' : 'no'}}</td></tr>
        <tr><th>Status</th><td>${{hb.status_label ?? hb.status_code ?? '—'}}</td></tr>
        <tr><th>Age</th><td>${{hb.age_s != null ? hb.age_s + 's' : '—'}}</td></tr>
        <tr><th>Round</th><td>${{hb.round ?? 0}}</td></tr>
        <tr><th>Host profile</th><td>${{hb.host_profile || '—'}}</td></tr>
      </table>`;

    const qm = DATA.quality_matrix || {{}};
    document.getElementById('qm').innerHTML = `
      <table>
        <tr><th>Overall</th><td>${{qm.overall_quality ?? '—'}}</td></tr>
        <tr><th>Risk mult</th><td>${{qm.risk_multiplier ?? '—'}}</td></tr>
        <tr><th>Exec quality</th><td>${{qm.execution_quality ?? qm.global_execution_quality ?? '—'}}</td></tr>
        <tr><th>Symbols</th><td>${{qm.symbol_count ?? 0}}</td></tr>
        <tr><th>Blocked tools</th><td>${{(qm.blocked_tool_categories || []).join(', ') || 'none'}}</td></tr>
      </table>`;

    const alerts = DATA.alerts || [];
    document.getElementById('alerts').innerHTML = alerts.length
      ? alerts.map(a => `<li>[${{a.severity}}] ${{a.code}}: ${{a.message}}</li>`).join('')
      : '<li>No alerts</li>';

    const series = DATA.pnl_series || {{}};
    new Chart(document.getElementById('pnlChart'), {{
      type: 'line',
      data: {{
        labels: series.labels || [],
        datasets: [
          {{ label: 'Cumulative $', data: series.cumulative_pnl_usd || [], borderColor: '#60a5fa', tension: 0.2 }},
          {{ label: 'Cycle $', data: series.cycle_pnl_usd || [], borderColor: '#3dd68c', tension: 0.2 }}
        ]
      }},
      options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});

    const cmp = DATA.sim_vs_live || {{}};
    const live = cmp.live || {{}};
    const sim = cmp.simulation || {{}};
    const allLabels = [...new Set([...(live.labels||[]), ...(sim.labels||[])])];
    const liveMap = Object.fromEntries((live.labels||[]).map((l,i)=>[l, live.total_cycle_pnl_usd[i]]));
    const simMap = Object.fromEntries((sim.labels||[]).map((l,i)=>[l, sim.composite_score[i]]));
    new Chart(document.getElementById('compareChart'), {{
      type: 'bar',
      data: {{
        labels: allLabels,
        datasets: [
          {{ label: 'Live cycle PnL $', data: allLabels.map(l => liveMap[l] ?? null), backgroundColor: 'rgba(96,165,250,0.7)' }},
          {{ label: 'Sim composite', data: allLabels.map(l => simMap[l] ?? null), backgroundColor: 'rgba(245,197,66,0.7)' }}
        ]
      }},
      options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});
  </script>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(
    _: None = Depends(require_dashboard_auth),
    refresh: int = 60,
) -> HTMLResponse:
    """HTML dashboard (Chart.js charts, auto-refresh)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    from core.observability.dashboard_data import build_dashboard_payload

    data = build_dashboard_payload()
    html = _render_dashboard_html(data, refresh_sec=max(0, min(refresh, 3600)))
    return HTMLResponse(html)


@router.get("/dashboard/data")
def dashboard_data(
    _: None = Depends(require_dashboard_auth),
    role: str = "trader",
    window_hours: int | None = None,
) -> JSONResponse:
    """JSON payload for custom integrations."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    from core.observability.dashboard_data import build_dashboard_payload

    body = build_dashboard_payload(role=role, window_hours=window_hours)
    return JSONResponse(content=body)
