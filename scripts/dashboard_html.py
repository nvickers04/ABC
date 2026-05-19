"""Browser-friendly HTML report for profitability cycle logs."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.profit_summary import aggregate_entries


def build_profile_timeseries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-profile cycle P&L and running sum over entry timestamps."""
    sorted_entries = sorted(entries, key=lambda e: str(e.get("ts") or ""))
    profiles = sorted({str(e.get("profit_profile") or "balanced") for e in sorted_entries})
    running: dict[str, float] = {p: 0.0 for p in profiles}
    series: dict[str, dict[str, list[Any]]] = {
        p: {"ts": [], "cycle_pnl": [], "cumulative_profile_pnl": []} for p in profiles
    }
    all_ts: list[str] = []
    all_cycle_pnl: list[float] = []
    all_profile: list[str] = []

    for e in sorted_entries:
        prof = str(e.get("profit_profile") or "balanced")
        cpnl = float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
        ts = str(e.get("ts") or "")
        running[prof] = running.get(prof, 0.0) + cpnl
        series[prof]["ts"].append(ts)
        series[prof]["cycle_pnl"].append(round(cpnl, 4))
        series[prof]["cumulative_profile_pnl"].append(round(running[prof], 4))
        all_ts.append(ts)
        all_cycle_pnl.append(round(cpnl, 4))
        all_profile.append(prof)

    return {
        "profiles": profiles,
        "by_profile": series,
        "scatter": {"ts": all_ts, "cycle_pnl": all_cycle_pnl, "profile": all_profile},
    }


def build_daily_profile_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (session_date, profit_profile)."""
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entries:
        day = str(e.get("session_date") or "")[:10] or "unknown"
        prof = str(e.get("profit_profile") or "balanced")
        key = (day, prof)
        if key not in buckets:
            buckets[key] = {
                "session_date": day,
                "profit_profile": prof,
                "cycles": 0,
                "cycle_pnl_usd": 0.0,
                "wins": 0,
                "trade_like": 0,
            }
        buckets[key]["cycles"] += 1
        cpnl = float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
        buckets[key]["cycle_pnl_usd"] += cpnl
        action = (e.get("trade_outcome") or {}).get("action", "")
        if action and action not in ("quality_status", "briefing", "market_hours", ""):
            if cpnl != 0 or (e.get("trade_outcome") or {}).get("order_id"):
                buckets[key]["trade_like"] += 1
                if cpnl > 0:
                    buckets[key]["wins"] += 1
    rows = list(buckets.values())
    for row in rows:
        row["cycle_pnl_usd"] = round(float(row["cycle_pnl_usd"]), 2)
        tl = int(row["trade_like"])
        row["win_rate_pct"] = round(row["wins"] / tl * 100.0, 1) if tl else 0.0
    rows.sort(key=lambda r: (r["session_date"], r["profit_profile"]), reverse=True)
    return rows


def _fmt_money(v: float) -> str:
    return f"${v:+,.2f}"


def _summary_cards(agg: dict[str, Any]) -> str:
    if not agg.get("cycles"):
        return "<p class='muted'>No cycle data in this window.</p>"
    cards = [
        ("Cycles", str(agg["cycles"])),
        ("Cycle P&L (sum)", _fmt_money(float(agg.get("total_cycle_pnl_usd", 0)))),
        ("Cumulative realized", _fmt_money(float(agg.get("cumulative_realized_pnl_usd", 0)))),
        ("LLM cost", f"${float(agg.get('llm_cost_usd', 0)):.4f}"),
        ("Win rate", f"{float(agg.get('win_rate_pct', 0)):.1f}%"),
        ("Max drawdown", f"{float(agg.get('max_drawdown_pct', 0)):.2f}%"),
    ]
    parts = [
        f"<motion-card><label>{html.escape(label)}</label>"
        f"<motion-value>{html.escape(value)}</motion-value></motion-card>"
        for label, value in cards
    ]
    inner = "".join(parts)
    inner = (
        inner.replace("<motion-card>", "<div class='card'>")
        .replace("</motion-card>", "</div>")
        .replace("<motion-value>", "<div class='value'>")
        .replace("</motion-value>", "</div>")
    )
    return "<div class='cards'>" + inner + "</div>"


def _profile_table(by_profile: dict[str, dict[str, Any]]) -> str:
    if not by_profile:
        return "<p class='muted'>No profile breakdown.</p>"
    rows_html = []
    for prof in sorted(by_profile.keys()):
        s = by_profile[prof]
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(prof)}</td>"
            f"<td>{int(s.get('cycles', 0))}</td>"
            f"<td>{_fmt_money(float(s.get('total_cycle_pnl_usd', 0)))}</td>"
            f"<td>{_fmt_money(float(s.get('last_cumulative_realized_pnl_usd', 0)))}</td>"
            f"<td>${float(s.get('llm_cost_usd', 0)):.4f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Profile</th><th>Cycles</th><th>Cycle P&L sum</th>"
        "<th>Last cumulative</th><th>LLM cost</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def _daily_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No daily rows.</p>"
    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(r['session_date'])}</td>"
            f"<td>{html.escape(r['profit_profile'])}</td>"
            f"<td>{r['cycles']}</td>"
            f"<td>{_fmt_money(float(r['cycle_pnl_usd']))}</td>"
            f"<td>{r['win_rate_pct']:.1f}%</td>"
            f"<td>{r['trade_like']}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Date</th><th>Profile</th><th>Cycles</th><th>Cycle P&L</th>"
        "<th>Win rate</th><th>Trade-like</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_html_report(
    entries: list[dict[str, Any]],
    *,
    window_label: str,
    title: str = "ABC Profitability Dashboard",
) -> str:
    """Return a self-contained HTML page (Plotly via CDN)."""
    agg = aggregate_entries(entries)
    ts_data = build_profile_timeseries(entries)
    daily_rows = build_daily_profile_rows(entries)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    q = agg.get("last_quality") or {}

    plot_payload = {
        "profiles": ts_data["profiles"],
        "by_profile": ts_data["by_profile"],
        "scatter": ts_data["scatter"],
        "daily": daily_rows,
    }
    plot_json = json.dumps(plot_payload, default=str)
    cards_html = _summary_cards(agg)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3d9cf5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1.5rem 2rem 3rem;
      line-height: 1.45;
    }}
    h1 {{ font-size: 1.6rem; margin: 0 0 0.25rem; }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    section {{
      background: var(--panel);
      border-radius: 10px;
      padding: 1rem 1.25rem;
      margin-bottom: 1.25rem;
    }}
    h2 {{ font-size: 1.1rem; margin: 0 0 0.75rem; color: var(--accent); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }}
    .card {{
      background: #243044;
      border-radius: 8px;
      padding: 0.75rem 1rem;
    }}
    .card label {{
      display: block;
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card .value {{ font-size: 1.25rem; font-weight: 600; margin-top: 0.25rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.45rem 0.6rem;
      border-bottom: 1px solid #2d3a4f;
    }}
    th {{ color: var(--muted); font-weight: 500; }}
    .chart {{ width: 100%; min-height: 320px; }}
    .muted {{ color: var(--muted); }}
    .quality {{ font-size: 0.9rem; margin-top: 0.75rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="sub">Window: {html.escape(window_label)} · Generated {html.escape(generated)} · {len(entries)} entries</p>

  <section>
    <h2>Summary</h2>
    {cards_html}
    <p class="quality">Last quality: <strong>{html.escape(str(q.get('overall_quality', '?')))}</strong>
      · risk multiplier {float(q.get('risk_multiplier', 0)):.2f}
      · execution {float(q.get('execution_quality', 0)):.2f}</p>
  </section>

  <section>
    <h2>Profile P&amp;L over time</h2>
    <div id="chart_cumulative" class="chart"></div>
  </section>

  <section>
    <h2>Per-cycle P&amp;L (colored by profile)</h2>
    <div id="chart_scatter" class="chart"></div>
  </section>

  <section>
    <h2>Daily P&amp;L by profile</h2>
    <div id="chart_daily" class="chart"></div>
  </section>

  <section>
    <h2>Profile totals (window)</h2>
    {_profile_table(agg.get("by_profile") or {})}
  </section>

  <section>
    <h2>Daily breakdown</h2>
    {_daily_table(daily_rows)}
  </section>

  <script>
    const DATA = {plot_json};
    const layoutBase = {{
      paper_bgcolor: '#1a2332',
      plot_bgcolor: '#1a2332',
      font: {{ color: '#e7ecf3' }},
      margin: {{ t: 40, r: 24, b: 48, l: 56 }},
      legend: {{ orientation: 'h', y: 1.12 }},
      xaxis: {{ gridcolor: '#2d3a4f' }},
      yaxis: {{ gridcolor: '#2d3a4f', tickprefix: '$' }},
    }};

    const cumTraces = DATA.profiles.map((p) => ({{
      type: 'scatter',
      mode: 'lines+markers',
      name: p,
      x: DATA.by_profile[p].ts,
      y: DATA.by_profile[p].cumulative_profile_pnl,
    }}));
    Plotly.newPlot('chart_cumulative', cumTraces, {{
      ...layoutBase,
      title: 'Cumulative cycle P&L by profile',
      yaxis: {{ ...layoutBase.yaxis, title: 'USD (sum of cycle P&L)' }},
    }}, {{ responsive: true }});

    const colors = {{}};
    DATA.profiles.forEach((p, i) => {{ colors[p] = i; }});
    Plotly.newPlot('chart_scatter', [{{
      type: 'scatter',
      mode: 'markers',
      x: DATA.scatter.ts,
      y: DATA.scatter.cycle_pnl,
      text: DATA.scatter.profile,
      marker: {{
        size: 8,
        color: DATA.scatter.profile.map(p => colors[p] || 0),
        colorscale: 'Viridis',
        showscale: false,
      }},
      hovertemplate: '%{{text}}<br>%{{x}}<br>$%{{y:.2f}}<extra></extra>',
    }}], {{
      ...layoutBase,
      title: 'Cycle realized P&L',
      showlegend: false,
    }}, {{ responsive: true }});

    const days = [...new Set(DATA.daily.map(r => r.session_date))].sort();
    const profs = DATA.profiles;
    const dailyTraces = profs.map(p => ({{
      type: 'bar',
      name: p,
      x: days,
      y: days.map(d => {{
        const row = DATA.daily.find(r => r.session_date === d && r.profit_profile === p);
        return row ? row.cycle_pnl_usd : 0;
      }}),
    }}));
    Plotly.newPlot('chart_daily', dailyTraces, {{
      ...layoutBase,
      title: 'Daily cycle P&L by profile',
      barmode: 'group',
      yaxis: {{ ...layoutBase.yaxis, title: 'USD' }},
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""


def write_html_report(
    path: Path,
    entries: list[dict[str, Any]],
    *,
    window_label: str,
    title: str = "ABC Profitability Dashboard",
) -> Path:
    """Write HTML report to ``path`` (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_html_report(entries, window_label=window_label, title=title),
        encoding="utf-8",
    )
    return path
