"""Meta-optimization: Grok reviews live performance and suggests prompt/tool edits (review-only)."""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_PATH = _REPO_ROOT / "core" / "_system_prompt_template.txt"
_EXECUTOR_PATH = _REPO_ROOT / "tools" / "tools_executor.py"

# Editable prompt regions (review-only; user applies diffs manually).
PROMPT_TARGET_IDS: tuple[str, ...] = (
    "system_prompt_template",
    "mode_guidance_paper",
    "mode_guidance_live",
    "mode_guidance_aggressive_paper",
    "independent_mode_cycle_guidance",
    "connected_mode_cycle_guidance",
)

PROMPT_TARGET_FILES: dict[str, str] = {
    "system_prompt_template": "core/_system_prompt_template.txt",
    "mode_guidance_paper": "core/prompt_config.py",
    "mode_guidance_live": "core/prompt_config.py",
    "mode_guidance_aggressive_paper": "core/prompt_config.py",
    "independent_mode_cycle_guidance": "core/prompt_config.py",
    "connected_mode_cycle_guidance": "core/prompt_config.py",
}


@dataclass
class StrategyCorpus:
    """Snapshot of strategy text sent to the meta-optimizer."""

    generated_at: str
    active_profile: str
    trading_mode: str
    prompt_regions: dict[str, str] = field(default_factory=dict)
    tool_schemas: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "active_profile": self.active_profile,
            "trading_mode": self.trading_mode,
            "prompt_regions": self.prompt_regions,
            "tool_schemas": self.tool_schemas,
            "notes": self.notes,
        }


def evolution_enabled() -> bool:
    raw = os.getenv("EVOLVE_STRATEGY_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def collect_performance_digest(*, days: int = 7) -> dict[str, Any]:
    """Last N days: cycle logs, optional DB trades, profile breakdown."""
    from core.live_profile_optimize import load_live_entries, score_profile_entries
    from core.profit_config_state import get_active_profile_label
    from core.profit_summary import aggregate_entries

    since = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    entries = load_live_entries(days=days)
    stats = aggregate_entries(entries)

    by_profile: dict[str, Any] = {}
    prof_groups: dict[str, list] = {}
    for e in entries:
        p = str(e.get("profit_profile") or "balanced").lower()
        prof_groups.setdefault(p, []).append(e)
    for p, rows in prof_groups.items():
        by_profile[p] = score_profile_entries(rows)

    trade_rows: list[dict[str, Any]] = []
    for e in entries:
        trade = e.get("trade_outcome") or {}
        action = str(trade.get("action") or "").strip()
        if not action or action in ("quality_status", "briefing", "market_hours"):
            continue
        pnl = float(e.get("pnl", {}).get("cycle_realized_pnl_usd", 0) or 0)
        if pnl == 0 and not trade.get("order_id"):
            continue
        trade_rows.append(
            {
                "ts": e.get("ts"),
                "session_date": e.get("session_date"),
                "profile": e.get("profit_profile"),
                "symbol": trade.get("symbol"),
                "action": action,
                "pnl_usd": pnl,
                "won": pnl > 0,
            }
        )

    db_trades: list[dict[str, Any]] = []
    try:
        from memory import get_db

        db = get_db()
        cur = db.execute(
            """
            SELECT ts, symbol, side, pnl, held_minutes
            FROM trades
            WHERE ts >= ?
            ORDER BY ts DESC
            LIMIT 200
            """,
            (since.isoformat(),),
        ).fetchall()
        for r in cur:
            db_trades.append(dict(r) if hasattr(r, "keys") else {"ts": r[0], "symbol": r[1], "pnl": r[3]})
    except Exception as e:
        db_trades = [{"error": str(e)}]

    return {
        "window_days": days,
        "window_start": since.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_profile": get_active_profile_label(),
        "cycle_stats": stats,
        "by_profile": by_profile,
        "trade_like_cycles": trade_rows[-50:],
        "db_trades_sample": db_trades[:50],
        "outcomes": stats.get("outcomes"),
    }


def snapshot_strategy_corpus(*, max_tools: int = 30) -> StrategyCorpus:
    """Current system prompt fragments and tool schema descriptions."""
    from core.central_profit_config import get_profit_config
    from core.profit_config_state import get_active_profile_label
    from core.prompt_config import (
        CONNECTED_MODE_CYCLE_GUIDANCE,
        INDEPENDENT_MODE_CYCLE_GUIDANCE,
        MODE_GUIDANCE_AGGRESSIVE_PAPER,
        MODE_GUIDANCE_LIVE,
        MODE_GUIDANCE_PAPER,
    )

    cfg = get_profit_config()
    pc = cfg.prompt
    reg = cfg.tools

    prompt_regions = {
        "system_prompt_template": _TEMPLATE_PATH.read_text(encoding="utf-8")
        if _TEMPLATE_PATH.is_file()
        else "",
        "mode_guidance_paper": pc.mode_guidance_paper,
        "mode_guidance_live": pc.mode_guidance_live,
        "mode_guidance_aggressive_paper": pc.mode_guidance_aggressive_paper,
        "independent_mode_cycle_guidance": pc.independent_mode_cycle_guidance,
        "connected_mode_cycle_guidance": pc.connected_mode_cycle_guidance,
    }
    # Module-level constants (edit location in prompt_config.py).
    prompt_regions["_constants"] = json.dumps(
        {
            "MODE_GUIDANCE_PAPER": MODE_GUIDANCE_PAPER[:200] + "...",
            "MODE_GUIDANCE_LIVE": MODE_GUIDANCE_LIVE[:200] + "...",
        },
        indent=0,
    )

    tools = reg.playbook_lines(max_tools=max_tools)
    notes = [
        "Review-only: apply diffs manually; nothing is auto-patched.",
        "Tool schemas are sourced from tools/tools_executor.py module docstring; "
        "registry copies them in core/tool_registry.py at runtime.",
        f"Executor docstring path: { _EXECUTOR_PATH.relative_to(_REPO_ROOT) }",
    ]

    return StrategyCorpus(
        generated_at=datetime.now(timezone.utc).isoformat(),
        active_profile=get_active_profile_label(),
        trading_mode=str(cfg.trading_mode),
        prompt_regions=prompt_regions,
        tool_schemas=tools,
        notes=notes,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + "\n... [truncated for token budget] ...\n"


def build_meta_optimizer_prompt(performance: dict[str, Any], corpus: StrategyCorpus) -> str:
    """User message for MULTI_AGENT_MODEL (no web tools)."""
    max_chars = int(os.getenv("EVOLVE_STRATEGY_MAX_PROMPT_CHARS", "28000"))
    perf_json = _truncate(json.dumps(performance, indent=2, default=str), max_chars // 2)
    corpus_json = _truncate(
        json.dumps(
            {
                "prompt_target_ids": list(PROMPT_TARGET_IDS),
                "prompt_regions": {k: corpus.prompt_regions.get(k, "")[:4000] for k in PROMPT_TARGET_IDS},
                "tool_schemas": corpus.tool_schemas,
                "notes": corpus.notes,
            },
            indent=2,
            default=str,
        ),
        max_chars // 2,
    )
    return f"""You are a profitability meta-optimizer for an autonomous trading agent (ABC / Grok trader).

Analyze the last {performance.get('window_days', 7)} days of LIVE paper/live cycle logs and trade outcomes.
Suggest CONSERVATIVE, review-only improvements to:
1) System prompt text (targets: {', '.join(PROMPT_TARGET_IDS)})
2) Tool schema descriptions shown to the model (tool name -> params -> intent)

Rules:
- Do NOT suggest disabling safety rails, QualityMatrix, loss caps, or broker guards.
- Prefer small, testable wording changes (clarity, sequencing, when to use tools, R:R discipline).
- Do NOT invent new tools or remove required tools.
- Output MUST be valid JSON only (no markdown outside the JSON), matching this schema:
{{
  "executive_summary": "string",
  "performance_diagnosis": ["string", ...],
  "prompt_suggestions": [
    {{
      "target_id": "mode_guidance_paper",
      "proposed_text": "full replacement text for that region",
      "rationale": "why",
      "risk": "low|medium"
    }}
  ],
  "tool_suggestions": [
    {{
      "tool_name": "plan_order",
      "proposed_schema_description": "{{symbol, side, quantity, ...}} -> intent line",
      "rationale": "why",
      "risk": "low|medium"
    }}
  ],
  "do_not_change": ["string", ...]
}}

Performance data:
{perf_json}

Current strategy corpus:
{corpus_json}
"""


def _parse_json_response(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


async def invoke_meta_optimizer(
    performance: dict[str, Any],
    corpus: StrategyCorpus,
) -> dict[str, Any]:
    """Call MULTI_AGENT_MODEL (xAI) for structured suggestions."""
    from xai_sdk import AsyncClient
    from xai_sdk.chat import user as sdk_user

    from core.prompt_config import get_prompt_config

    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY or GROK_API_KEY required for evolve_strategy")

    model = os.getenv("MULTI_AGENT_MODEL", "").strip() or get_prompt_config().multi_agent_model
    prompt = build_meta_optimizer_prompt(performance, corpus)
    client = AsyncClient(api_key=api_key)
    chat = client.chat.create(model=model, messages=[sdk_user(prompt)])
    response = await chat.sample()
    content = str(response.content or "")
    parsed = _parse_json_response(content)
    cost_usd = 0.0
    try:
        from data.cost_tracker import get_cost_tracker

        if hasattr(response, "usage") and response.usage:
            cost_usd = get_cost_tracker().log_llm_usage(
                model,
                usage=response.usage,
                purpose="evolve_strategy_meta",
            )
    except Exception:
        pass

    return {
        "model": model,
        "raw_response_chars": len(content),
        "estimated_cost_usd": round(cost_usd, 4),
        "suggestions": parsed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _unified_diff_block(
    path: str,
    old: str,
    new: str,
    *,
    from_label: str = "current",
    to_label: str = "proposed",
) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    if not old_lines and not new_lines:
        return ""
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            fromfiledate=from_label,
            tofiledate=to_label,
            lineterm="",
        )
    )


def build_review_diff(
    corpus: StrategyCorpus,
    suggestions: dict[str, Any],
) -> str:
    """Unified diff for user review (not applied automatically)."""
    blocks: list[str] = [
        "# evolve_strategy.py — REVIEW ONLY — apply manually after inspection",
        f"# Generated: {datetime.now(timezone.utc).isoformat()}",
        "# Files: prompt_config.py, _system_prompt_template.txt, tools/tools_executor.py (tool lines)",
        "",
    ]

    for item in suggestions.get("prompt_suggestions") or []:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("target_id") or "").strip()
        proposed = str(item.get("proposed_text") or "")
        if not tid or not proposed:
            continue
        current = corpus.prompt_regions.get(tid, "")
        rel = PROMPT_TARGET_FILES.get(tid, f"core/prompt_config.py ({tid})")
        blocks.append(f"# --- prompt: {tid} — {item.get('rationale', '')[:120]}")
        blocks.append(
            _unified_diff_block(rel, current, proposed, from_label="current", to_label="proposed")
        )
        blocks.append("")

    for item in suggestions.get("tool_suggestions") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name") or "").strip().lower()
        proposed = str(item.get("proposed_schema_description") or "")
        if not name or not proposed:
            continue
        current = corpus.tool_schemas.get(name, "")
        blocks.append(f"# --- tool: {name} — {item.get('rationale', '')[:120]}")
        blocks.append(
            _unified_diff_block(
                f"tools/tools_executor.py ({name} line in module docstring)",
                current,
                proposed,
            )
        )
        blocks.append(
            f"# Registry mirror: core/tool_registry.py ToolSpec.schema_description for '{name}'"
        )
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"


def format_review_report(
    performance: dict[str, Any],
    corpus: StrategyCorpus,
    analysis: dict[str, Any],
) -> str:
    """Human-readable summary for terminal / logs."""
    sug = analysis.get("suggestions") or {}
    lines = [
        "=" * 72,
        "  Strategy evolution (review-only)",
        "=" * 72,
        f"  Model:     {analysis.get('model')}",
        f"  Profile:   {corpus.active_profile}  mode={corpus.trading_mode}",
        f"  Window:    {performance.get('window_days')} days ({performance.get('window_start', '')[:10]})",
        f"  Cycles:    {(performance.get('cycle_stats') or {}).get('cycles', 0)}",
        f"  Est. cost: ${analysis.get('estimated_cost_usd', 0):.4f}",
        "",
        "  Executive summary",
        f"    {sug.get('executive_summary', '(none)')}",
        "",
        f"  Prompt suggestions: {len(sug.get('prompt_suggestions') or [])}",
        f"  Tool suggestions:   {len(sug.get('tool_suggestions') or [])}",
        "",
        "  Output is a .patch diff — inspect and apply manually.",
        "=" * 72,
    ]
    return "\n".join(lines)


async def run_strategy_evolution(
    *,
    days: int = 7,
    max_tools: int = 30,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Full pipeline: digest → corpus → Grok → diff artifacts."""
    out_dir = output_dir or (_REPO_ROOT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    performance = collect_performance_digest(days=days)
    corpus = snapshot_strategy_corpus(max_tools=max_tools)
    analysis = await invoke_meta_optimizer(performance, corpus)
    suggestions = analysis.get("suggestions") or {}

    diff_text = build_review_diff(corpus, suggestions)
    patch_path = out_dir / f"evolve_strategy_{stamp}.patch"
    json_path = out_dir / f"evolve_strategy_{stamp}.json"

    payload = {
        "review_only": True,
        "performance": performance,
        "corpus": corpus.to_dict(),
        "analysis": analysis,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    patch_path.write_text(diff_text, encoding="utf-8")

    return {
        "patch_path": str(patch_path),
        "json_path": str(json_path),
        "report": format_review_report(performance, corpus, analysis),
        "analysis": analysis,
    }


__all__ = [
    "PROMPT_TARGET_IDS",
    "StrategyCorpus",
    "build_meta_optimizer_prompt",
    "build_review_diff",
    "collect_performance_digest",
    "evolution_enabled",
    "format_review_report",
    "invoke_meta_optimizer",
    "run_strategy_evolution",
    "snapshot_strategy_corpus",
]
