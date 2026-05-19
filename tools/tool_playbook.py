"""Compact per-tool intent lines for the agent system prompt (see ``render_compact_playbook``)."""

from __future__ import annotations

# Nuanced one-liners (override pattern defaults).
_PRECISE: dict[str, str] = {
    "research": "Deep web/X discovery when cheap tools cannot answer; cost=llm.",
    "research_engine": "In-process scorer only; evolution lives in research_daemon.py.",
    "prior_research": "Pull cached multi-agent runs; cost=free (reads DB).",
    "context_quality": "Inspect researcher status, memory source (postgres vs local), risk multiplier, and overall context quality. Call this at the start of any cycle in Independent Mode to know exactly how much you can trust your information.",
    "briefing": "Default summary includes template OOS track_record on ACTION_REQUIRED + template_leaderboard; detail=strategies for full table.",
    "plan_order": "Size/type/stop plan before any live order; universe guard on entries.",
    "enter_option": "Contract pick + optional execute; prefer after plan_order/chain.",
    "calculate_size": "Risk-based share count; needs fresh quote; use before limits.",
    "instrument_selector": "Map thesis to structures; use before exotic option combos.",
    "flatten_limits": "EMERGENCY: cancel+flatten via limits; operator-only.",
    "cancel_all_orphans": "EMERGENCY: cancel orphans; operator-only.",
    "cancel_order": "Cancel one working order by id; after verify open_orders.",
    "cancel_stops": "Bulk-cancel stops for symbol; may warn if other clientIds.",
    "modify_stop": "Adjust aux on existing stop; needs order_id from open_orders.",
    "signal_breakdown": "Why composite moved; needs recent composite row.",
    "execution_status": "Autoresearch snapshot stats + slippage tables.",
    "trader_rules": "Engine + risk rails status; alias for status/engine_status.",
    "open_hypotheses": "Review queued trader hypotheses.",
    "quality_status": "Read-only inspect tool (PR3). Returns canonical QualityMatrix status: overall_quality (full/limited/minimal/degraded), risk_multiplier, blocked_tool_categories, force_conservative_reasoning, llm_call_config, can_initiate_new_risk, global/symbol aggregates, recent_provenance count, and to_prompt_block summary. Primary calibration entrypoint; call early in every Independent Mode cycle. Pure inspection — enforcement is host-side.",
    "quality_for_symbol": "Read-only inspect tool (PR3). Per-symbol SymbolQuality view: local quality_score (0-1 from recent trade_feedback gaps + volume), avg_execution_gap, feedback_count, recent_pnl_samples. Use to differentiate strong local track-record names vs weak before any sizing or conviction in Independent Mode. Cross with provenance_audit for tool history on that symbol.",
    "provenance_audit": "Read-only inspect tool (PR3 Tool-Provenance Heavy core). Heavy forensics: recent ToolUsageRecords (tool_name, called_at, symbol, success, latency_ms, source, context) + DecisionProvenanceSnapshots (ts, cycle_id, decision_type, tools_used list at that moment, context_quality, outcome, notes, quality_state). Supports window (1-30, default 12) and symbol filter. The key signal for detecting tool staleness, decision coverage gaps, and what the system actually had when it made prior choices. Filter per-symbol to audit your thesis support. Pure read-only; no mutation.",
    "current_constraints": "Compat policy summary (Balanced polish). Risk scale, mode, new_entry_policy. Prefer the three canonical quality_* tools above for full matrix + provenance depth.",
    "update_working_memory": "Structured memory for thesis/watchlist; caps per section.",
    "clear_working_memory_entry": "Remove one WM row by section+entry_id.",
    "multi_leg": "Generic spread dispatcher; validate strikes exist on chain first.",
    "refresh_state": "Full textual snapshot; heavy—use when context stale.",
}

_ORDER_FAMILY = frozenset(
    {
        "market_order",
        "limit_order",
        "stop_order",
        "stop_limit",
        "trailing_stop",
        "trailing_stop_limit",
        "bracket_order",
        "oca_order",
        "moc_order",
        "moo_order",
        "loc_order",
        "loo_order",
        "gtd_order",
        "fok_order",
        "ioc_order",
        "adaptive_order",
        "midprice_order",
        "relative_order",
        "vwap_order",
        "twap_order",
        "iceberg_order",
        "snap_mid_order",
    }
)

_OPTION_STRUCT = frozenset(
    {
        "buy_option",
        "vertical_spread",
        "iron_condor",
        "iron_butterfly",
        "straddle",
        "strangle",
        "calendar_spread",
        "diagonal_spread",
        "butterfly",
        "ratio_spread",
        "jade_lizard",
        "collar",
        "covered_call",
        "cash_secured_put",
        "protective_put",
        "close_option",
        "close_spread",
        "roll_option",
    }
)

_RESEARCH_DATA = frozenset(
    {
        "quote",
        "candles",
        "fundamentals",
        "earnings",
        "atr",
        "iv_info",
        "news",
        "analysts",
        "extended_fundamentals",
        "institutional_data",
        "insider_data",
        "peer_comparison",
        "chart_intraday",
        "chart_swing",
        "chart_full",
        "chart_quick",
        "option_chain",
        "option_quote",
        "option_greeks",
        "position_greeks",
    }
)

_KNOWLEDGE = frozenset({"market_hours", "budget", "economic_calendar"})

_ACCOUNT = frozenset(
    {
        "account",
        "positions",
        "open_orders",
        "get_position",
        "refresh_state",
    }
)

_STATS = frozenset({"stats", "daily_summary", "review_trades"})

SECTION: dict[str, str] = {}

for _t in _RESEARCH_DATA | {"research", "research_engine", "prior_research", "briefing", "context_quality"}:
    SECTION[_t] = "RESEARCH & DATA"
for _t in _KNOWLEDGE:
    SECTION[_t] = "SESSION & COST"
for _t in ("plan_order", "enter_option", "calculate_size", "instrument_selector"):
    SECTION[_t] = "PLANNING & SIZING"
for _t in _ACCOUNT:
    SECTION[_t] = "BROKER READ"
for _t in (
    "cancel_order",
    "cancel_stops",
    "cancel_all_orphans",
    "flatten_limits",
    "modify_stop",
):
    SECTION[_t] = "ORDER CONTROL"
for _t in _ORDER_FAMILY:
    SECTION[_t] = "STOCK ORDERS"
for _t in _OPTION_STRUCT:
    SECTION[_t] = "OPTION ORDERS"
for _t in ("option_chain", "option_quote", "option_greeks", "position_greeks"):
    SECTION[_t] = "RESEARCH & DATA"
for _t in ("multi_leg",):
    SECTION[_t] = "OPTION ORDERS"
for _t in ("execution_status", "trader_rules", "open_hypotheses", "signal_breakdown", "quality_status", "quality_for_symbol", "provenance_audit", "current_constraints"):
    SECTION[_t] = "OBSERVABILITY"
for _t in _STATS:
    SECTION[_t] = "OBSERVABILITY"
for _t in ("update_working_memory", "clear_working_memory_entry"):
    SECTION[_t] = "WORKING MEMORY"


def _register_missing_sections() -> None:
    from tools.tools_executor import get_valid_actions

    for t in get_valid_actions():
        if t not in SECTION:
            SECTION[t] = "OTHER"


_register_missing_sections()

_CACHE: dict[str, str] | None = None


def _line_for(name: str) -> str:
    if name in _PRECISE:
        return _PRECISE[name]
    if name in _RESEARCH_DATA:
        return "Screening read; use before conviction trades. cost=data."
    if name in _KNOWLEDGE:
        return "Context for scheduling/risk; cost=free or cheap API."
    if name in _ACCOUNT:
        return "Broker/account snapshot; start-of-cycle hygiene. cost=broker read."
    if name in _ORDER_FAMILY:
        return "Live stock order via IBKR; clear working orders if IB caps; cost=broker."
    if name in _OPTION_STRUCT:
        return "Options path; listed strikes+exp; margin/cash rules apply. cost=broker."
    if name in _STATS:
        return "Performance introspection; cost=free(DB)."
    if name in ("execution_status", "trader_rules", "open_hypotheses", "signal_breakdown", "quality_status", "quality_for_symbol", "provenance_audit", "current_constraints"):
        return "Quality/constraints/provenance forensics. cost=free."
    if name in ("update_working_memory", "clear_working_memory_entry"):
        return "Human-in-loop memory; keep entries short. cost=free."
    return "Utility tool; see tools_executor AVAILABLE TOOLS for params. cost=mixed."


def _lines() -> dict[str, str]:
    global _CACHE
    if _CACHE is None:
        from tools.tools_executor import get_valid_actions

        _CACHE = {n: _line_for(n) for n in get_valid_actions()}
    return _CACHE


def playbook_line(tool: str) -> str:
    """Return the compact playbook line for one tool (must exist in registry)."""
    lines = _lines()
    if tool not in lines:
        raise KeyError(f"Unknown tool {tool!r}")
    return lines[tool]


def render_compact_playbook(max_chars: int = 4000) -> str:
    """Render grouped playbook text for injection into the system prompt."""
    from tools.tools_executor import get_valid_actions

    lines_map = _lines()
    missing = set(get_valid_actions()) - set(lines_map)
    if missing:
        raise RuntimeError(f"tool_playbook missing keys: {sorted(missing)}")

    section_order = [
        "RESEARCH & DATA",
        "SESSION & COST",
        "PLANNING & SIZING",
        "BROKER READ",
        "ORDER CONTROL",
        "STOCK ORDERS",
        "OPTION ORDERS",
        "OBSERVABILITY",
        "WORKING MEMORY",
        "OTHER",
    ]
    blocks: list[str] = []
    header = "=== TOOL PLAYBOOK (compact) ==="
    blocks.append(header)
    for sec in section_order:
        tools = sorted(t for t, s in SECTION.items() if s == sec and t in lines_map)
        if not tools:
            continue
        body = "\n".join(f"  {t}: {lines_map[t]}" for t in tools)
        blocks.append(f"{sec}:\n{body}")
    text = "\n\n".join(blocks)
    trailer = "\n\n(playbook truncated — see tools/tools_executor.py AVAILABLE TOOLS)"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(trailer)] + trailer


def validate_playbook_complete() -> None:
    """Import-time guard in tests; ensures registry and playbook stay aligned."""
    from tools.tools_executor import get_valid_actions

    reg = set(get_valid_actions())
    have = set(_lines())
    if reg != have:
        raise RuntimeError(
            f"playbook/registry mismatch missing={sorted(reg - have)} extra={sorted(have - reg)}"
        )
