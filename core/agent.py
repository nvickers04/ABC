"""
Hybrid ReAct Agent Loop — Reasoning + Multi-Agent Research

Architecture:
  - REASONING model (single-agent) drives the ReAct loop with client-side tools
  - MULTI-AGENT model (4/16 agents) is exposed as a research() tool for web/X search
  - The reasoning model decides when to call research() — no separate coordination

Grok chains actions until it signals 'done' to refresh context.
research() is the primary discovery mechanism.
"""

import ast
import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

from core.config import (
    SYSTEM_PROMPT,
    RISK_PER_TRADE,
    CYCLE_SLEEP_SECONDS,

    MAX_DAILY_LOSS_PCT,
    MAX_DAILY_LLM_COST,
    LLM_TEMPERATURE,
    LLM_SEED,
    LLM_MAX_TOKENS,
    TRADING_MODE,
    PAPER_AGGRESSIVE,
    MIN_RR_RATIO,
)
from xai_sdk.chat import system as sdk_system, user as sdk_user

from core.grok_llm import get_grok_llm
from data.cost_tracker import get_cost_tracker
from data.data_provider import get_data_provider
from data.broker_gateway import create_gateway, BrokerConfigError
from data.market_hours import get_market_hours_provider
from tools.tools_executor import ToolExecutor

logger = logging.getLogger(__name__)


# ── JSON Parsing Helpers ────────────────────────────────────────

def _try_repair_json(raw: str) -> list[dict[str, Any]]:
    """Best-effort repair for almost-valid model JSON output.
    
    Handles:
    - Code fence wrappers
    - Smart quotes / trailing commas
    - Truncated JSON (missing closing braces/brackets from token limit)
    """
    text = (raw or "").strip()
    if not text:
        return []

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    elif first != -1 and (last == -1 or last <= first):
        # Truncated JSON — close all open braces/brackets
        text = text[first:]
        text = _close_truncated_json(text)

    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        obj = json.loads(text)
        return [obj] if isinstance(obj, dict) else [i for i in obj if isinstance(i, dict)]
    except Exception:
        pass

    try:
        obj = ast.literal_eval(text)
        return [obj] if isinstance(obj, dict) else [i for i in obj if isinstance(i, dict)]
    except Exception:
        return []


def _close_truncated_json(text: str) -> str:
    """Close truncated JSON by balancing braces/brackets and fixing dangling strings."""
    # Strip trailing partial tokens (incomplete key/value after last comma or colon)
    # Remove trailing partial string value (e.g., `"key": "some truncated text`)
    text = re.sub(r',\s*"[^"]*$', '', text)  # trailing key without value
    text = re.sub(r':\s*"[^"]*$', ': ""', text)  # truncated string value → empty
    text = re.sub(r':\s*$', ': null', text)  # hanging colon
    text = re.sub(r',\s*$', '', text)  # trailing comma

    # Count open/close braces and brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    # Close in reverse order (brackets first since they're usually nested inside objects)
    text += ']' * max(0, open_brackets)
    text += '}' * max(0, open_braces)
    return text


def _parse_json_objects(raw: str) -> list[dict]:
    """Extract all JSON objects from raw LLM output."""
    if "```json" in raw:
        json_str = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        json_str = raw.split("```")[1].split("```")[0]
    else:
        json_str = raw

    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    json_str = json_str.strip()
    while idx < len(json_str):
        brace = json_str.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(json_str, brace)
            objects.append(obj)
            idx = brace + end
        except json.JSONDecodeError:
            idx = brace + 1

    if not objects:
        objects = _try_repair_json(raw)

    return objects


def _now_et() -> datetime:
    """Current time in US/Eastern."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        from datetime import timezone, timedelta
        return datetime.now(timezone(timedelta(hours=-5)))


# ── Order types the agent can use (for tracking in aggressive mode) ──

ALL_ORDER_TYPES = {
    "market_order", "limit_order", "stop_order", "stop_limit",
    "trailing_stop", "bracket_order", "oca_order",
    "adaptive_order", "midprice_order", "vwap_order", "twap_order",
    "relative_order", "snap_mid_order",
    "vertical_spread", "iron_condor", "straddle", "strangle",
    "calendar_spread", "diagonal_spread", "butterfly", "collar",
    "buy_option",
}


# ── The Agent ───────────────────────────────────────────────────

class TradingAgent:
    """
    Minimal autonomous trading agent — pure ReAct loop.

    Grok calls tools, sees results, decides next action.
    No orchestration, no registries, no rigid strategies.
    """

    def __init__(self, gateway, tools: ToolExecutor, grok=None):
        self.grok = grok or get_grok_llm()
        self.gateway = gateway
        self.cost_tracker = get_cost_tracker()
        self.tools = tools

        # Circuit breaker
        self._consecutive_failures = 0
        self._cycle_failures = 0
        self._max_consecutive = 5
        self._max_per_cycle = 8

        # Session state
        self._cycle_id = 0
        self._last_cycle_summary = ""
        self._halted = False
        self._start_of_day_cash: Optional[float] = None
        self._market_snapshots: list[str] = []  # Rolling 5-cycle summaries
        self._research_results: dict[str, tuple[str, str]] = {}  # query -> (result_summary, timestamp)
        self._used_order_types: set[str] = set()  # Track which order types used this session

        logger.info("TradingAgent initialized (minimal ReAct mode)")

    def _append_snapshot(self, summary: str):
        """Append a cycle snapshot, keeping last 5."""
        self._market_snapshots.append(f"[{datetime.now().strftime('%H:%M')}] {summary}")
        if len(self._market_snapshots) > 5:
            self._market_snapshots = self._market_snapshots[-5:]

    async def _get_cash(self) -> float:
        """Get current cash balance from broker (TotalCashValue ONLY, never AvailableFunds)."""
        try:
            summary = await self.gateway.get_account_summary()
            return summary.get("totalcashvalue", 0) or 0
        except Exception:
            return self.gateway.cash_value if self.gateway else 0

    def _capture_start_of_day_cash(self):
        """Capture cash at session start for daily loss tracking."""
        if self._start_of_day_cash is not None:
            return
        # Use net liquidation as the baseline (not cash — cash drops on stock buys)
        net_liq = self.gateway.net_liquidation if self.gateway else 0
        cash = self.gateway.cash_value if self.gateway else 0
        baseline = net_liq if net_liq > 0 else cash
        if baseline > 0:
            self._start_of_day_cash = baseline
            logger.info(
                f"Start-of-day Net Liq: ${baseline:,.2f} (cash: ${cash:,.2f}) "
                f"(loss limit: -{MAX_DAILY_LOSS_PCT}% = ${baseline * MAX_DAILY_LOSS_PCT / 100:,.2f})"
            )

    def _check_daily_loss(self) -> Optional[float]:
        """Check if daily loss limit breached using NET LIQUIDATION (not cash).
        
        Cash drops when you buy stock but net liq stays the same — using cash
        would trigger false emergency flattens on any stock purchase.
        """
        if not self._start_of_day_cash or self._start_of_day_cash <= 0:
            return None
        # Use net liquidation (includes position value), not raw cash
        current = self.gateway.net_liquidation if self.gateway else 0
        if current <= 0:
            # Fallback to cash if net liq not available
            current = self.gateway.cash_value if self.gateway else 0
        if current <= 0:
            return None
        loss_pct = (self._start_of_day_cash - current) / self._start_of_day_cash * 100
        return loss_pct if loss_pct >= MAX_DAILY_LOSS_PCT else None

    async def _emergency_flatten(self, reason: str):
        """EMERGENCY: Flatten all positions and halt for the day."""
        logger.critical("=" * 60)
        logger.critical(f"EMERGENCY FLATTEN: {reason}")
        logger.critical("=" * 60)
        if self.gateway:
            await self.gateway.flatten_all()
        self._halted = True

    def _check_llm_cost(self) -> bool:
        """Returns True if daily LLM cost ceiling reached."""
        summary = self.cost_tracker.get_budget_summary()
        return summary.today_llm_cost >= MAX_DAILY_LLM_COST

    def _get_research_briefing(self) -> str | None:
        """Read best strategy per track + live signals from research DB."""
        try:
            from memory import get_db
            db = get_db()

            # Best strategy per track
            rows = db.execute(
                """SELECT s.id, s.track, s.expectancy, s.hit_rate, s.avg_rr,
                          s.total_signals, s.llm_analysis
                   FROM strategies s
                   INNER JOIN (
                       SELECT track, MAX(expectancy) as max_exp
                       FROM strategies WHERE kept = 1
                       GROUP BY track
                   ) best ON s.track = best.track AND s.expectancy = best.max_exp
                   WHERE s.kept = 1
                   ORDER BY s.expectancy DESC"""
            ).fetchall()
            if not rows:
                return None

            lines = ["=== RESEARCH BRIEFING ==="]
            for row in rows:
                lines.append(
                    f"  [{row['track'].upper()}] #{row['id']}: "
                    f"exp={row['expectancy']:.4f} hit={row['hit_rate']:.0f}% "
                    f"R:R={row['avg_rr']:.1f} ({row['total_signals']} signals)"
                )
                if row["llm_analysis"]:
                    snippet = row["llm_analysis"][:200].replace("\n", " ")
                    lines.append(f"    Insight: {snippet}")

            # Live signals for today (all tracks)
            live = db.execute(
                """SELECT track, symbol, direction, order_type, setup_type,
                          entry_price, target_price, stop_price
                   FROM live_signals ORDER BY track, symbol"""
            ).fetchall()
            if live:
                lines.append(f"  Live signals today: {len(live)}")
                for sig in live[:15]:
                    lines.append(
                        f"    [{sig['track']}] {sig['symbol']} {sig['direction']} "
                        f"{sig['order_type']} @ {sig['entry_price']:.2f} "
                        f"tgt={sig['target_price']:.2f} stp={sig['stop_price']:.2f} "
                        f"({sig['setup_type']})"
                    )
                if len(live) > 15:
                    lines.append(f"    ... and {len(live) - 15} more")

            return "\n".join(lines)
        except Exception:
            return None

    def record_trade(self, symbol: str, side: str, pnl: float, held_minutes: int = 0):
        """Record a closed trade into the research memory DB."""
        try:
            from memory import get_db
            from datetime import datetime, timezone
            db = get_db()
            db.execute(
                """INSERT INTO trades (ts, symbol, side, pnl, held_minutes)
                   VALUES (?, ?, ?, ?, ?)""",
                (datetime.now(timezone.utc).isoformat(), symbol, side, pnl, held_minutes),
            )
            db.commit()
        except Exception as e:
            logger.debug(f"Failed to record trade: {e}")

    async def _build_state_context(self) -> str:
        """Build the complete dynamic state context for the agent.

        One function, one source of truth. Includes:
        - Market session + session-specific constraints
        - Account balances
        - Open positions with P&L
        - Open orders
        """
        lines = []

        # ── Market session ──────────────────────────────────────
        session = "UNKNOWN"
        try:
            mh = get_market_hours_provider()
            info = mh.get_session_info()
            session = info["session"].upper()
            time_et = info["current_time_et"]
            detail = ""
            if "minutes_to_open" in info:
                detail = f" | Open in {info['minutes_to_open']}min"
            elif "minutes_to_close" in info:
                detail = f" | Close in {info['minutes_to_close']}min"
            lines.append(f"═══ MARKET: {session} ({time_et} ET){detail} ═══")
        except Exception as e:
            lines.append(f"═══ MARKET: UNKNOWN (error: {e}) ═══")

        # Session constraint reminder (reinforces system prompt rules)
        if session == "PREMARKET":
            lines.append("  ⚠ Limit orders ONLY. bracket_order will timeout. Options won't fill.")
        elif session == "POSTMARKET":
            lines.append("  ⚠ Limit orders ONLY. Market orders rejected.")
        elif session == "CLOSED":
            lines.append("  ⛔ Market CLOSED. Research/plan only — do NOT place orders.")

        lines.append("")
        lines.append("═══ ACCOUNT ═══")
        try:
            summary = await self.gateway.get_account_summary()
            cash = summary.get("totalcashvalue", 0)
            net_liq = summary.get("netliquidation", 0)
            daily_pnl = summary.get("dailypnl", "N/A")
            unreal = summary.get("unrealizedpnl", "N/A")
            real = summary.get("realizedpnl", "N/A")
            lines.append(f"Cash (SIZE FROM THIS): ${cash:,.2f}  |  NetLiq: ${net_liq:,.2f}")
            lines.append(f"Day P&L: {daily_pnl}  |  Unreal: {unreal}  |  Real: {real}")
        except Exception as e:
            lines.append(f"Account error: {e}")

        lines.append("")
        lines.append("═══ POSITIONS (manage first) ═══")
        try:
            positions = await self.gateway.get_positions()
            if not positions:
                lines.append("No open positions.")
            else:
                for p in positions:
                    sym = p.get("symbol", "?")
                    qty = p.get("quantity", 0)
                    avg = p.get("avg_cost", 0)
                    mkt = p.get("market_price", 0)
                    pnl = p.get("unrealized_pnl", 0)
                    sec = p.get("sec_type", "STK")
                    pnl_pct = ((mkt - avg) / avg * 100) if avg > 0 else 0
                    if sec == "OPT":
                        strike = p.get("strike", "")
                        right = p.get("right", "")
                        exp = p.get("expiration", "")
                        dte = "?"
                        try:
                            from datetime import datetime as _dt
                            exp_date = _dt.strptime(str(exp)[:8], "%Y%m%d").date()
                            dte = (exp_date - _dt.now().date()).days
                        except Exception:
                            pass
                        lines.append(
                            f"  {sym} {right}{strike} exp={exp} DTE={dte}: {qty} @ ${avg:.2f} "
                            f"| mkt ${mkt:.2f} | P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )
                    else:
                        lines.append(
                            f"  {sym}: {qty} @ ${avg:.2f} "
                            f"| mkt ${mkt:.2f} | P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )
        except Exception as e:
            lines.append(f"Position error: {e}")

        lines.append("")
        lines.append("═══ OPEN ORDERS ═══")
        try:
            orders = await self.gateway.get_open_orders()
            if not orders:
                lines.append("No open orders.")
            else:
                for o in orders:
                    sym = o.get("symbol", "?")
                    action = o.get("action", "?")
                    qty = o.get("quantity", 0)
                    otype = o.get("order_type", "?")
                    lmt = o.get("lmt_price")
                    aux = o.get("aux_price")
                    oid = o.get("order_id", "?")
                    sec = o.get("sec_type", "STK")
                    trail_pct = o.get("trail_percent")
                    price_str = ""
                    if lmt:
                        price_str = f"lmt ${lmt:.2f}"
                    if aux:
                        price_str += f" aux ${aux:.2f}"
                    if trail_pct:
                        price_str += f" trail {trail_pct:.1f}%"
                    # Tag non-stock orders so agent doesn't confuse them
                    tag = ""
                    if sec == "BAG":
                        tag = " [COMBO/SPREAD]"
                    elif sec == "OPT":
                        tag = " [OPTION]"
                    lines.append(f"  #{oid} {action} {qty} {sym} {otype} {price_str}{tag}")
        except Exception as e:
            lines.append(f"Order error: {e}")

        # ── Research briefing ───────────────────────────────────
        try:
            briefing = self._get_research_briefing()
            if briefing:
                lines.append("")
                lines.append(briefing)
        except Exception as e:
            logger.debug(f"Research briefing unavailable: {e}")

        return "\n".join(lines)

    async def run_cycle(self) -> int:
        """
        Run one cycle of the ReAct loop.

        Returns cooldown seconds.
        """
        self._cycle_failures = 0
        self._cycle_id += 1

        # Build state context from broker
        state_text = await self._build_state_context()

        # Check daily loss
        loss_pct = self._check_daily_loss()
        if loss_pct is not None:
            await self._emergency_flatten(f"Daily loss: -{loss_pct:.1f}%")
            return 999999

        # Check LLM cost
        if self._check_llm_cost():
            logger.warning(f"LLM cost ceiling ${MAX_DAILY_LLM_COST} reached — halting")
            self._halted = True
            return 999999

        # Build context
        cost_line = ""
        try:
            summary = self.cost_tracker.get_budget_summary()
            cost_pct = summary.today_llm_cost / MAX_DAILY_LLM_COST * 100
            cost_line = f"\nLLM COST: ${summary.today_llm_cost:.2f} / ${MAX_DAILY_LLM_COST:.2f} ({cost_pct:.0f}%)"
        except Exception:
            pass

        continuity = ""
        if self._last_cycle_summary:
            continuity = f"\nLAST CYCLE: {self._last_cycle_summary}\n"
        if self._market_snapshots:
            continuity += "RECENT SNAPSHOTS:\n" + "\n".join(self._market_snapshots[-5:]) + "\n"
        if self._research_results:
            continuity += "\n═══ PRIOR RESEARCH (already paid for — reuse before re-researching) ═══\n"
            for query, (summary, ts) in self._research_results.items():
                continuity += f"[{ts}] Q: {query}\n{summary[:800]}\n\n"

        aggressive_nudge = ""
        if PAPER_AGGRESSIVE:
            # Get session to make nudge context-appropriate
            _session = "regular"
            try:
                _mh = get_market_hours_provider()
                _info = _mh.get_session_info()
                _session = _info.get("session", "regular")
            except Exception:
                pass

            # Build order-type diversity hint
            untested = sorted(ALL_ORDER_TYPES - self._used_order_types)
            tested_str = ", ".join(sorted(self._used_order_types)) if self._used_order_types else "none yet"
            untested_str = ", ".join(untested[:10]) if untested else "all tested!"

            if _session == "regular":
                aggressive_nudge = (
                    f"\nAGGRESSIVE TEST: Trade aggressively. Try order types you haven't used yet."
                    f"\n  Used so far: {tested_str}"
                    f"\n  Untested: {untested_str}"
                    f"\n  Pick an untested type when the setup fits."
                )
            elif _session in ("premarket", "postmarket"):
                aggressive_nudge = (
                    "\nAGGRESSIVE TEST: Research movers, queue limit_order entries for open. "
                    "Don't churn on queued orders \u2014 they fill at 9:30. "
                    "Options and bracket_order won't fill until regular hours."
                )
            else:  # closed
                aggressive_nudge = (
                    "\nAGGRESSIVE TEST: Market closed. Research and plan for next session."
                )

        context = f"""{state_text}
{cost_line}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{continuity}{aggressive_nudge}
Account state above. What is your next action?"""

        # ── Build xAI SDK chat instance ─────────────────────────
        chat = self.grok.client.chat.create(
            model=self.grok.model,
            messages=[
                sdk_system(SYSTEM_PROMPT),
                sdk_user(context),
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            seed=LLM_SEED,
        )

        # ── Inner ReAct loop ────────────────────────────────────
        turn = 0

        cycle_actions = []

        while True:
            turn += 1

            # Re-check daily loss every turn
            loss_pct = self._check_daily_loss()
            if loss_pct is not None:
                await self._emergency_flatten(f"Daily loss mid-cycle: -{loss_pct:.1f}%")
                return 999999

            logger.info(f"Cycle {self._cycle_id} turn {turn}")

            try:
                response = await chat.sample()

                raw = response.content

                # Log cost
                usage = response.usage
                self.cost_tracker.log_llm_call(
                    model=self.grok.model,
                    tokens_in=usage.prompt_tokens,
                    tokens_out=usage.completion_tokens,
                    purpose="agent_loop",
                )
                cost_summary = self.cost_tracker.get_budget_summary()
                reasoning_tok = getattr(usage, 'reasoning_tokens', 0) or 0
                logger.info(
                    f"LLM {usage.prompt_tokens}+{usage.completion_tokens} tok "
                    f"(reasoning: {reasoning_tok}) | ${cost_summary.today_llm_cost:.4f} today"
                )

                # Log what the model actually said
                logger.info(f"Grok -> {(raw or '(empty)')[:500]}")

                # ── Parse JSON actions ──────────────────────────
                json_objects = _parse_json_objects(raw or "")

                if not json_objects:
                    logger.warning(f"No valid JSON: {raw[:100]}")
                    chat.append(response)
                    chat.append(sdk_user("Please respond with valid JSON for your next action."))
                    continue

                # Process actions — one real action per response
                last_result = None

                for action_data in json_objects:
                    action = action_data.get("action", "")

                    # "done" ends the cycle
                    if action in ("done", "wait", "FINAL_DECISION"):
                        summary = action_data.get('summary', action_data.get('reason', action_data.get('reasoning', '')))[:200]
                        cooldown = action_data.get('cooldown', CYCLE_SLEEP_SECONDS)
                        try:
                            cooldown = max(5, min(int(cooldown), 3600))  # 5s–1hr bounds
                        except (TypeError, ValueError):
                            cooldown = CYCLE_SLEEP_SECONDS
                        logger.info(f"Decision: done ({cooldown}s) | {summary}")
                        cycle_actions.append("done")
                        self._last_cycle_summary = (
                            f"Cycle {self._cycle_id}: {len(cycle_actions)} actions — "
                            + ", ".join(cycle_actions[-5:])
                        )
                        self._append_snapshot(f"C{self._cycle_id}: done")
                        return cooldown

                    else:
                        # ── Execute tool action ────────────────
                        last_result = await self.tools.execute(action, action_data)
                        sym = action_data.get('symbol', '')

                        # Cache research results for cross-cycle memory
                        if action == 'research' and last_result.success:
                            data = last_result.data if hasattr(last_result, 'data') else last_result
                            if isinstance(data, dict) and 'result' in data:
                                query = action_data.get('query', '')
                                # Keep first 2000 chars of result as summary
                                summary_text = str(data['result'])[:2000]
                                ts = datetime.now().strftime('%H:%M')
                                self._research_results[query] = (summary_text, ts)
                                # Evict oldest if > 10 cached queries
                                if len(self._research_results) > 10:
                                    oldest_key = next(iter(self._research_results))
                                    del self._research_results[oldest_key]

                        # Clean tool result log — show success/error + key data
                        if last_result.success:
                            data = last_result.data if hasattr(last_result, 'data') else last_result
                            # Extract the most useful summary fields
                            if isinstance(data, dict):
                                summary_keys = ['strategy', 'order_id', 'filled', 'last', 'bid', 'ask',
                                                'change_pct', 'volume', 'count', 'result', 'decision']
                                summary = {k: data[k] for k in summary_keys if k in data}
                                if summary:
                                    logger.info(f"Tool OK: {action}({sym}) -> {summary}")
                                else:
                                    logger.info(f"Tool OK: {action}({sym}) -> {str(data)[:200]}")
                            else:
                                logger.info(f"Tool OK: {action}({sym}) -> {str(data)[:200]}")
                        else:
                            err = last_result.error if hasattr(last_result, 'error') else str(last_result)
                            logger.warning(f"Tool FAIL: {action}({sym}) -> {str(err)[:300]}")
                        cycle_actions.append(f"{action}({sym})")
                        break

                # Feed result back
                if last_result is not None:
                    if not last_result.success:
                        self._consecutive_failures += 1
                        self._cycle_failures += 1
                    else:
                        self._consecutive_failures = 0

                    # Circuit breaker
                    if self._consecutive_failures >= self._max_consecutive:
                        logger.warning(f"Circuit breaker: {self._consecutive_failures} consecutive failures")
                        return 60
                    if self._cycle_failures >= self._max_per_cycle:
                        logger.warning(f"Circuit breaker: {self._cycle_failures} failures this cycle")
                        return 60

                    # Track order types used (for aggressive mode diversity)
                    if action in ALL_ORDER_TYPES:
                        self._used_order_types.add(action)

                    # Refresh state from broker
                    state_text = await self._build_state_context()

                    chat.append(response)
                    chat.append(sdk_user(
                        f"TOOL RESULT:\n{last_result}\n\n"
                        f"{state_text}\n\nWhat's your next action?"
                    ))

            except Exception as e:
                logger.error(f"Turn error: {e}", exc_info=True)
                return 60


# ── Entry Point ─────────────────────────────────────────────────

async def run_agent():
    """Main entry point for the autonomous trading agent."""
    logger.info("=" * 60)
    logger.info("GROK 4.20 TRADER — Pure ReAct")
    logger.info(f"Risk per trade: {RISK_PER_TRADE*100:.1f}%  |  Min R:R: {MIN_RR_RATIO}:1")
    logger.info(f"Trading mode: {TRADING_MODE}")
    if PAPER_AGGRESSIVE:
        logger.info(">>> AGGRESSIVE PAPER MODE — stress testing complex orders <<<")
    logger.info("=" * 60)

    # Connect broker
    try:
        gateway = await create_gateway({})
    except BrokerConfigError as e:
        logger.error(f"Broker config error: {e}")
        return
    except ConnectionError as e:
        logger.error(f"Broker connection failed: {e}")
        return

    # Build tools and agent
    data_provider = get_data_provider()
    market_hours = get_market_hours_provider()

    tools = ToolExecutor(
        gateway,
        data_provider,
        market_hours_provider=market_hours,
        cost_tracker=get_cost_tracker(),
    )
    agent = TradingAgent(gateway, tools)

    logger.info("Agent started (paper mode recommended)")
    agent._capture_start_of_day_cash()

    # ── Main loop ───────────────────────────────────────────────
    cycle = 0
    while True:
        if agent._halted:
            logger.critical("AGENT HALTED — waiting for market close")
            await asyncio.sleep(300)
            try:
                mh = get_market_hours_provider()
                info = mh.get_session_info()
                if info.get("session") == "closed":
                    break
            except Exception:
                pass
            continue

        if agent._start_of_day_cash is None:
            agent._capture_start_of_day_cash()

        # No auto-shutdown — Grok decides hold time (overnight OK)
        # Only halt if agent is explicitly halted (daily loss, LLM cost)
        try:
            mh = get_market_hours_provider()
            info = mh.get_session_info()
            session = info.get("session", "")
            # Log session state but DO NOT force shutdown
            if session == "closed":
                now_et = _now_et()
                if now_et.hour >= 20:  # After 8 PM ET, go to sleep mode
                    logger.info("Market closed, extended hours over — sleeping 30 min")
                    await asyncio.sleep(1800)
                    continue
        except Exception:
            pass

        cycle += 1
        logger.info(f"{'='*40} CYCLE {cycle} {'='*40}")

        try:
            wait_seconds = await agent.run_cycle()
            logger.info(f"Cooldown: {wait_seconds}s")
            await asyncio.sleep(wait_seconds)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            await asyncio.sleep(30)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for lib in ("httpx", "httpcore", "grpc", "xai_sdk",
                "ib_insync.wrapper",
                "ib_insync.ib", "ib_insync.client", "ib_insync.decoder",
                "ib_insync.connection", "ib_insync.flexreport", "ib_insync.order",
                "asyncio", "nest_asyncio"):        logging.getLogger(lib).setLevel(logging.WARNING)
    print("Grok 4.20 Trader started (paper mode recommended)")
    asyncio.run(run_agent())
