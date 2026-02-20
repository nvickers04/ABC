"""
Minimal ReAct Agent Loop — THE ONLY BRAIN

Architecture: Grok drives the loop. We provide tools and context.
Grok chains actions until it signals 'done' to refresh context.
No orchestration, no registries, no rigid strategies.
Grok reasons freely within iron-clad risk parameters.

DESIGN PHILOSOPHY:
- Pure model autonomy — Grok decides what to research, when to trade, when to wait
- Dynamic liquidity — Grok filters for liquid names, decides hold time (overnight OK)
- Thin tools — wrappers around broker + market data, nothing more
- Configurable risk — RISK_PER_TRADE from .env, enforced in code + prompt
- Paranoid defaults — temperature=0.0, seed=42, WAIT is the default outcome
- No automatic EOD close — Grok holds winners if edge remains
"""

import ast
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import (
    SYSTEM_PROMPT,
    RISK_PER_TRADE,
    CYCLE_SLEEP_SECONDS,
    MAX_TURNS_PER_CYCLE,
    FINAL_DECISION_NUDGE_TURN,
    MAX_DAILY_LOSS_PCT,
    MAX_DAILY_LLM_COST,
    LLM_TEMPERATURE,
    LLM_SEED,
    LLM_MAX_TOKENS,
    PAPER_AGGRESSIVE,
    MIN_RR_RATIO,
    MIN_CONFIDENCE_PCT,
)
from core.grok_llm import get_grok_llm
from data.cost_tracker import get_cost_tracker
from data.data_provider import get_data_provider
from data.broker_gateway import create_gateway, BrokerConfigError
from data.market_hours import get_market_hours_provider
from tools.tools_executor import ToolExecutor, get_valid_actions

logger = logging.getLogger(__name__)


# ── JSON Parsing Helpers ────────────────────────────────────────

def _try_repair_json(raw: str) -> list[dict[str, Any]]:
    """Best-effort repair for almost-valid model JSON output."""
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


# ── Confidence Validation ───────────────────────────────────────

CONFIDENCE_BANDS = {"low", "medium", "high"}


def _validate_confidence(action_data: dict) -> tuple[bool, str]:
    """Validate confidence metadata on every action."""
    conf = action_data.get("confidence")
    if not isinstance(conf, dict):
        return False, "Missing required object: confidence"
    band = str(conf.get("band", "")).strip().lower()
    if band not in CONFIDENCE_BANDS:
        return False, "confidence.band must be one of: low, medium, high"
    if not str(conf.get("why", "")).strip():
        return False, "confidence.why is required"
    evidence = conf.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not evidence:
        return False, "confidence.evidence must contain at least one item"
    if "unknowns" not in conf:
        return False, "confidence.unknowns is required (can be empty list)"
    return True, ""


def _extract_confidence(action_data: dict) -> dict:
    """Extract normalized confidence metadata."""
    conf = action_data.get("confidence")
    if not isinstance(conf, dict):
        return {}
    band = str(conf.get("band", "")).strip().lower()
    why = str(conf.get("why", "")).strip()
    evidence = conf.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    unknowns = conf.get("unknowns", [])
    if isinstance(unknowns, str):
        unknowns = [unknowns]
    return {"band": band, "why": why, "evidence": evidence, "unknowns": unknowns}


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

        logger.info("TradingAgent initialized (minimal ReAct mode)")

    def _append_snapshot(self, summary: str):
        """Append a cycle snapshot, keeping last 5."""
        self._market_snapshots.append(f"[{datetime.now().strftime('%H:%M')}] {summary}")
        if len(self._market_snapshots) > 5:
            self._market_snapshots = self._market_snapshots[-5:]

    async def _get_cash(self) -> float:
        """Get current cash balance from broker."""
        try:
            summary = await self.gateway.get_account_summary()
            return summary.get("totalcashvalue", 0) or summary.get("availablefunds", 0) or 0
        except Exception:
            return self.gateway.cash_value if self.gateway else 0

    def _capture_start_of_day_cash(self):
        """Capture cash at session start for daily loss tracking."""
        if self._start_of_day_cash is not None:
            return
        cash = self.gateway.cash_value if self.gateway else 0
        if cash > 0:
            self._start_of_day_cash = cash
            logger.info(
                f"Start-of-day Cash: ${cash:,.2f} "
                f"(loss limit: -{MAX_DAILY_LOSS_PCT}% = ${cash * MAX_DAILY_LOSS_PCT / 100:,.2f})"
            )

    def _check_daily_loss(self) -> Optional[float]:
        """Check if daily loss limit breached. Returns loss % if breached."""
        if not self._start_of_day_cash or self._start_of_day_cash <= 0:
            return None
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

    async def _build_state_context(self) -> str:
        """Query broker directly and build a state context string for the agent."""
        lines = ["═══ ACCOUNT STATE ═══"]
        try:
            summary = await self.gateway.get_account_summary()
            cash = summary.get("totalcashvalue", 0)
            avail = summary.get("availablefunds", 0)
            net_liq = summary.get("netliquidation", 0)
            daily_pnl = summary.get("dailypnl", "N/A")
            unreal = summary.get("unrealizedpnl", "N/A")
            real = summary.get("realizedpnl", "N/A")
            lines.append(f"Cash: ${cash:,.2f}  |  Available: ${avail:,.2f}  |  NetLiq: ${net_liq:,.2f}")
            lines.append(f"Daily P&L: {daily_pnl}  |  Unrealized: {unreal}  |  Realized: {real}")
        except Exception as e:
            lines.append(f"Account query error: {e}")

        lines.append("")
        lines.append("═══ POSITIONS ═══")
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
                    if sec == "OPT":
                        strike = p.get("strike", "")
                        right = p.get("right", "")
                        exp = p.get("expiration", "")
                        lines.append(
                            f"  {sym} {right}{strike} {exp}: {qty} @ ${avg:.2f} "
                            f"| mkt ${mkt:.2f} | P&L ${pnl:+,.2f}"
                        )
                    else:
                        lines.append(
                            f"  {sym}: {qty} shares @ ${avg:.2f} "
                            f"| mkt ${mkt:.2f} | P&L ${pnl:+,.2f}"
                        )
        except Exception as e:
            lines.append(f"Position query error: {e}")

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
                    price_str = ""
                    if lmt:
                        price_str = f"lmt ${lmt:.2f}"
                    if aux:
                        price_str += f" aux ${aux:.2f}"
                    lines.append(f"  #{oid} {action} {qty} {sym} {otype} {price_str}")
        except Exception as e:
            lines.append(f"Order query error: {e}")

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

        aggressive_nudge = ""
        if PAPER_AGGRESSIVE:
            aggressive_nudge = (
                "\nAGGRESSIVE TEST MODE: Evaluate at least 3-5 setups from market_scan. "
                "Try a complex options strategy this cycle (spread, iron condor, calendar, straddle). "
                "Test different order types (bracket, trailing stop, adaptive). Break things safely."
            )

        context = f"""{state_text}
{cost_line}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{continuity}{aggressive_nudge}
You have account state above. Call market_scan() then find a trade. What is your next action?"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        # ── Inner ReAct loop ────────────────────────────────────
        turn = 0
        consecutive_thinks = 0
        cycle_actions = []

        while True:
            turn += 1

            if turn > MAX_TURNS_PER_CYCLE:
                logger.warning(f"Turn limit ({MAX_TURNS_PER_CYCLE}) reached — forcing WAIT")
                self._last_cycle_summary = f"Cycle {self._cycle_id}: WAIT (turn limit)"
                self._append_snapshot(f"C{self._cycle_id}: WAIT (limit)")
                return CYCLE_SLEEP_SECONDS

            # Nudge for FINAL_DECISION when approaching turn limit
            if turn == FINAL_DECISION_NUDGE_TURN:
                logger.info(f"Turn {turn}: nudging for FINAL_DECISION")
                messages.append({
                    "role": "user",
                    "content": (
                        f"TURN {turn}/{MAX_TURNS_PER_CYCLE} — you are running out of turns. "
                        "Issue your FINAL_DECISION NOW as JSON: "
                        '{"action": "FINAL_DECISION", "decision": "WAIT"|"TRADE", ...}'
                    ),
                })

            # Re-check daily loss every turn
            loss_pct = self._check_daily_loss()
            if loss_pct is not None:
                await self._emergency_flatten(f"Daily loss mid-cycle: -{loss_pct:.1f}%")
                return 999999

            logger.info(f"Cycle {self._cycle_id} turn {turn}")

            try:
                response = await self.grok.client.chat.completions.create(
                    model=self.grok.model,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    seed=LLM_SEED,
                    timeout=60.0,
                )

                raw = response.choices[0].message.content

                # Log cost
                usage = response.usage
                self.cost_tracker.log_llm_call(
                    model=self.grok.model,
                    tokens_in=usage.prompt_tokens,
                    tokens_out=usage.completion_tokens,
                    purpose="agent_loop",
                )
                cost_summary = self.cost_tracker.get_budget_summary()
                logger.info(f"LLM {usage.prompt_tokens}+{usage.completion_tokens} tok | ${cost_summary.today_llm_cost:.4f} today")

                # ── Check for FINAL_DECISION in raw text ────────
                if "FINAL_DECISION: WAIT" in (raw or "") or '"decision": "WAIT"' in (raw or "").replace("'", '"'):
                    reason = raw.split("WAIT")[-1].strip(" |\n\r")[:200] if "FINAL_DECISION: WAIT" in (raw or "") else ""
                    logger.info(f"Decision: WAIT | {reason}")
                    messages.append({"role": "assistant", "content": raw})
                    self._last_cycle_summary = f"Cycle {self._cycle_id}: WAIT — {reason[:80]}"
                    self._append_snapshot(f"C{self._cycle_id}: WAIT")
                    return CYCLE_SLEEP_SECONDS

                if "FINAL_DECISION: TRADE" in (raw or "") or '"decision": "TRADE"' in (raw or "").replace("'", '"'):
                    logger.info(f"Decision: TRADE | {raw[-300:]}")
                    messages.append({"role": "assistant", "content": raw})
                    # Don't return — tell Grok to execute the trade NOW
                    messages.append({
                        "role": "user",
                        "content": (
                            "TRADE APPROVED. Now EXECUTE it. Use bracket_order, market_order, "
                            "limit_order, or an options strategy tool to place the actual order. "
                            "Do NOT just signal — place the trade now."
                        ),
                    })
                    continue

                # ── Parse JSON actions ──────────────────────────
                json_objects = _parse_json_objects(raw or "")

                if not json_objects:
                    logger.warning(f"No valid JSON: {raw[:100]}")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": "Please respond with valid JSON for your next action.",
                    })
                    continue

                # Process actions — one real action per response
                last_result = None

                for action_data in json_objects:
                    action = action_data.get("action", "")

                    # Handle new structured FINAL_DECISION action
                    if action == "FINAL_DECISION" or action == "final_decision":
                        decision = action_data.get("decision", "WAIT").upper()
                        reason = action_data.get("reason", "")[:200]
                        if decision == "TRADE":
                            ticker = action_data.get("ticker", "?")
                            size = action_data.get("size", 0)
                            stop = action_data.get("stop", 0)
                            target = action_data.get("target", 0)
                            tactic = action_data.get("tactic", "")
                            logger.info(f"Decision: TRADE {ticker} x{size} stop={stop} tgt={target} | {tactic}")
                            messages.append({"role": "assistant", "content": raw})
                            # Don't return — tell Grok to execute NOW
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"TRADE APPROVED for {ticker}. Now EXECUTE it immediately. "
                                    f"Use bracket_order (preferred — sets entry + stop + target), "
                                    f"market_order, limit_order, or an options strategy tool. "
                                    f"Your plan: {size} shares, stop ${stop}, target ${target}. "
                                    f"Place the order NOW — do not re-analyze."
                                ),
                            })
                            self._last_cycle_summary = f"Cycle {self._cycle_id}: TRADE {ticker}"
                            self._append_snapshot(f"C{self._cycle_id}: TRADE {ticker}")
                            continue
                        else:
                            logger.info(f"Decision: WAIT | {reason}")
                            messages.append({"role": "assistant", "content": raw})
                            self._last_cycle_summary = f"Cycle {self._cycle_id}: WAIT — {reason[:80]}"
                            self._append_snapshot(f"C{self._cycle_id}: WAIT")
                            return CYCLE_SLEEP_SECONDS

                    # Validate confidence
                    conf_ok, conf_err = _validate_confidence(action_data)
                    if not conf_ok:
                        logger.warning(f"Confidence invalid: {conf_err}")
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({
                            "role": "user",
                            "content": json.dumps({
                                "system_notice": "RESPONSE REJECTED: confidence metadata missing/invalid.",
                                "error": conf_err,
                                "required_schema": {
                                    "action": "<tool>",
                                    "confidence": {
                                        "band": "low|medium|high",
                                        "why": "short rationale",
                                        "evidence": ["signals"],
                                        "unknowns": ["uncertainties"],
                                    },
                                },
                                "valid_actions": get_valid_actions()[:25],
                            }, indent=2),
                        })
                        last_result = None
                        break

                    # ── Meta-actions ────────────────────────────
                    if action == "wait":
                        logger.info(f"Decision: WAIT | {action_data.get('reasoning', '')[:200]}")
                        messages.append({"role": "assistant", "content": raw})
                        self._last_cycle_summary = f"Cycle {self._cycle_id}: WAIT"
                        self._append_snapshot(f"C{self._cycle_id}: WAIT")
                        return CYCLE_SLEEP_SECONDS

                    elif action == "think":
                        thought = action_data.get("thought", "")
                        logger.debug(f"Think: {thought[:200]}")
                        consecutive_thinks += 1
                        messages.append({"role": "assistant", "content": raw})
                        if consecutive_thinks >= 3:
                            logger.info(f"Think-loop nudge after {consecutive_thinks} thinks")
                            messages.append({
                                "role": "user",
                                "content": "You have been thinking for multiple turns without new data. "
                                           "Issue your FINAL_DECISION now as JSON: "
                                           '{"action": "FINAL_DECISION", "decision": "WAIT"|"TRADE", ...}'
                            })
                        continue

                    elif action == "feedback":
                        issue = action_data.get("issue", "")
                        logger.warning(f"FEEDBACK: {issue}")
                        continue

                    elif action == "done":
                        reasoning = action_data.get("reasoning", "")
                        logger.info(f"Decision: DONE | {reasoning[:200]}")
                        cycle_actions.append("done")
                        self._last_cycle_summary = (
                            f"Cycle {self._cycle_id}: {len(cycle_actions)} actions — "
                            + ", ".join(cycle_actions[-5:])
                        )
                        self._append_snapshot(f"C{self._cycle_id}: DONE")
                        return CYCLE_SLEEP_SECONDS

                    else:
                        # ── Execute ONE real tool action ────────
                        consecutive_thinks = 0
                        last_result = await self.tools.execute(action, action_data)
                        sym = action_data.get('symbol', '')
                        logger.info(f"Tool: {action}({sym}) -> {str(last_result)[:150]}")
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

                    # Refresh state from broker
                    state_text = await self._build_state_context()

                    conf = _extract_confidence(action_data)
                    conf_note = ""
                    if conf:
                        conf_note = f"\nCONFIDENCE: {conf['band'].upper()} — {conf['why']}"

                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"TOOL RESULT:\n{last_result}\n{conf_note}\n\n"
                            f"{state_text}\n\nWhat's your next action?"
                        ),
                    })

            except Exception as e:
                logger.error(f"Turn error: {e}", exc_info=True)
                return 60


# ── Entry Point ─────────────────────────────────────────────────

async def run_agent():
    """Main entry point for the autonomous trading agent."""
    logger.info("=" * 60)
    logger.info("GROK 4.20 TRADER — Pure ReAct")
    logger.info(f"Risk per trade: {RISK_PER_TRADE*100:.1f}%  |  Min R:R: {MIN_RR_RATIO}:1  |  Min conf: {MIN_CONFIDENCE_PCT}%")
    if PAPER_AGGRESSIVE:
        logger.info(">>> SUPER AGGRESSIVE PAPER MODE ACTIVE <<<")
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
            cooldown = min(wait_seconds, CYCLE_SLEEP_SECONDS) if wait_seconds < 999999 else wait_seconds
            logger.info(f"Cooldown: {cooldown}s")
            await asyncio.sleep(cooldown)

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
    for lib in ("httpx", "httpcore", "openai", "ib_insync.wrapper",
                "ib_insync.ib", "ib_insync.client"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    print("Grok 4.20 Trader started (paper mode recommended)")
    asyncio.run(run_agent())
