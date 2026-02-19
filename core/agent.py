"""
Minimal ReAct Agent Loop — THE ONLY BRAIN

Built for Grok 4.2 — Alpha Arena winning style (pure autonomy, max WAIT, 0.5% risk)

Architecture: Grok drives the loop. We provide tools and context.
Grok chains actions until it signals 'done' to refresh context.
No orchestration, no registries, no rigid strategies.
Grok reasons freely within iron-clad risk parameters.

DESIGN PHILOSOPHY:
- Pure model autonomy — Grok decides what to research, when to trade, when to wait
- Dynamic liquidity — Grok filters for liquid names, decides hold time (overnight OK)
- Thin tools — wrappers around broker + market data, nothing more
- Iron-clad risk — 0.5% max risk per trade enforced in code, not just prompt
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

import yaml

from core.config import (
    SYSTEM_PROMPT,
    CYCLE_SLEEP_SECONDS,
    MAX_TURNS_PER_CYCLE,
    MAX_DAILY_LOSS_PCT,
    MAX_DAILY_LLM_COST,
    LLM_TEMPERATURE,
    LLM_SEED,
    LLM_MAX_TOKENS,
)
from core.grok_llm import get_grok_llm
from data.cost_tracker import get_cost_tracker
from data.live_state import get_live_state
from data.data_provider import get_data_provider
from data.broker_gateway import create_gateway, BrokerConfigError
from data.market_hours import get_market_hours_provider
from tools.tools_executor import ToolExecutor, get_valid_actions

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config/trading.yaml")


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


def load_trading_config() -> dict:
    """Load trading config from YAML with defaults."""
    defaults = {
        "agent": {
            "temperature": {"default": 0.0, "trading": 0.0, "research": 0.0},
            "circuit_breaker": {
                "max_consecutive_failures": 5,
                "max_failures_per_cycle": 8,
                "recovery_seconds": 60,
            },
        }
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                loaded = yaml.safe_load(f) or {}
            for section in defaults:
                if section in loaded and isinstance(loaded[section], dict):
                    for key in loaded[section]:
                        defaults[section][key] = loaded[section][key]
            return defaults
        except Exception as e:
            logger.warning(f"Failed to load {CONFIG_FILE}: {e}, using defaults")
    return defaults


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

    def __init__(self, gateway, tools: ToolExecutor, live_state=None, grok=None):
        self.grok = grok or get_grok_llm()
        self.gateway = gateway
        self.cost_tracker = get_cost_tracker()
        self.tools = tools
        self.live_state = live_state or get_live_state()

        # Circuit breaker
        self._consecutive_failures = 0
        self._cycle_failures = 0
        self._max_consecutive = 5
        self._max_per_cycle = 8

        # Session state
        self._cycle_id = 0
        self._last_cycle_summary = ""
        self._halted = False
        self._start_of_day_net_liq: Optional[float] = None

        logger.info("TradingAgent initialized (minimal ReAct mode)")

    def _capture_start_of_day_net_liq(self):
        """Capture net liquidation at session start for daily loss tracking."""
        if self._start_of_day_net_liq is not None:
            return
        net_liq = self.live_state._net_liq
        if net_liq > 0:
            self._start_of_day_net_liq = net_liq
            logger.info(
                f"Start-of-day Net Liq: ${net_liq:,.2f} "
                f"(loss limit: -{MAX_DAILY_LOSS_PCT}% = ${net_liq * MAX_DAILY_LOSS_PCT / 100:,.2f})"
            )

    def _check_daily_loss(self) -> Optional[float]:
        """Check if daily loss limit breached. Returns loss % if breached."""
        if not self._start_of_day_net_liq or self._start_of_day_net_liq <= 0:
            return None
        current = self.live_state._net_liq
        if current <= 0:
            return None
        loss_pct = (self._start_of_day_net_liq - current) / self._start_of_day_net_liq * 100
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

    async def run_cycle(self) -> int:
        """
        Run one cycle of the ReAct loop.

        Returns cooldown seconds.
        """
        self._cycle_failures = 0
        self._cycle_id += 1

        # Refresh option greeks
        try:
            await self.live_state.refresh_option_greeks()
        except Exception as e:
            logger.debug(f"Greek refresh skipped: {e}")

        # Get live state
        live_state_text = self.live_state.format_for_agent()

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

        context = f"""{live_state_text}
{cost_line}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{continuity}
Be extremely conservative. Assess positions and research opportunities. What is your next action?"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        # ── Inner ReAct loop ────────────────────────────────────
        turn = 0
        cycle_actions = []

        while True:
            turn += 1

            if turn > MAX_TURNS_PER_CYCLE:
                logger.warning(f"Turn limit ({MAX_TURNS_PER_CYCLE}) reached")
                return 0

            # Re-check daily loss every turn
            loss_pct = self._check_daily_loss()
            if loss_pct is not None:
                await self._emergency_flatten(f"Daily loss mid-cycle: -{loss_pct:.1f}%")
                return 999999

            logger.info(f"Cycle {self._cycle_id} turn {turn}...")

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
                logger.info(f"Grok: {raw[:300]}...")

                # Log cost
                usage = response.usage
                self.cost_tracker.log_llm_call(
                    model=self.grok.model,
                    tokens_in=usage.prompt_tokens,
                    tokens_out=usage.completion_tokens,
                    purpose="agent_loop",
                )

                # ── Check for FINAL_DECISION in raw text ────────
                if "FINAL_DECISION: WAIT" in (raw or ""):
                    logger.info(f"[WAIT] {raw[-300:]}")
                    messages.append({"role": "assistant", "content": raw})
                    self._last_cycle_summary = f"Cycle {self._cycle_id}: WAIT"
                    return CYCLE_SLEEP_SECONDS

                if "FINAL_DECISION: TRADE" in (raw or ""):
                    logger.info(f"[TRADE SIGNAL] {raw}")
                    # The trade details are in the FINAL_DECISION line —
                    # but actual execution happens via tool calls within the cycle.
                    # If we reach here, log it and let the next cycle handle execution.
                    messages.append({"role": "assistant", "content": raw})
                    self._last_cycle_summary = f"Cycle {self._cycle_id}: TRADE SIGNAL"
                    return 15  # Quick restart to execute

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
                        logger.info(f"Agent wants to WAIT: {action_data.get('reasoning', '')[:200]}")
                        # In this minimal version, WAIT is respected — the agent IS allowed to wait.
                        # Alpha Arena philosophy: doing nothing is often the best trade.
                        messages.append({"role": "assistant", "content": raw})
                        self._last_cycle_summary = f"Cycle {self._cycle_id}: WAIT"
                        return CYCLE_SLEEP_SECONDS

                    elif action == "think":
                        thought = action_data.get("thought", "")
                        logger.info(f"Thinking: {thought[:200]}")
                        continue

                    elif action == "feedback":
                        issue = action_data.get("issue", "")
                        logger.warning(f"FEEDBACK: {issue}")
                        continue

                    elif action == "done":
                        reasoning = action_data.get("reasoning", "")
                        logger.info(f"Cycle done: {reasoning[:300]}")
                        cycle_actions.append("done")
                        self._last_cycle_summary = (
                            f"Cycle {self._cycle_id}: {len(cycle_actions)} actions — "
                            + ", ".join(cycle_actions[-5:])
                        )
                        return CYCLE_SLEEP_SECONDS

                    else:
                        # ── Execute ONE real tool action ────────
                        last_result = await self.tools.execute(action, action_data)
                        logger.info(f"Tool result: {str(last_result)[:200]}")
                        cycle_actions.append(f"{action}({action_data.get('symbol', '')})")
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

                    # Refresh live state
                    live_state_text = self.live_state.format_for_agent()

                    conf = _extract_confidence(action_data)
                    conf_note = ""
                    if conf:
                        conf_note = f"\nCONFIDENCE: {conf['band'].upper()} — {conf['why']}"

                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"TOOL RESULT:\n{last_result}\n{conf_note}\n\n"
                            f"{live_state_text}\n\nWhat's your next action? Be conservative."
                        ),
                    })

            except Exception as e:
                logger.error(f"Turn error: {e}", exc_info=True)
                return 60


# ── Entry Point ─────────────────────────────────────────────────

async def run_agent():
    """Main entry point for the minimal Grok 4.2 trading agent."""
    logger.info("=" * 60)
    logger.info("MINIMAL GROK 4.2 TRADER — Pure ReAct (Alpha Arena style)")
    logger.info("=" * 60)

    # Load config
    raw_config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                raw_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return

    # Connect broker
    try:
        gateway = await create_gateway(raw_config)
    except BrokerConfigError as e:
        logger.error(f"Broker config error: {e}")
        return
    except ConnectionError as e:
        logger.error(f"Broker connection failed: {e}")
        return

    # Wire live state to broker events
    await gateway.wire_live_state()

    # Build tools and agent
    data_provider = get_data_provider()
    live_state = get_live_state()
    market_hours = get_market_hours_provider()

    tools = ToolExecutor(
        gateway,
        data_provider,
        live_state=live_state,
        market_hours_provider=market_hours,
    )
    agent = TradingAgent(gateway, tools, live_state=live_state)

    logger.info("Agent started (paper mode recommended)")
    agent._capture_start_of_day_net_liq()

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

        if agent._start_of_day_net_liq is None:
            agent._capture_start_of_day_net_liq()

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
        logger.info(f"\n{'='*40}\nCYCLE {cycle}\n{'='*40}")

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
    logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
    logging.getLogger("ib_insync.ib").setLevel(logging.WARNING)
    print("Minimal Grok 4.2 Trader started (paper mode recommended)")
    asyncio.run(run_agent())
