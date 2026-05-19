"""
Hybrid ReAct Agent Loop — Reasoning + Multi-Agent Research

Architecture:
  - REASONING model (single-agent) drives the ReAct loop with client-side tools
  - MULTI-AGENT model (4/16 agents) is exposed as a research() tool for web/X search
  - The reasoning model decides when to call research() — no separate coordination

Grok chains actions until it signals 'done' to refresh context.
research() is the primary discovery mechanism.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, cast

from xai_sdk.chat import system as sdk_system
from xai_sdk.chat import user as sdk_user

from core.async_utils import safe_sleep as _safe_sleep
from core.config import (
    AGENT_TOOL_FEEDBACK_MAX_CHARS,
    CYCLE_SLEEP_SECONDS,
    EOD_FLATTEN_MINUTES,
    INTRADAY_DRAWDOWN_PCT,
    LLM_MAX_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    MAX_DAILY_LLM_COST,
    MAX_DAILY_LOSS_PCT,
    MIN_RR_RATIO,
    PAPER_AGGRESSIVE,
    PRESCAN_PROMPT_EXPENSIVE_RESEARCH,
    RISK_PER_TRADE,
    SYSTEM_PROMPT,
    TOOL_PLAYBOOK_MAX_CHARS,
    TRADING_MODE,
)
from core.grok_llm import get_grok_llm
from core.json_parse import _parse_json_objects
from core.log_context import (
    bind_trader_cycle_context,
    get_logger,
)
from core.research_topics import _categorize_query
from core.runtime import SafetyController, StateContextBuilder
from core.runtime.cycle import evaluate_gap_guard
from core.runtime.operating_context import get_operating_context
from core.runtime.scheduler import CycleScheduler

# Re-exported for backward compatibility (these used to live in this module).
from core.runtime.state_context import _SECTOR_MAP, _sector_of  # noqa: F401
from data.broker_gateway import BrokerConfigError, create_gateway
from data.cost_tracker import get_cost_tracker
from data.data_provider import get_data_provider
from data.market_hours import get_market_hours_provider
from tools.tools_executor import ToolExecutor

logger = get_logger(__name__)


def _truncate_for_react_context(text: str, max_chars: int) -> str:
    """Bound tool payloads re-appended each ReAct turn (full history is re-sent)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n... [truncated for prompt budget]\n"
    room = max_chars - len(marker)
    if room < 200:
        return text[:max_chars]
    return text[:room] + marker


# Note: _now_et is now defined in core.runtime.scheduler and re-imported above
# for backward compatibility with any external callers.


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

        # Runtime modules (extracted from this class).
        self._state_context_builder = StateContextBuilder(gateway)
        self._operating_context = get_operating_context()

        # Best-effort detection of researcher availability at startup
        if not self._operating_context.sync_researcher_from_heartbeat():
            self._operating_context.quality.working_memory_completeness = 0.5
            logger.warning(
                "Starting in INDEPENDENT MODE — researcher unavailable. "
                "Using local memory fallback and reduced risk."
            )

        try:
            from core.quality.quality_matrix import get_quality_matrix_service
            svc = get_quality_matrix_service()
            svc.populate()
            logger.info(
                "QualityMatrix initial populate at startup: overall=%s rm=%.2f",
                svc.get_matrix().overall_quality,
                svc.get_matrix().risk_multiplier,
            )
        except Exception as _init_qm:
            logger.debug("QualityMatrix startup populate skipped (non-fatal): %s", _init_qm)

        self._safety = SafetyController(
            gateway,
            self.cost_tracker,
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
            intraday_drawdown_pct=INTRADAY_DRAWDOWN_PCT,
            max_daily_llm_cost=MAX_DAILY_LLM_COST,
        )

        # Circuit breaker
        self._consecutive_failures = 0
        self._cycle_failures = 0
        self._max_consecutive = 5
        self._max_per_cycle = 8

        # Session state
        self._cycle_id = 0
        self._last_step = "init"
        self._last_step_ts = time.time()
        self._last_cycle_summary = ""
        self._last_wait_reason = ""    # why the agent chose to wait (set in done)
        self._last_wake_reason = ""    # what woke it (scorer_round_N / timeout / ...)
        self._halted = False
        self._gap_guard_until: Optional[datetime] = None   # Delay entries after large gap
        self._eod_flattened_date: Optional[str] = None     # Track EOD flatten per day
        self._market_snapshots: list[str] = []  # Rolling 5-cycle summaries
        self._research_results: dict[str, dict] = {}  # query -> {summary, ts, session, category, ttl, created}
        self._current_session: str = "unknown"
        self._daily_review_date: Optional[str] = None  # Track which day we've reviewed
        self._pre_scan_date: Optional[str] = None  # Track which day we've pre-scanned

        logger.info("TradingAgent initialized (minimal ReAct mode)")

    # ── Backward-compat shims for fields moved into SafetyController ──
    @property
    def _start_of_day_cash(self) -> Optional[float]:
        return self._safety.start_of_day_cash

    @_start_of_day_cash.setter
    def _start_of_day_cash(self, value: Optional[float]) -> None:
        self._safety.start_of_day_cash = value

    @property
    def _session_high_water(self) -> Optional[float]:
        return self._safety.session_high_water

    @_session_high_water.setter
    def _session_high_water(self, value: Optional[float]) -> None:
        self._safety.session_high_water = value

    def _append_snapshot(self, summary: str):
        """Append a compact cycle snapshot for rolling continuity."""
        from core.runtime.prompt_budget import CYCLE_SNAPSHOT_CHARS, CYCLE_SNAPSHOT_MAX

        short = (summary or "")[:CYCLE_SNAPSHOT_CHARS]
        self._market_snapshots.append(f"C{self._cycle_id}:{short}")
        if len(self._market_snapshots) > CYCLE_SNAPSHOT_MAX:
            self._market_snapshots = self._market_snapshots[-CYCLE_SNAPSHOT_MAX:]

    def _prune_research_cache(self, current_session: str):
        """Remove expired research entries by TTL and session boundary."""
        import time
        now = time.time()
        to_delete = []
        for query, entry in self._research_results.items():
            ttl = entry.get("ttl", 0)
            created = entry.get("created", 0)
            entry_session = entry.get("session", "unknown")
            # TTL-based expiry (ticker=30m, sector=4h)
            if ttl > 0 and (now - created) > ttl:
                to_delete.append(query)
            # Session boundary expiry (macro entries expire on session change)
            elif ttl == 0 and entry_session != current_session and entry_session != "unknown":
                to_delete.append(query)
        for q in to_delete:
            del self._research_results[q]
        if to_delete:
            logger.debug(f"Pruned {len(to_delete)} stale research entries")

    async def _get_cash(self) -> float:
        """Get current cash balance from broker (TotalCashValue ONLY, never AvailableFunds)."""
        try:
            summary = await self.gateway.get_account_summary()
            return summary.get("totalcashvalue", 0) or 0
        except Exception:
            return self.gateway.cash_value if self.gateway else 0

    def _capture_start_of_day_cash(self):
        """Delegate to :meth:`SafetyController.capture_start_of_day_cash`."""
        self._safety.capture_start_of_day_cash()

    def _check_daily_loss(self) -> Optional[float]:
        """Delegate to :meth:`SafetyController.check_daily_loss`."""
        return self._safety.check_daily_loss()

    def _check_intraday_drawdown(self) -> Optional[float]:
        """Delegate to :meth:`SafetyController.check_intraday_drawdown`."""
        return self._safety.check_intraday_drawdown()

    async def _emergency_flatten(self, reason: str):
        """EMERGENCY: Flatten all positions and halt for the day."""
        logger.critical("=" * 60)
        logger.critical(f"EMERGENCY FLATTEN: {reason}")
        logger.critical("=" * 60)
        if self.gateway:
            for attempt in range(3):
                result = await self.gateway.flatten_all()
                errors = result.get("errors", [])
                closed = result.get("positions_closed", 0)
                total = result.get("positions_total", 0)
                logger.critical(
                    f"Flatten attempt {attempt+1}: {closed}/{total} positions closed, "
                    f"{len(errors)} errors"
                )
                if not errors or closed >= total:
                    break
                logger.critical(f"Flatten errors: {errors} — retrying in 2s")
                await _safe_sleep(2)
            else:
                logger.critical("FLATTEN INCOMPLETE after 3 attempts — positions may remain open!")
        self._halted = True

    def _check_llm_cost(self) -> bool:
        """Delegate to :meth:`SafetyController.check_llm_cost`."""
        return self._safety.check_llm_cost()

    def _check_llm_token_limits(self) -> Optional[str]:
        """Return a reason string if any daily token bucket is exhausted."""
        checker = getattr(self.cost_tracker, "check_daily_token_limits", None)
        if checker is not None and callable(checker):
            return cast(Callable[[], Optional[str]], checker)()  # pylint: disable=not-callable
        return None

    def record_trade(self, symbol: str, side: str, pnl: float, held_minutes: int = 0):
        """Record a closed trade into the research memory DB."""
        from memory import record_trade
        record_trade(symbol, side, pnl, held_minutes)

    def _emit_hypothesis(
        self,
        hypothesis_type: str,
        description: str,
        suggested_action: str,
        priority: int = 5,
        related_slot: int | None = None,
        related_trade_id: int | None = None,
    ) -> None:
        """Write a self-improvement hypothesis to the trader_hypotheses table.

        Deduplicates by skipping if an identical (type, description) hypothesis
        was emitted within the last 6 hours.
        """
        try:
            from memory import get_db
            db = get_db()
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
            dup = db.execute(
                """SELECT 1 FROM trader_hypotheses
                   WHERE hypothesis_type = ? AND description = ? AND ts > ?
                   LIMIT 1""",
                (hypothesis_type, description, cutoff),
            ).fetchone()
            if dup:
                return
            db.execute(
                """INSERT INTO trader_hypotheses
                   (ts, source, hypothesis_type, description, suggested_action,
                    priority, status, related_slot, related_trade_id, env_key, signed_fitness, kept)
                   VALUES (?, 'trader_agent', ?, ?, ?, ?, 'open', ?, ?, NULL, NULL, 0)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    hypothesis_type,
                    description,
                    suggested_action,
                    priority,
                    related_slot,
                    related_trade_id,
                ),
            )
            db.commit()
            logger.debug(f"Hypothesis emitted: [{hypothesis_type}] {description[:80]}")
        except Exception as exc:
            logger.debug(f"Failed to emit hypothesis: {exc}")

    async def _run_daily_review(self) -> None:
        """End-of-day review: aggregate performance, run execution analysis if data threshold met.

        Delegates to :mod:`core.runtime.review`.
        """
        from core.runtime.review import run_daily_review
        await run_daily_review(self)

    async def _run_execution_analysis(self) -> None:
        """Delegates to :mod:`core.runtime.execution_analysis`."""
        from core.runtime.execution_analysis import run_execution_analysis
        await run_execution_analysis(self)

    def _test_proposal(self, proposal: dict, groups: dict) -> float | None:
        """Delegates to :mod:`core.runtime.execution_analysis`."""
        from core.runtime.execution_analysis import test_proposal
        return test_proposal(proposal, groups)

    def _evaluate_risk_ramp(self, db, today: str) -> None:
        """Check if live trading performance warrants risk ramp-up.

        Delegates to :mod:`core.runtime.review`.
        """
        from core.runtime.review import evaluate_risk_ramp
        evaluate_risk_ramp(db, today)

    async def _build_state_context(self) -> str:
        """Delegates to :class:`StateContextBuilder` (see ``tools._state_builder``)."""
        return await self._state_context_builder.build()

    async def run_cycle(self) -> int:
        """
        Run one cycle of the ReAct loop.

        Returns cooldown seconds.
        """
        self._cycle_failures = 0
        self._cycle_id += 1
        bind_trader_cycle_context(cycle_id=self._cycle_id)
        self._last_step = "detect_session"
        self._last_step_ts = time.time()
        logger.info("cycle_step", step="detect_session")

        # Detect current market session and prune stale research
        try:
            _mh = get_market_hours_provider()
            _info = _mh.get_session_info()
            self._current_session = _info.get("session", "unknown")
        except Exception:
            pass
        self._prune_research_cache(self._current_session)

        # Daily review: run once when session transitions to postmarket
        if self._current_session == "postmarket":
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_review_date != today:
                self._daily_review_date = today
                await self._run_daily_review()

        # Pre-scan: inject research prompt on first premarket/regular cycle of the day
        pre_scan_prompt = ""
        if self._current_session in ("premarket", "regular"):
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._pre_scan_date != today:
                self._pre_scan_date = today
                if PRESCAN_PROMPT_EXPENSIVE_RESEARCH:
                    pre_scan_prompt = (
                        "\n🔍 PRE-SCAN: New trading day. Before placing any trades, run research() "
                        "on today's macro outlook (e.g. 'market outlook today earnings economic data') "
                        "and check economic_calendar(). Review briefing(detail='environment') for regime context. "
                        "This primes your thesis with fresh data before committing capital.\n"
                    )
                else:
                    pre_scan_prompt = (
                        "\n🔍 PRE-SCAN: New trading day. Use cheap tools first: market_hours, "
                        "economic_calendar, briefing(), prior_research — not research() (multi-agent web/X) "
                        "unless cheaper sources are insufficient. research() is capped per day in config.\n"
                    )
                logger.info("Pre-scan prompt injected for new trading day")

        # Open volatility guard: detect overnight gap and delay entries
        gap_guard_prompt = await evaluate_gap_guard(self)

        # Build state context from broker
        self._last_step = "build_state_context"
        self._last_step_ts = time.time()
        logger.info("CYCLE %d step=build_state_context", self._cycle_id)
        state_text = await self._build_state_context()

        # Check daily loss
        loss_pct = self._check_daily_loss()
        if loss_pct is not None:
            await self._emergency_flatten(f"Daily loss: -{loss_pct:.1f}%")
            return 999999

        # Check intraday drawdown (peak-to-trough)
        dd_pct = self._check_intraday_drawdown()
        if dd_pct is not None:
            await self._emergency_flatten(
                f"Intraday drawdown: -{dd_pct:.1f}% from session high"
            )
            return 999999

        # EOD flatten: close all positions N minutes before market close
        if self._current_session == "regular":
            try:
                _info_eod = _mh.get_session_info()
                mins_to_close = _info_eod.get("minutes_to_close")
                today_eod = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if (mins_to_close is not None
                        and mins_to_close <= EOD_FLATTEN_MINUTES
                        and self._eod_flattened_date != today_eod):
                    self._eod_flattened_date = today_eod
                    await self._emergency_flatten(
                        f"EOD flatten: {mins_to_close} min to close"
                    )
                    # Don't halt — just flatten and continue (postmarket may follow)
                    self._halted = False
                    return CYCLE_SLEEP_SECONDS
            except Exception:
                pass

        # Check LLM cost
        if self._check_llm_cost():
            logger.warning(f"LLM cost ceiling ${MAX_DAILY_LLM_COST} reached — halting")
            self._halted = True
            return 999999

        tok_reason = self._check_llm_token_limits()
        if tok_reason:
            logger.warning("LLM token daily limit (%s) reached — halting", tok_reason)
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

        from zoneinfo import ZoneInfo

        from core.runtime.cycle_context import build_continuity_from_agent, build_cycle_user_context
        from core.runtime.prompt_budget import log_cycle_prompt_budget

        _et_now = datetime.now(ZoneInfo("America/New_York"))
        continuity = build_continuity_from_agent(self)
        if continuity:
            continuity = continuity if continuity.endswith("\n") else continuity + "\n"

        context, prompt_metrics = await build_cycle_user_context(
            operating_context=self._operating_context,
            state_text=state_text,
            cost_line=cost_line,
            continuity_text=continuity,
            pre_scan_prompt=pre_scan_prompt,
            gap_guard_prompt=gap_guard_prompt,
            et_now=_et_now,
        )
        self._cycle_prompt_metrics = prompt_metrics
        # ── Build xAI SDK chat instance ─────────────────────────
        self._last_step = "chat_create"
        self._last_step_ts = time.time()
        logger.info("CYCLE %d step=chat_create", self._cycle_id)
        _system_content = SYSTEM_PROMPT
        try:
            from tools.tool_playbook import render_compact_playbook

            _system_content = SYSTEM_PROMPT + "\n\n" + render_compact_playbook(TOOL_PLAYBOOK_MAX_CHARS)
        except Exception as e:
            logger.warning("tool playbook append failed: %s", e)
        prompt_metrics.system_chars = len(_system_content)
        prompt_metrics.est_system_tokens = max(
            1, int((len(_system_content) + 3) / 4)
        )
        prompt_metrics.record_user_total()
        log_cycle_prompt_budget(self._cycle_id, prompt_metrics)
        logger.debug(
            "CYCLE %d prompt caps: playbook_max=%d tool_feedback_max=%d wm_max via env CYCLE_WM_MAX_CHARS",
            self._cycle_id,
            TOOL_PLAYBOOK_MAX_CHARS,
            AGENT_TOOL_FEEDBACK_MAX_CHARS,
        )
        # QualityMatrix controls temperature/max_tokens (host override).
        try:
            from core.quality.quality_matrix import get_quality_matrix_service
            svc = get_quality_matrix_service()
            m = svc.get_matrix()
            cfg = m.get_llm_call_config()
            eff_temp = float(cfg.get("temperature", LLM_TEMPERATURE))
            eff_max_tokens = int(cfg.get("max_tokens", LLM_MAX_TOKENS))
            q_for_log = m.overall_quality
        except Exception:
            ctx = self._operating_context
            model_over = ctx.get_model_overrides()
            eff_temp = float(model_over.get("temperature", LLM_TEMPERATURE))
            eff_max_tokens = int(model_over.get("max_tokens", LLM_MAX_TOKENS))
            q_for_log = getattr(ctx.quality_matrix, "overall_quality", "unknown")

        logger.info(
            "CYCLE %d model-config from QualityMatrix: temp=%.2f max_tokens=%d (quality=%s)",
            self._cycle_id, eff_temp, eff_max_tokens, q_for_log
        )
        chat = self.grok.client.chat.create(
            model=self.grok.model,
            messages=[
                sdk_system(_system_content),
                sdk_user(context),
            ],
            temperature=eff_temp,
            max_tokens=eff_max_tokens,
            seed=LLM_SEED,
        )

        # ── Inner ReAct loop ────────────────────────────────────
        turn = 0

        cycle_actions = []
        invalid_json_streak = 0

        while True:
            turn += 1

            # Re-check daily loss every turn
            loss_pct = self._check_daily_loss()
            if loss_pct is not None:
                await self._emergency_flatten(f"Daily loss mid-cycle: -{loss_pct:.1f}%")
                return 999999

            logger.info(f"Cycle {self._cycle_id} turn {turn}")

            try:
                # Retry Grok API with exponential backoff
                response = None
                for _api_attempt in range(3):
                    try:
                        lim = self._check_llm_token_limits()
                        if lim:
                            logger.warning(
                                "Stopping ReAct mid-cycle: token limit %s hit before sample",
                                lim,
                            )
                            self._halted = True
                            return 999999
                        response = await chat.sample()
                        break
                    except Exception as api_err:
                        if _api_attempt < 2:
                            wait = 2 ** (_api_attempt + 1)  # 2s, 4s
                            logger.warning(f"Grok API error (attempt {_api_attempt+1}/3): {api_err} — retrying in {wait}s")
                            await _safe_sleep(wait)
                        else:
                            raise  # Last attempt, propagate

                raw = response.content

                # Log cost
                usage = response.usage
                self.cost_tracker.log_llm_usage(
                    self.grok.model,
                    usage=usage,
                    purpose="agent_loop",
                )
                cost_summary = self.cost_tracker.get_budget_summary()
                reasoning_tok = getattr(usage, "reasoning_tokens", 0) or 0
                cached_tok = getattr(usage, "cached_prompt_text_tokens", 0) or 0
                logger.info(
                    "LLM prompt=%s completion=%s reasoning=%s cached_text=%s | $%.4f today",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    reasoning_tok,
                    cached_tok,
                    cost_summary.today_llm_cost,
                )
                if hasattr(self, "_cycle_prompt_metrics"):
                    from core.runtime.prompt_budget import attach_turn_usage, log_cycle_prompt_budget

                    m = self._cycle_prompt_metrics
                    m.react_turn = turn
                    attach_turn_usage(m, usage)
                    log_cycle_prompt_budget(self._cycle_id, m)
                lim2 = self._check_llm_token_limits()
                if lim2:
                    logger.warning(
                        "Stopping after sample: token limit %s exceeded (logged this call)",
                        lim2,
                    )
                    self._halted = True
                    return 999999

                # Log what the model actually said
                logger.info(f"Grok -> {(raw or '(empty)')[:500]}")

                # ── Parse JSON actions ──────────────────────────
                json_objects = _parse_json_objects(raw or "")

                if not json_objects:
                    invalid_json_streak += 1
                    logger.warning(
                        f"No valid JSON (streak={invalid_json_streak}): {(raw or '')[:100]}"
                    )
                    chat.append(response)
                    if invalid_json_streak >= 3:
                        logger.warning(
                            "Circuit breaker: repeated invalid JSON responses "
                            f"({invalid_json_streak})"
                        )
                        return 60
                    chat.append(sdk_user(
                        "Respond with exactly ONE JSON object and no prose/code fences.\n"
                        "Required shape: {\"action\":\"<tool_or_done>\", ...params}.\n"
                        "If waiting, use: "
                        "{\"action\":\"done\",\"summary\":\"...\",\"cooldown\":30}."
                    ))
                    continue
                invalid_json_streak = 0

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
                        wait_reason = str(action_data.get('wait_reason', '') or '').strip()[:160]
                        self._last_wait_reason = wait_reason
                        wr_log = f" | waiting because: {wait_reason}" if wait_reason else ""
                        logger.info(f"Decision: done ({cooldown}s) | {summary}{wr_log}")
                        cycle_actions.append("done")
                        from core.runtime.prompt_budget import (
                            CYCLE_LAST_SUMMARY_CHARS,
                            truncate_text,
                        )

                        self._last_cycle_summary = truncate_text(
                            f"C{self._cycle_id}: "
                            + ", ".join(cycle_actions[-4:])
                            + (f" — {summary}" if summary else ""),
                            CYCLE_LAST_SUMMARY_CHARS,
                            marker="…",
                        )
                        self._append_snapshot(f"C{self._cycle_id}: done")

                        try:
                            from datetime import datetime as _dt
                            from datetime import timezone as _tz

                            from core.quality.quality_matrix import (
                                DecisionProvenanceSnapshot,
                                ToolUsageRecord,
                                get_quality_matrix_service,
                            )
                            svc = get_quality_matrix_service()
                            # Capture recent tool usages as provenance (lightweight copy)
                            recent_tools = [
                                ToolUsageRecord(
                                    tool_name=getattr(t, 'tool_name', str(t)),
                                    called_at=getattr(t, 'called_at', _dt.now(_tz.utc)),
                                    symbol=getattr(t, 'symbol', None),
                                    success=getattr(t, 'success', True),
                                )
                                for t in svc.get_matrix().recent_tool_usage[-5:]
                            ]
                            snap = DecisionProvenanceSnapshot(
                                ts=_dt.now(_tz.utc),
                                cycle_id=self._cycle_id,
                                decision_type="done",
                                symbol=None,
                                tools_used=recent_tools,
                                quality_state={"overall": svc.get_matrix().overall_quality},
                                context_quality=svc.get_matrix().overall_quality,
                                outcome="done",
                                notes=summary[:120],
                            )
                            svc.record_decision_snapshot(snap)
                        except Exception as _prov_err:
                            logger.debug(f"QualityMatrix provenance snapshot skipped: {_prov_err}")

                        return cooldown

                    else:
                        # ── Execute tool action ────────────────
                        last_result = await self.tools.execute(action, action_data)
                        sym = action_data.get('symbol', '')

                        # Cache research results for cross-cycle memory
                        if action == 'research' and last_result.success:
                            data = last_result.data if hasattr(last_result, 'data') else last_result
                            if isinstance(data, dict) and 'result' in data:
                                import time as _time
                                query = action_data.get('query', '')
                                summary_text = str(data['result'])[:2000]
                                ts = datetime.now().strftime('%H:%M')
                                cat, ttl = _categorize_query(query)
                                self._research_results[query] = {
                                    "summary": summary_text,
                                    "ts": ts,
                                    "session": self._current_session,
                                    "category": cat,
                                    "ttl": ttl,
                                    "created": _time.time(),
                                }
                                # Evict oldest if > 15 cached queries
                                if len(self._research_results) > 15:
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

                    chat.append(response)
                    # State is provided once at cycle start; agent calls
                    # refresh_state tool when it needs an updated snapshot.
                    _fb = _truncate_for_react_context(
                        str(last_result), AGENT_TOOL_FEEDBACK_MAX_CHARS
                    )
                    chat.append(sdk_user(
                        f"TOOL RESULT:\n{_fb}\n\nWhat's your next action?"
                    ))

            except Exception as e:
                logger.error(f"Turn error: {e}", exc_info=True)
                return 60


# ── Entry Point ─────────────────────────────────────────────────

async def run_agent():
    """Main entry point for the autonomous trading agent."""
    logger.info("=" * 60)
    logger.info("GROK TRADER — Pure ReAct")
    logger.info(f"Risk per trade: {RISK_PER_TRADE*100:.1f}%  |  Min R:R: {MIN_RR_RATIO}:1")
    logger.info(f"Trading mode: {TRADING_MODE}")
    if PAPER_AGGRESSIVE:
        logger.info(">>> AGGRESSIVE PAPER MODE — stress testing complex orders <<<")
    logger.info("=" * 60)

    # Connect broker (retry with backoff)
    gateway = None
    for _conn_attempt in range(5):
        try:
            gateway = await create_gateway({})
            break
        except (BrokerConfigError, ConnectionError) as e:
            if _conn_attempt < 4:
                wait = 2 ** (_conn_attempt + 1)  # 2, 4, 8, 16s
                logger.warning(f"Broker connect attempt {_conn_attempt+1}/5 failed: {e} — retrying in {wait}s")
                await _safe_sleep(wait)
            else:
                logger.error(f"Broker connection failed after 5 attempts: {e}")
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
    # Let the refresh_state tool call back into the agent
    tools._state_builder = agent._build_state_context
    tools._agent = agent

    logger.info("Agent started (paper mode recommended)")
    agent._capture_start_of_day_cash()

    # ── Main loop (delegated to CycleScheduler) ─────────────────
    from core.wake_events import wake_bus
    scheduler = CycleScheduler(
        agent=agent,
        wake_bus=wake_bus,
        market_hours_provider=market_hours,
    )
    try:
        await scheduler.run()
    finally:
        if gateway:
            try:
                await gateway.disconnect()
            except Exception:
                pass


# Trader entry point: python __main__.py (see docs/entry-points.md).
