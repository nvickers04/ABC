"""
Hybrid ReAct Agent Loop — Reasoning + Multi-Agent Research

Architecture:
  - REASONING model (single-agent) drives the ReAct loop with client-side tools
  - MULTI-AGENT model (4/16 agents) is exposed as a research() tool for web/X search
  - The reasoning model decides when to call research() — no separate coordination

Grok chains actions until it signals 'done' to refresh context.
research() is the primary discovery mechanism.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from core.async_utils import safe_sleep as _safe_sleep
from core.config import (
    SYSTEM_PROMPT,
    RISK_PER_TRADE,
    CYCLE_SLEEP_SECONDS,

    MAX_DAILY_LOSS_PCT,
    INTRADAY_DRAWDOWN_PCT,
    EOD_FLATTEN_MINUTES,
    OPEN_GAP_GUARD_PCT,
    OPEN_GUARD_DELAY_MINUTES,
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
from core.runtime import SafetyController, StateContextBuilder
from core.runtime.scheduler import CycleScheduler, _now_et
from core.runtime.cycle import evaluate_gap_guard
# Re-exported for backward compatibility (these used to live in this module).
from core.runtime.state_context import _SECTOR_MAP, _sector_of  # noqa: F401
from data.cost_tracker import get_cost_tracker
from data.data_provider import get_data_provider
from data.broker_gateway import create_gateway, BrokerConfigError
from data.market_hours import get_market_hours_provider
from tools.tools_executor import ToolExecutor

logger = logging.getLogger(__name__)


# ── Research Cache Entry ────────────────────────────────────────
# Implementations live in core.research_topics and core.json_parse
# (PR14 extraction). Re-exported here for back-compat with any
# external callers and with internal uses below.

from core.research_topics import (  # noqa: E402,F401
    _TICKER_SYMBOLS,
    _categorize_query,
    _get_ticker_symbols,
)
from core.json_parse import (  # noqa: E402,F401
    _close_truncated_json,
    _parse_json_objects,
    _try_repair_json,
)


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
        """Append a cycle snapshot, keeping last 5."""
        self._market_snapshots.append(f"[{datetime.now().strftime('%H:%M')}] {summary}")
        if len(self._market_snapshots) > 5:
            self._market_snapshots = self._market_snapshots[-5:]

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

        Thin shim — body lives in :mod:`core.runtime.review` (PR21).
        """
        from core.runtime.review import run_daily_review
        await run_daily_review(self)

    async def _run_execution_analysis(self) -> None:
        """Analyze execution snapshots: calibrate simulator slippage, propose graduated params.

        Runs when enough new snapshots have accumulated. Groups filled snapshots
        by (order_type, time_bucket, atr_bucket) and computes empirical medians.
        Then asks Grok to review the data and propose execution config changes.
        """
        try:
            import statistics as _stats
            from memory import (
                get_db, get_filled_snapshots, get_graduated_params,
                get_research_config, set_research_config,
                upsert_calibrated_slippage, insert_graduated_param,
                validate_param_key, deactivate_graduated_param,
                get_snapshots_for_param_review,
            )
            db = get_db()
            last_id = int(get_research_config("last_analysis_snapshot_id", 0.0))

            # Get all filled snapshots since last analysis
            snapshots = get_filled_snapshots(since_id=last_id, limit=500)
            if len(snapshots) < 5:
                logger.info(f"Execution analysis: only {len(snapshots)} new snapshots, skipping")
                return

            # ── 1. Calibrate simulator slippage ──────────────────
            # Group by (order_type, time_bucket, atr_bucket)
            from collections import defaultdict
            groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            # Also collect broad groups for fallbacks
            by_order_type: dict[str, list[float]] = defaultdict(list)

            for snap in snapshots:
                slip = snap.get("slippage_bps")
                if slip is None:
                    continue
                ot = snap.get("order_type", "unknown") or "unknown"
                tb = snap.get("time_bucket", "all") or "all"
                ab = snap.get("atr_bucket", "all") or "all"
                groups[(ot, tb, ab)].append(abs(slip))
                by_order_type[ot].append(abs(slip))

            calibrated_count = 0
            for (ot, tb, ab), slips in groups.items():
                if len(slips) < 3:
                    continue
                slips.sort()
                median = _stats.median(slips)
                p25 = slips[len(slips) // 4] if len(slips) >= 4 else slips[0]
                p75 = slips[3 * len(slips) // 4] if len(slips) >= 4 else slips[-1]
                upsert_calibrated_slippage(
                    order_type=ot, time_bucket=tb, atr_bucket=ab,
                    median_bps=round(median, 2), sample_count=len(slips),
                    p25_bps=round(p25, 2), p75_bps=round(p75, 2),
                )
                calibrated_count += 1

            # Also store time-bucket calibrations: (ot, tb, "all") — used by simulator
            by_ot_tb: dict[tuple[str, str], list[float]] = defaultdict(list)
            for (ot, tb, ab), slips in groups.items():
                by_ot_tb[(ot, tb)].extend(slips)
            for (ot, tb), slips in by_ot_tb.items():
                if len(slips) < 3:
                    continue
                slips.sort()
                median = _stats.median(slips)
                upsert_calibrated_slippage(
                    order_type=ot, time_bucket=tb, atr_bucket="all",
                    median_bps=round(median, 2), sample_count=len(slips),
                )

            # Also store broad per-order-type calibrations (time_bucket='all', atr_bucket='all')
            for ot, slips in by_order_type.items():
                if len(slips) < 3:
                    continue
                slips.sort()
                median = _stats.median(slips)
                upsert_calibrated_slippage(
                    order_type=ot, time_bucket="all", atr_bucket="all",
                    median_bps=round(median, 2), sample_count=len(slips),
                )

            logger.info(f"Simulator calibration: {calibrated_count} bucket(s) updated from {len(snapshots)} snapshots")

            # ── 1b. Review active graduated params (closed-loop) ─
            active_params = get_graduated_params(active_only=True)
            for gp in active_params:
                review = get_snapshots_for_param_review(gp["id"], gp["ts"])
                after = review["after"]
                before = review["before"]
                if len(after) < 10:
                    continue  # Not enough post-activation data yet
                if not before:
                    continue  # No baseline data — can't judge regression
                after_median = _stats.median(after)
                before_median = _stats.median(before)
                # Deactivate if post-activation slippage is worse (higher) than pre-activation
                if after_median > before_median * 1.10:
                    deactivate_graduated_param(
                        gp["id"],
                        f"Rolled back: after median {after_median:.1f}bps > before {before_median:.1f}bps "
                        f"(n_after={len(after)}, n_before={len(before)})"
                    )
                    logger.info(
                        f"Rolled back graduated param {gp['param_key']}: "
                        f"after={after_median:.1f}bps vs before={before_median:.1f}bps"
                    )

            # ── 2. Propose graduated params via LLM ──────────────
            # Build summary stats for the LLM
            summary_lines = []
            for (ot, tb, ab), slips in sorted(groups.items(), key=lambda x: -len(x[1])):
                if len(slips) < 3:
                    continue
                median = _stats.median(slips)
                mean = _stats.mean(slips)
                summary_lines.append(
                    f"  {ot} | {tb} | {ab}: n={len(slips)}, "
                    f"median={median:.1f}bps, mean={mean:.1f}bps, "
                    f"range=[{min(slips):.1f}, {max(slips):.1f}]"
                )

            # Get current graduated params for context
            current_params = get_graduated_params(active_only=True)
            param_lines = [
                f"  {p['param_key']} = {p['param_value']} (since {p['ts'][:10]}, {p['improvement_bps']:.1f}bps improvement)"
                for p in current_params
            ] if current_params else ["  (none)"]

            if summary_lines:
                prompt = f"""Execution analysis review. {len(snapshots)} new order executions analyzed.

Slippage by (order_type | time_bucket | atr_bucket):
{chr(10).join(summary_lines[:20])}

Current graduated parameters:
{chr(10).join(param_lines)}

Based on this data, propose 0 to 2 execution config changes that would reduce slippage.
Only propose a change if the evidence is clear (n >= 10 for both comparison groups).
Changes must be specific and testable, e.g., "use adaptive orders instead of market orders during open for high-ATR stocks".

CRITICAL: param_key MUST use exactly this 4-part dot-separated format:
  {{order_type}}.{{intent}}.{{time_bucket}}.{{atr_bucket}}

The param_key describes the CONTEXT to match (the CURRENT order type and conditions),
NOT the new value. The new value goes in param_value.
Example: to switch from market to adaptive for entries at open with high ATR:
  param_key: "market.entry.open.high"  (matches current market orders in this context)
  param_value: "adaptive"  (the replacement order type)
  previous_value: "market"  (what it replaces)

Valid order_type: market, limit, stop_entry, bracket, trailing_stop, oca_exit, midprice, adaptive, vwap, twap, relative, snap_mid, moc, moo, loc, loo, all
Valid intent: entry, exit, stop, all
Valid time_bucket: open, morning, midday, close, extended, all
Valid atr_bucket: low, medium, high, all

Respond with ONLY a JSON array of objects:
  "param_key": "order_type.intent.time_bucket.atr_bucket",
  "param_value": "the new value",
  "previous_value": "what it replaces or null",
  "evidence_summary": "brief stats justification",
  "estimated_improvement_bps": number

If no changes warranted: []"""

                try:
                    chat = self.grok.client.chat.create(
                        model=self.grok.model,
                        messages=[
                            sdk_system("You are an execution quality analyst. Review slippage data and propose concrete parameter changes. Be conservative — only propose when evidence is strong. Respond ONLY with a JSON array."),
                            sdk_user(prompt),
                        ],
                        temperature=0.3,
                        max_tokens=1024,
                    )
                    response = None
                    for _api_attempt in range(3):
                        try:
                            response = await chat.sample()
                            break
                        except Exception as api_err:
                            if _api_attempt < 2:
                                await _safe_sleep(2 ** (_api_attempt + 1))
                            else:
                                raise

                    usage = response.usage
                    self.cost_tracker.log_llm_call(
                        model=self.grok.model,
                        tokens_in=usage.prompt_tokens,
                        tokens_out=usage.completion_tokens,
                        purpose="execution_analysis",
                    )

                    raw = response.content or ""
                    proposals = _parse_json_objects(raw)
                    if not proposals:
                        try:
                            parsed = json.loads(raw.strip())
                            if isinstance(parsed, list):
                                proposals = parsed
                        except Exception:
                            pass
                    if len(proposals) == 1 and isinstance(proposals[0], list):
                        proposals = proposals[0]

                    for prop in proposals[:2]:
                        if not isinstance(prop, dict) or not prop.get("param_key"):
                            continue
                        # Validate param_key against structured schema
                        key_err = validate_param_key(prop["param_key"])
                        if key_err:
                            logger.info(f"Proposal rejected (bad key): {key_err}")
                            continue
                        # Validate with Mann-Whitney U test if we have comparison data
                        p_value = self._test_proposal(prop, groups)
                        if p_value is not None and p_value < 0.20:
                            insert_graduated_param(
                                param_key=prop["param_key"],
                                param_value=str(prop.get("param_value", "")),
                                previous_value=prop.get("previous_value"),
                                evidence_json=json.dumps(prop.get("evidence_summary", "")),
                                snapshots_analyzed=len(snapshots),
                                improvement_bps=float(prop.get("estimated_improvement_bps", 0)),
                                p_value=p_value,
                            )
                            logger.info(f"Graduated param: {prop['param_key']} = {prop['param_value']} (p={p_value:.3f})")
                        else:
                            logger.info(f"Proposal rejected (p={p_value}): {prop.get('param_key')}")

                except Exception as llm_err:
                    logger.warning(f"Execution analysis LLM call failed: {llm_err}")

            # Mark analysis as complete up to the latest snapshot
            max_id = max(s["id"] for s in snapshots)
            set_research_config("last_analysis_snapshot_id", float(max_id), "execution_analysis_complete")
            logger.info(f"Execution analysis complete: analyzed {len(snapshots)} snapshots")

        except Exception as e:
            logger.warning(f"Execution analysis failed: {e}", exc_info=True)

    def _test_proposal(self, proposal: dict, groups: dict) -> float | None:
        """Run Mann-Whitney U test on a proposal's implied comparison.

        Uses the structured param_key to identify the target bucket group
        and compares it against all other groups of the same order type.
        Returns p-value or None if insufficient data.
        """
        try:
            from scipy.stats import mannwhitneyu
        except ImportError:
            # scipy not available — reject, can't validate
            logger.info("scipy not available, cannot validate proposal statistically")
            return None

        key = proposal.get("param_key", "")
        parts = key.split(".")
        if len(parts) != 4:
            return None

        target_ot, target_intent, target_tb, target_ab = parts

        # Collect the target group's slippage data
        target_data = []
        other_data = []
        for (ot, tb, ab), slips in groups.items():
            if len(slips) < 3:
                continue
            ot_match = (target_ot == "all" or ot == target_ot)
            tb_match = (target_tb == "all" or tb == target_tb)
            ab_match = (target_ab == "all" or ab == target_ab)
            if ot_match and tb_match and ab_match:
                target_data.extend(slips)
            elif ot_match:
                # Same order type but different bucket — use as comparison
                other_data.extend(slips)

        if len(target_data) >= 5 and len(other_data) >= 5:
            try:
                _, p_value = mannwhitneyu(target_data, other_data, alternative='two-sided')
                return round(p_value, 4)
            except Exception:
                pass

        # Not enough data for statistical comparison — reject
        logger.debug(
            f"Proposal {key} rejected: insufficient data "
            f"(target={len(target_data)}, other={len(other_data)}, need 5 each)"
        )
        return None

    def _evaluate_risk_ramp(self, db, today: str) -> None:
        """Check if live trading performance warrants risk ramp-up.

        Thin shim — body lives in :mod:`core.runtime.review` (PR28).
        """
        from core.runtime.review import evaluate_risk_ramp
        evaluate_risk_ramp(db, today)

    async def _build_state_context(self) -> str:
        """Thin shim that delegates to :class:`StateContextBuilder`.

        Kept on the agent so existing callers (notably
        ``tools._state_builder = agent._build_state_context``) continue
        to work without changes.
        """
        return await self._state_context_builder.build()

    async def run_cycle(self) -> int:
        """
        Run one cycle of the ReAct loop.

        Returns cooldown seconds.
        """
        self._cycle_failures = 0
        self._cycle_id += 1
        self._last_step = "detect_session"
        self._last_step_ts = time.time()
        logger.info("CYCLE %d step=detect_session", self._cycle_id)

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
                pre_scan_prompt = (
                    "\n🔍 PRE-SCAN: New trading day. Before placing any trades, run research() "
                    "on today's macro outlook (e.g. 'market outlook today earnings economic data') "
                    "and check economic_calendar(). Review briefing(detail='environment') for regime context. "
                    "This primes your thesis with fresh data before committing capital.\n"
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
        if self._last_wait_reason or self._last_wake_reason:
            wr = self._last_wait_reason or "(none stated)"
            wk = self._last_wake_reason or "(timer)"
            continuity += f"WAITED BECAUSE: {wr}\nWOKE BECAUSE: {wk}\n"
        if self._market_snapshots:
            continuity += "RECENT SNAPSHOTS:\n" + "\n".join(self._market_snapshots[-5:]) + "\n"

        from zoneinfo import ZoneInfo
        _et_now = datetime.now(ZoneInfo("America/New_York"))
        # ── Working Memory: curate then render ──────────────────
        # Auto-injected at the top so the agent sees its own prior
        # interpretations (theses, verdicts, watch-fors) before today's
        # state.  See docs/PLAN_COGNITIVE_ARCHITECTURE.md §2-§3.
        wm_block = ""
        try:
            from memory.working_memory import get_working_memory
            wm = get_working_memory()
            wm.curate()
            wm_block = wm.render() + "\n\n"
        except Exception as e:
            logger.debug("working_memory render failed: %s", e)

        # ── Attention: sync new watching_for into triggers, render fires ──
        # Rendered ABOVE working memory so the agent sees "what just
        # fired" before "what I was thinking earlier".  See plan §4.
        attn_block = ""
        try:
            from core.runtime import attention as _attention
            from memory import get_db as _get_db
            _conn = _get_db()
            _attention.sync_from_working_memory(_conn)
            rendered = _attention.render_attention_block(_conn)
            if rendered:
                attn_block = rendered + "\n\n"
        except Exception as e:
            logger.debug("attention render failed: %s", e)

        # ── Intuition: top-N by attention score ──────────────────
        # Rendered just below ATTENTION so "what just fired" comes
        # before "where my edge is sitting".  See plan §5.
        intu_block = ""
        try:
            from core.runtime import intuition as _intuition
            from memory import get_db as _get_db2
            _conn2 = _get_db2()
            rendered_i = _intuition.render_intuition_block(_conn2)
            if rendered_i:
                intu_block = rendered_i + "\n\n"
        except Exception as e:
            logger.debug("intuition render failed: %s", e)

        context = f"""{attn_block}{intu_block}{wm_block}{state_text}
{cost_line}
Time: {_et_now.strftime('%Y-%m-%d %H:%M:%S')} ET
{continuity}{pre_scan_prompt}{gap_guard_prompt}
Account state above. Start by calling briefing() to assess research status."""

        # ── Build xAI SDK chat instance ─────────────────────────
        self._last_step = "chat_create"
        self._last_step_ts = time.time()
        logger.info("CYCLE %d step=chat_create", self._cycle_id)
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
                # Retry Grok API with exponential backoff
                response = None
                for _api_attempt in range(3):
                    try:
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
                        wait_reason = str(action_data.get('wait_reason', '') or '').strip()[:160]
                        self._last_wait_reason = wait_reason
                        wr_log = f" | waiting because: {wait_reason}" if wait_reason else ""
                        logger.info(f"Decision: done ({cooldown}s) | {summary}{wr_log}")
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
                    chat.append(sdk_user(
                        f"TOOL RESULT:\n{last_result}\n\nWhat's your next action?"
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
    print("Grok Trader started (paper mode recommended)")
    asyncio.run(run_agent())
