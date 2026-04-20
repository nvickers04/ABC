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
from data.cost_tracker import get_cost_tracker
from data.data_provider import get_data_provider
from data.broker_gateway import create_gateway, BrokerConfigError
from data.market_hours import get_market_hours_provider
from tools.tools_executor import ToolExecutor

logger = logging.getLogger(__name__)


# ── Symbol → sector map for portfolio concentration summary ─────
# Matches the diversified RESEARCH_UNIVERSE.  Used for cheap in-prompt
# exposure reporting; no external lookup.  Unknown symbols bucket as
# "other".
_SECTOR_MAP: dict[str, str] = {
    # Mega-cap tech / AI
    "NVDA": "tech", "META": "tech", "AMD": "tech", "AVGO": "tech",
    # High-growth software
    "CRWD": "software", "NET": "software", "PLTR": "software", "APP": "software",
    # Fintech
    "SOFI": "fintech", "HOOD": "fintech",
    # Consumer discretionary
    "DKNG": "discretionary", "CAVA": "discretionary",
    # Healthcare
    "LLY": "healthcare", "UNH": "healthcare", "GILD": "healthcare",
    # Energy
    "XOM": "energy", "OXY": "energy",
    # Traditional financials
    "JPM": "financials", "GS": "financials",
    # Industrials
    "CAT": "industrials", "UPS": "industrials",
    # Staples
    "COST": "staples", "WMT": "staples",
    # Materials
    "FCX": "materials",
    # EM / China
    "BABA": "em",
    # Common hedges
    "SPY": "index_hedge", "QQQ": "index_hedge", "IWM": "index_hedge",
}


def _sector_of(symbol: str) -> str:
    return _SECTOR_MAP.get((symbol or "").upper(), "other")


# ── Research Cache Entry ────────────────────────────────────────

_TICKER_SYMBOLS: set[str] = set()

def _get_ticker_symbols() -> set[str]:
    """Lazily load RESEARCH_UNIVERSE symbols for topic categorization."""
    global _TICKER_SYMBOLS
    if not _TICKER_SYMBOLS:
        try:
            from research.config import RESEARCH_UNIVERSE
            _TICKER_SYMBOLS = {s.upper() for s in RESEARCH_UNIVERSE}
        except Exception:
            pass
    return _TICKER_SYMBOLS


def _categorize_query(query: str) -> tuple[str, int]:
    """Categorize a research query and return (category, ttl_seconds).

    Categories:
      macro  — market-wide events, Fed, economic → TTL=session boundary
      sector — industry, sector rotation           → TTL=4h
      ticker — specific symbol mentions             → TTL=30min
    """
    q = query.upper()
    tickers = _get_ticker_symbols()
    # Check if query mentions a known ticker
    for sym in tickers:
        if sym in q.split():
            return "ticker", 1800  # 30 min
    # Sector keywords
    sector_kw = {"SECTOR", "INDUSTRY", "ROTATION", "CYCLICAL", "DEFENSIVE", "GROWTH", "VALUE"}
    if any(kw in q for kw in sector_kw):
        return "sector", 14400  # 4 hours
    return "macro", 0  # 0 means "until session change"


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
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


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
        self._last_step = "init"
        self._last_step_ts = time.time()
        self._last_cycle_summary = ""
        self._last_wait_reason = ""    # why the agent chose to wait (set in done)
        self._last_wake_reason = ""    # what woke it (scorer_round_N / timeout / ...)
        self._halted = False
        self._start_of_day_cash: Optional[float] = None
        self._session_high_water: Optional[float] = None  # Peak NLV for drawdown tracking
        self._gap_guard_until: Optional[datetime] = None   # Delay entries after large gap
        self._eod_flattened_date: Optional[str] = None     # Track EOD flatten per day
        self._market_snapshots: list[str] = []  # Rolling 5-cycle summaries
        self._research_results: dict[str, dict] = {}  # query -> {summary, ts, session, category, ttl, created}
        self._current_session: str = "unknown"
        self._daily_review_date: Optional[str] = None  # Track which day we've reviewed
        self._pre_scan_date: Optional[str] = None  # Track which day we've pre-scanned

        logger.info("TradingAgent initialized (minimal ReAct mode)")

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

    def _check_intraday_drawdown(self) -> Optional[float]:
        """Check peak-to-trough drawdown within current session.

        Tracks the highest NLV seen today. If current NLV drops more than
        INTRADAY_DRAWDOWN_PCT from the peak, returns the drawdown %.
        """
        current = self.gateway.net_liquidation if self.gateway else 0
        if current <= 0:
            current = self.gateway.cash_value if self.gateway else 0
        if current <= 0:
            return None

        # Update high water mark
        if self._session_high_water is None or current > self._session_high_water:
            self._session_high_water = current

        drawdown_pct = (self._session_high_water - current) / self._session_high_water * 100
        return drawdown_pct if drawdown_pct >= INTRADAY_DRAWDOWN_PCT else None

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
        """Returns True if daily LLM cost ceiling reached."""
        summary = self.cost_tracker.get_budget_summary()
        return summary.today_llm_cost >= MAX_DAILY_LLM_COST

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

        Called once when session transitions to POSTMARKET. Computes execution
        gaps, triggers execution analysis when enough snapshots exist.
        """
        try:
            from memory import get_db, get_new_snapshot_count
            db = get_db()
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # 1. Aggregate today's trade_feedback for execution gap
            rows = db.execute(
                """SELECT slot, symbol,
                          AVG(execution_gap) as avg_gap,
                          COUNT(*) as n,
                          SUM(actual_pnl) as total_pnl,
                          SUM(simulated_return) as total_sim
                   FROM trade_feedback
                   WHERE ts >= ? || 'T00:00:00'
                   GROUP BY slot""",
                (today,),
            ).fetchall()

            for r in rows:
                gap = r["avg_gap"] if r["avg_gap"] else 0
                if abs(gap) > 0.005:  # >0.5% gap is noteworthy
                    self._emit_hypothesis(
                        hypothesis_type="execution_gap",
                        description=f"Slot {r['slot']} avg execution gap {gap:+.3f} on {r['n']} trades",
                        suggested_action=f"Factor {gap:+.1%} execution cost into slot {r['slot']} sizing",
                        priority=4,
                        related_slot=r["slot"],
                    )

            today_trades = db.execute(
                "SELECT COUNT(*) as n, SUM(pnl) as total_pnl FROM trades WHERE ts >= ? || 'T00:00:00'",
                (today,),
            ).fetchone()
            day_trade_count = today_trades["n"] if today_trades and today_trades["n"] else 0

            # 2. Execution analysis: triggered by data threshold
            new_snaps = get_new_snapshot_count()
            if new_snaps >= 10:
                await self._run_execution_analysis()

            # 3. Risk ramp-up evaluation (live mode only: 0.5% → 1.0%)
            if TRADING_MODE == "live":
                self._evaluate_risk_ramp(db, today)

            logger.info(f"Daily review complete for {today}: {day_trade_count} trades, {new_snaps} new snapshots")
        except Exception as e:
            logger.warning(f"Daily review failed: {e}")

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
        """Check if live trading performance warrants risk ramp-up 0.5% → 1.0%.

        Criteria: 10+ trading days with trades, cumulative P&L > 0, win rate > 45%.
        """
        try:
            from memory import get_research_config, set_research_config
            if get_research_config("risk_ramp_approved", 0.0) >= 1.0:
                return  # Already ramped

            # Count trading days with at least 1 trade in last 30 days
            stats = db.execute(
                """SELECT COUNT(DISTINCT date(ts)) as trading_days,
                          COUNT(*) as total_trades,
                          SUM(pnl) as total_pnl,
                          SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
                   FROM trades
                   WHERE ts >= date(?, '-30 days') || 'T00:00:00'""",
                (today,),
            ).fetchone()

            if not stats or not stats["total_trades"]:
                return

            trading_days = stats["trading_days"] or 0
            total_trades = stats["total_trades"] or 0
            total_pnl = stats["total_pnl"] or 0
            wins = stats["wins"] or 0
            win_rate = wins / total_trades if total_trades > 0 else 0

            if trading_days >= 10 and total_pnl > 0 and win_rate > 0.45:
                set_research_config(
                    "risk_ramp_approved", 1.0,
                    f"Auto-approved: {trading_days} days, {total_trades} trades, "
                    f"P&L=${total_pnl:.2f}, WR={win_rate:.0%}"
                )
                logger.info(
                    f"RISK RAMP-UP APPROVED: {trading_days} days, "
                    f"{total_trades} trades, P&L=${total_pnl:.2f}, WR={win_rate:.0%} → 1.0%"
                )
            else:
                logger.info(
                    f"Risk ramp check: {trading_days}/10 days, "
                    f"P&L=${total_pnl:.2f}, WR={win_rate:.0%} — not yet"
                )
        except Exception as e:
            logger.warning(f"Risk ramp evaluation failed: {e}")

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
            if info.get("early_close"):
                detail += f" | ⚠ EARLY CLOSE {info['close_time']} ET"
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
        lines.append("═══ POSITIONS (verdict required for each before cycle ends) ═══")
        _positions_snapshot = []
        try:
            positions = await self.gateway.get_positions()
            _positions_snapshot = positions or []
            if not positions:
                lines.append("No open positions.")
            else:
                lines.append(
                    "  Each position needs a verdict this cycle: "
                    "HOLD (one-line reason) | TRIM | CLOSE | ROLL | TIGHTEN_STOP | HEDGE."
                )
                lines.append(
                    "  \"Brackets exist\" is not a verdict. Ask: is the original thesis intact, "
                    "and is this still the best use of this capital?"
                )
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

        # ── Portfolio risk summary (cheap aggregation, no extra calls) ──
        try:
            if _positions_snapshot:
                net_liq_val = 0.0
                cash_val = 0.0
                try:
                    _s = await self.gateway.get_account_summary()
                    net_liq_val = float(_s.get("netliquidation", 0) or 0)
                    cash_val = float(_s.get("totalcashvalue", 0) or 0)
                except Exception:
                    pass

                long_stock_notional = 0.0
                short_stock_notional = 0.0
                opt_long_contracts = 0
                opt_short_contracts = 0
                total_unreal = 0.0
                # sym -> {notional, unreal}
                per_symbol: dict[str, dict[str, float]] = {}

                for p in _positions_snapshot:
                    sec = p.get("sec_type", "STK")
                    sym = p.get("symbol", "?")
                    qty = float(p.get("quantity", 0) or 0)
                    mkt = float(p.get("market_price", 0) or 0)
                    unreal = float(p.get("unrealized_pnl", 0) or 0)
                    total_unreal += unreal
                    if sec == "OPT":
                        # option notional ≈ qty * 100 * mkt (premium paid/received)
                        notional = abs(qty) * 100.0 * mkt
                        if qty > 0:
                            opt_long_contracts += int(qty)
                        else:
                            opt_short_contracts += int(-qty)
                    else:
                        notional = abs(qty) * mkt
                        if qty > 0:
                            long_stock_notional += notional
                        else:
                            short_stock_notional += notional
                    agg = per_symbol.setdefault(sym, {"notional": 0.0, "unreal": 0.0})
                    agg["notional"] += notional
                    agg["unreal"] += unreal

                lines.append("")
                lines.append("═══ PORTFOLIO RISK ═══")
                if net_liq_val > 0:
                    pct_long = long_stock_notional / net_liq_val * 100
                    pct_cash = cash_val / net_liq_val * 100
                    lines.append(
                        f"Stock long: ${long_stock_notional:,.0f} ({pct_long:.1f}% of NetLiq)  "
                        f"| Cash: ${cash_val:,.0f} ({pct_cash:.1f}%)"
                    )
                    # Idle-cash flag: if more than 30% of NetLiq is in cash,
                    # the agent has slack and should actively evaluate top
                    # candidates instead of defaulting to skip.
                    if pct_cash > 30:
                        lines.append(
                            f"  ⚠ IDLE CASH: {pct_cash:.0f}% of NetLiq is uninvested. "
                            "Required this cycle: evaluate the top composite from briefing() "
                            "that is not already held — chart_intraday + one context tool "
                            "(news OR iv_info), then verdict TAKE or PASS with reason. "
                            "PASS is fully valid (most evals should pass on marginal/warming "
                            "edge days). Do NOT enter a weak setup just because cash is idle. "
                            "What's not allowed is ending the cycle with NO evaluation."
                        )
                else:
                    lines.append(f"Stock long: ${long_stock_notional:,.0f}  | Cash: ${cash_val:,.0f}")
                if opt_long_contracts or opt_short_contracts:
                    lines.append(
                        f"Options: {opt_long_contracts} long contracts, {opt_short_contracts} short contracts "
                        f"(run position_greeks for net delta/vega/theta)"
                    )
                lines.append(f"Open unrealized P&L: ${total_unreal:+,.2f}")

                # Concentration: top 3 symbols by notional
                if per_symbol:
                    top_syms = sorted(
                        per_symbol.items(), key=lambda kv: kv[1]["notional"], reverse=True
                    )[:3]
                    if net_liq_val > 0:
                        conc_str = ", ".join(
                            f"{s}={v['notional'] / net_liq_val * 100:.1f}%" for s, v in top_syms
                        )
                    else:
                        conc_str = ", ".join(f"{s}=${v['notional']:,.0f}" for s, v in top_syms)
                    lines.append(f"Concentration (top 3 of NetLiq): {conc_str}")

                # Sector exposure — group per-symbol notional by sector
                if per_symbol:
                    sector_totals: dict[str, float] = {}
                    for sym, v in per_symbol.items():
                        sector_totals[_sector_of(sym)] = (
                            sector_totals.get(_sector_of(sym), 0.0) + v["notional"]
                        )
                    if sector_totals:
                        sorted_sectors = sorted(
                            sector_totals.items(), key=lambda kv: kv[1], reverse=True
                        )
                        if net_liq_val > 0:
                            sec_str = ", ".join(
                                f"{s}={n / net_liq_val * 100:.1f}%"
                                for s, n in sorted_sectors if n > 0
                            )
                        else:
                            sec_str = ", ".join(
                                f"{s}=${n:,.0f}" for s, n in sorted_sectors if n > 0
                            )
                        if sec_str:
                            lines.append(f"Sector exposure: {sec_str}")
        except Exception as e:
            logger.debug(f"Portfolio risk summary failed: {e}")

        # ── Research engine status (scorer + template evolution) ──
        try:
            from signals import scorer as _sc
            from signals import template_evolution as _te
            sc_state = "off"
            if _sc.is_scorer_running():
                sc_state = "paused" if _sc.is_scorer_paused() else "running"
            te_state = "off"
            if _te.is_evolution_running():
                te_state = "paused" if _te.is_evolution_paused() else "running"

            # IR snapshot freshness — tells the agent whether briefing.edge
            # is current or stale. Stale edge can still be used but the agent
            # should discount it.
            age_str = "unknown"
            try:
                from memory import get_research_config
                import time as _time
                ts = float(get_research_config("ir_snapshot_ts", 0.0))
                if ts > 0:
                    age_s = _time.time() - ts
                    if age_s < 60:
                        age_str = f"{age_s:.0f}s ago"
                    elif age_s < 3600:
                        age_str = f"{age_s/60:.0f}min ago"
                    elif age_s < 86400:
                        age_str = f"{age_s/3600:.1f}h ago (STALE)"
                    else:
                        age_str = f"{age_s/86400:.1f}d ago (VERY STALE)"
            except Exception:
                pass

            lines.append("")
            lines.append(
                f"═══ RESEARCH ENGINE ═══ scorer={sc_state}  evolution={te_state}  "
                f"edge_snapshot={age_str}"
            )
            lines.append("(control via research_engine action=start|pause|resume|stop)")
        except Exception as e:
            logger.debug(f"Engine status failed: {e}")

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

        # ── Active graduated params (execution research) ────────
        try:
            from memory import get_graduated_params
            params = get_graduated_params(active_only=True)
            if params:
                lines.append("")
                lines.append("═══ EXECUTION OPTIMIZATIONS (auto-graduated) ═══")
                for p in params[:5]:
                    lines.append(f"  {p['param_key']} = {p['param_value']} (+{p['improvement_bps']:.1f}bps)")
        except Exception:
            pass

        return "\n".join(lines)

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
        gap_guard_prompt = ""
        if self._current_session == "regular":
            try:
                if self._gap_guard_until and datetime.now(timezone.utc) < self._gap_guard_until:
                    mins_left = int((self._gap_guard_until - datetime.now(timezone.utc)).total_seconds() / 60)
                    gap_guard_prompt = (
                        f"\n⚠️ GAP GUARD ACTIVE: Large overnight gap detected. "
                        f"Wait {mins_left} more minutes before new entries. "
                        f"Manage existing positions only.\n"
                    )
                elif self._gap_guard_until is None:
                    # First regular-session cycle: check for gap
                    _info_gap = _mh.get_session_info()  # noqa: F841
                    mins_since_open = _info_gap.get("minutes_to_close")
                    # Only check gap in first 5 minutes of regular session
                    if mins_since_open is not None:
                        total_regular = 390  # minutes in regular session
                        mins_elapsed = total_regular - mins_since_open
                        if mins_elapsed <= 5:
                            try:
                                dp = get_data_provider()
                                spy_quote = await dp.get_quote("SPY")
                                if spy_quote:
                                    last = spy_quote.get("last", 0) or spy_quote.get("close", 0)
                                    prev_close = spy_quote.get("previous_close", 0) or spy_quote.get("close", 0)
                                    if last > 0 and prev_close > 0:
                                        gap_pct = abs(last - prev_close) / prev_close * 100
                                        if gap_pct >= OPEN_GAP_GUARD_PCT:
                                            self._gap_guard_until = (
                                                datetime.now(timezone.utc)
                                                + timedelta(minutes=OPEN_GUARD_DELAY_MINUTES)
                                            )
                                            gap_guard_prompt = (
                                                f"\n⚠️ GAP GUARD: SPY gapped {gap_pct:.1f}% overnight. "
                                                f"Waiting {OPEN_GUARD_DELAY_MINUTES} min before new entries. "
                                                f"Manage existing positions only.\n"
                                            )
                                            logger.warning(f"Gap guard triggered: SPY gap {gap_pct:.1f}%")
                                        else:
                                            self._gap_guard_until = datetime.min.replace(tzinfo=timezone.utc)
                            except Exception as e:
                                logger.debug(f"Gap guard quote check failed: {e}")
                                self._gap_guard_until = datetime.min.replace(tzinfo=timezone.utc)
            except Exception:
                pass

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
        context = f"""{state_text}
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
    logger.info("GROK 4.20 TRADER — Pure ReAct")
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

    # ── Main loop ───────────────────────────────────────────────
    cycle = 0
    try:
        while True:
            if agent._halted:
                logger.critical("AGENT HALTED — waiting for market close")
                await _safe_sleep(300)
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
                        await _safe_sleep(1800)
                        continue
            except Exception:
                pass

            cycle += 1
            logger.info(f"{'='*40} CYCLE {cycle} {'='*40}")

            try:
                wait_seconds = await agent.run_cycle()
                logger.info(f"Cooldown: up to {wait_seconds}s (event-driven)")
                from core.wake_events import wake_bus
                wake_reason = await wake_bus.wait(wait_seconds)
                agent._last_wake_reason = wake_reason
                logger.info(f"Woke: {wake_reason}")

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                await _safe_sleep(30)
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
    print("Grok 4.20 Trader started (paper mode recommended)")
    asyncio.run(run_agent())
