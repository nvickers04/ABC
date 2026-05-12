"""
LLM Cost Tracker - Budget awareness for autonomous trading.

This module tracks:
1. LLM API costs (per call and cumulative)
2. Trading P&L (realized profits/losses)
3. Research budget (50% of profits allocated to LLM credits)

The LLM can query this to understand its resource consumption
and available budget for additional research.

POLICY: 50% of realized trading profits are allocated to LLM research credits.
        This creates a self-sustaining system where good trading funds more research.

NOTE: Currently trust-based. No hard enforcement until profits exist.

Usage:
    from data.cost_tracker import CostTracker, get_cost_tracker
    
    tracker = get_cost_tracker()
    
    # Log an LLM call
    tracker.log_llm_call(model="grok-2", tokens_in=1500, tokens_out=500)
    
    # Log a realized trade
    tracker.log_trade_pnl(symbol="AAPL", pnl=150.00)
    
    # LLM checks its budget
    budget = tracker.get_budget_summary()
"""

import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Profit allocation to LLM research
PROFIT_ALLOCATION_PCT = 0.50  # 50% of profits go to LLM credits

# Cost per 1M tokens (approximate; align with https://docs.x.ai/docs/pricing )
MODEL_COSTS = {
    # xAI Grok 4.3 — docs.x.ai/developers/models/grok-4.3 (cached input priced separately)
    "grok-4.3": {"input": 1.25, "cached_input": 0.20, "output": 2.50},
    "grok-4.3-latest": {"input": 1.25, "cached_input": 0.20, "output": 2.50},
    "grok-latest": {"input": 1.25, "cached_input": 0.20, "output": 2.50},
    # xAI Grok 4.20-era slugs (legacy logs + prior builds)
    "grok-4.20-0309-reasoning": {"input": 2.00, "cached_input": 2.00, "output": 6.00},
    "grok-4.20-experimental-beta-0304-reasoning": {"input": 2.00, "cached_input": 2.00, "output": 6.00},
    "grok-4.20-experimental-beta-0304-non-reasoning": {"input": 2.00, "cached_input": 2.00, "output": 6.00},
    "grok-4.20-multi-agent-experimental-beta-0304": {"input": 1.25, "cached_input": 0.20, "output": 2.50},
    "grok-4.20-multi-agent": {"input": 1.25, "cached_input": 0.20, "output": 2.50},
    "grok-4.20-multi-agent-0309": {"input": 1.25, "cached_input": 0.20, "output": 2.50},
    # xAI Grok (legacy)
    "grok-2": {"input": 2.00, "cached_input": 2.00, "output": 10.00},
    "grok-2-mini": {"input": 0.30, "cached_input": 0.30, "output": 0.50},
    "grok-3": {"input": 3.00, "cached_input": 3.00, "output": 15.00},
    "grok-3-fast": {"input": 1.00, "cached_input": 1.00, "output": 5.00},
    "grok-4-1-fast-reasoning": {"input": 0.15, "cached_input": 0.15, "output": 0.60},
    
    # Default fallback (no cached discount — conservative)
    "default": {"input": 2.00, "cached_input": 2.00, "output": 10.00},
}

# Data kept in-memory only — no daily_store persistence


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LLMCall:
    """Record of a single LLM API call."""
    timestamp: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    purpose: str = ""  # e.g., "screening", "tactic_selection", "research"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradePnL:
    """Record of realized P&L from a trade."""
    timestamp: str
    symbol: str
    pnl_usd: float
    trade_type: str = ""  # e.g., "day_trade", "swing"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DailySummary:
    """Daily cost and P&L summary."""
    date: str
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    trades_closed: int = 0
    realized_pnl_usd: float = 0.0
    research_budget_usd: float = 0.0  # 50% of profits
    # xAI token tallies (daily, persisted) — see data.xai_usage.extract_billing_token_counts
    llm_noncached_prompt_text_tokens: int = 0
    llm_cached_prompt_text_tokens: int = 0
    llm_prompt_image_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_reasoning_tokens: int = 0
    llm_output_priced_tokens: int = 0  # completion + reasoning (for caps / reporting)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetSummary:
    """Current budget state for LLM visibility."""
    # Today
    today_llm_cost: float = 0.0
    today_llm_calls: int = 0
    today_realized_pnl: float = 0.0
    today_research_budget: float = 0.0
    today_noncached_prompt_text_tokens: int = 0
    today_cached_prompt_text_tokens: int = 0
    today_prompt_image_tokens: int = 0
    today_completion_tokens: int = 0
    today_reasoning_tokens: int = 0
    today_output_priced_tokens: int = 0

    # All time
    total_llm_cost: float = 0.0
    total_llm_calls: int = 0
    total_realized_pnl: float = 0.0
    total_research_budget: float = 0.0
    
    # Budget status
    budget_remaining: float = 0.0  # research_budget - llm_cost
    budget_status: str = "OK"  # OK, LOW, EXCEEDED, NO_PROFITS_YET
    
    # Policy
    profit_allocation_pct: float = PROFIT_ALLOCATION_PCT
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_llm_string(self) -> str:
        """Format for LLM context."""
        lines = [
            "=== LLM COST & BUDGET STATUS ===",
            "",
            f"TODAY ({date.today().isoformat()}):",
            f"  LLM Calls: {self.today_llm_calls}",
            f"  LLM Cost: ${self.today_llm_cost:.4f}",
            f"  Prompt text (non-cached) tokens today: {self.today_noncached_prompt_text_tokens:,}",
            f"  Cached prompt text tokens today: {self.today_cached_prompt_text_tokens:,}",
            f"  Prompt image tokens today: {self.today_prompt_image_tokens:,}",
            f"  Completion tokens today: {self.today_completion_tokens:,}",
            f"  Reasoning tokens today: {self.today_reasoning_tokens:,}",
            f"  Output-priced (completion+reasoning) today: {self.today_output_priced_tokens:,}",
            f"  Realized P&L: ${self.today_realized_pnl:.2f}",
            f"  Research Budget (50% of profits): ${self.today_research_budget:.2f}",
            "",
            "ALL TIME:",
            f"  Total LLM Calls: {self.total_llm_calls}",
            f"  Total LLM Cost: ${self.total_llm_cost:.4f}",
            f"  Total Realized P&L: ${self.total_realized_pnl:.2f}",
            f"  Total Research Budget: ${self.total_research_budget:.2f}",
            "",
            "BUDGET STATUS:",
            f"  Budget Remaining: ${self.budget_remaining:.2f}",
            f"  Status: {self.budget_status}",
            "",
            "POLICY:",
            f"  {int(self.profit_allocation_pct * 100)}% of realized profits allocated to LLM research.",
        ]
        
        if self.budget_status == "NO_PROFITS_YET":
            lines.append("  (Currently trust-based - no profits to allocate yet)")
        
        return "\n".join(lines)


# =============================================================================
# COST TRACKER
# =============================================================================

class CostTracker:
    """
    Tracks LLM costs and trading P&L for budget awareness.
    
    Uses daily-rotated files (data/YYYY-MM-DD/cost_tracker.json).
    Each day starts fresh. Old days are archived in their date folders.
    """
    
    def __init__(self):
        self.llm_calls: List[LLMCall] = []
        self.trade_pnls: List[TradePnL] = []
        self.daily_summaries: Dict[str, DailySummary] = {}
        
        self._load()

    # =========================================================================
    # PERSISTENCE — SQLite-backed daily summaries
    # =========================================================================

    _DB_PATH = Path(__file__).parent.parent / "memory" / "abc.db"

    def _get_db(self) -> sqlite3.Connection:
        """Get or create a connection to the shared DB."""
        if not hasattr(self, "_db") or self._db is None:
            self._DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._DB_PATH), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS cost_daily_summaries (
                    date TEXT PRIMARY KEY,
                    llm_calls INTEGER DEFAULT 0,
                    llm_cost_usd REAL DEFAULT 0.0,
                    trades_closed INTEGER DEFAULT 0,
                    realized_pnl_usd REAL DEFAULT 0.0,
                    research_budget_usd REAL DEFAULT 0.0,
                    llm_noncached_prompt_text_tokens INTEGER DEFAULT 0,
                    llm_cached_prompt_text_tokens INTEGER DEFAULT 0,
                    llm_prompt_image_tokens INTEGER DEFAULT 0,
                    llm_completion_tokens INTEGER DEFAULT 0,
                    llm_reasoning_tokens INTEGER DEFAULT 0,
                    llm_output_priced_tokens INTEGER DEFAULT 0
                )
            """)
            self._db.commit()
            self._ensure_cost_schema_columns(self._db)
        return self._db

    @staticmethod
    def _ensure_cost_schema_columns(db: sqlite3.Connection) -> None:
        """Migrate older ``cost_daily_summaries`` rows forward (SQLite ADD COLUMN)."""
        try:
            have = {
                str(r["name"])
                for r in db.execute("PRAGMA table_info(cost_daily_summaries)").fetchall()
            }
        except Exception:
            return
        alters = [
            ("llm_noncached_prompt_text_tokens", "INTEGER DEFAULT 0"),
            ("llm_cached_prompt_text_tokens", "INTEGER DEFAULT 0"),
            ("llm_prompt_image_tokens", "INTEGER DEFAULT 0"),
            ("llm_completion_tokens", "INTEGER DEFAULT 0"),
            ("llm_reasoning_tokens", "INTEGER DEFAULT 0"),
            ("llm_output_priced_tokens", "INTEGER DEFAULT 0"),
        ]
        for col, decl in alters:
            if col not in have:
                try:
                    db.execute(f"ALTER TABLE cost_daily_summaries ADD COLUMN {col} {decl}")
                    db.commit()
                except sqlite3.OperationalError:
                    pass

    @staticmethod
    def _row_int(row: sqlite3.Row, key: str, default: int = 0) -> int:
        try:
            v = row[key]
            return int(v) if v is not None else default
        except (KeyError, IndexError, TypeError, ValueError):
            return default

    def _load(self):
        """Load daily summaries from SQLite."""
        self.llm_calls = []
        self.trade_pnls = []
        self.daily_summaries = {}
        try:
            db = self._get_db()
            for row in db.execute("SELECT * FROM cost_daily_summaries").fetchall():
                self.daily_summaries[row["date"]] = DailySummary(
                    date=row["date"],
                    llm_calls=self._row_int(row, "llm_calls"),
                    llm_cost_usd=float(row["llm_cost_usd"] or 0.0),
                    trades_closed=self._row_int(row, "trades_closed"),
                    realized_pnl_usd=float(row["realized_pnl_usd"] or 0.0),
                    research_budget_usd=float(row["research_budget_usd"] or 0.0),
                    llm_noncached_prompt_text_tokens=self._row_int(
                        row, "llm_noncached_prompt_text_tokens"
                    ),
                    llm_cached_prompt_text_tokens=self._row_int(
                        row, "llm_cached_prompt_text_tokens"
                    ),
                    llm_prompt_image_tokens=self._row_int(row, "llm_prompt_image_tokens"),
                    llm_completion_tokens=self._row_int(row, "llm_completion_tokens"),
                    llm_reasoning_tokens=self._row_int(row, "llm_reasoning_tokens"),
                    llm_output_priced_tokens=self._row_int(row, "llm_output_priced_tokens"),
                )
        except Exception as e:
            logger.warning(f"Cost tracker load failed (starting fresh): {e}")
            self.daily_summaries = {}
    
    def _init_empty(self):
        """Initialize empty state."""
        self.llm_calls = []
        self.trade_pnls = []
        self.daily_summaries = {}
    
    def _save(self):
        """Persist daily summaries to SQLite."""
        try:
            db = self._get_db()
            for day_key, summary in self.daily_summaries.items():
                db.execute(
                    """INSERT INTO cost_daily_summaries
                       (date, llm_calls, llm_cost_usd, trades_closed,
                        realized_pnl_usd, research_budget_usd,
                        llm_noncached_prompt_text_tokens, llm_cached_prompt_text_tokens,
                        llm_prompt_image_tokens, llm_completion_tokens, llm_reasoning_tokens,
                        llm_output_priced_tokens)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(date) DO UPDATE SET
                           llm_calls = excluded.llm_calls,
                           llm_cost_usd = excluded.llm_cost_usd,
                           trades_closed = excluded.trades_closed,
                           realized_pnl_usd = excluded.realized_pnl_usd,
                           research_budget_usd = excluded.research_budget_usd,
                           llm_noncached_prompt_text_tokens = excluded.llm_noncached_prompt_text_tokens,
                           llm_cached_prompt_text_tokens = excluded.llm_cached_prompt_text_tokens,
                           llm_prompt_image_tokens = excluded.llm_prompt_image_tokens,
                           llm_completion_tokens = excluded.llm_completion_tokens,
                           llm_reasoning_tokens = excluded.llm_reasoning_tokens,
                           llm_output_priced_tokens = excluded.llm_output_priced_tokens""",
                    (
                        day_key,
                        summary.llm_calls,
                        summary.llm_cost_usd,
                        summary.trades_closed,
                        summary.realized_pnl_usd,
                        summary.research_budget_usd,
                        summary.llm_noncached_prompt_text_tokens,
                        summary.llm_cached_prompt_text_tokens,
                        summary.llm_prompt_image_tokens,
                        summary.llm_completion_tokens,
                        summary.llm_reasoning_tokens,
                        summary.llm_output_priced_tokens,
                    ),
                )
            db.commit()
        except Exception as e:
            logger.warning(f"Cost tracker save failed: {e}")
    
    # =========================================================================
    # COST CALCULATION
    # =========================================================================

    def _calculate_cost_from_counts(self, model: str, counts: Dict[str, int]) -> float:
        """USD estimate from xAI-style token buckets (per 1M token list prices)."""
        costs = MODEL_COSTS.get(model.lower(), MODEL_COSTS["default"])
        inp = float(costs["input"])
        cached_rate = float(costs.get("cached_input", inp))
        out = float(costs["output"])
        nc = max(0, int(counts.get("noncached_prompt_text", 0)))
        csh = max(0, int(counts.get("cached_prompt_text", 0)))
        pimg = max(0, int(counts.get("prompt_image", 0)))
        outp = max(0, int(counts.get("output_priced", 0)))
        return (
            (nc / 1_000_000) * inp
            + (csh / 1_000_000) * cached_rate
            + (pimg / 1_000_000) * inp
            + (outp / 1_000_000) * out
        )
    
    def _calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Legacy single-bucket estimate (no cache / reasoning split)."""
        return self._calculate_cost_from_counts(
            model,
            {
                "noncached_prompt_text": max(0, int(tokens_in)),
                "cached_prompt_text": 0,
                "prompt_image": 0,
                "completion": max(0, int(tokens_out)),
                "reasoning": 0,
                "output_priced": max(0, int(tokens_out)),
                "prompt_total": max(0, int(tokens_in)),
            },
        )
    
    # =========================================================================
    # LOGGING
    # =========================================================================

    def log_llm_usage(
        self,
        model: str,
        *,
        usage: Any = None,
        counts: Optional[Dict[str, int]] = None,
        purpose: str = "",
    ) -> float:
        """Log one LLM call using xAI usage buckets (preferred) or pre-extracted counts."""
        from data.xai_usage import extract_billing_token_counts

        if counts is None:
            if usage is None:
                raise ValueError("log_llm_usage requires usage= or counts=")
            counts = extract_billing_token_counts(usage)
        cost = self._calculate_cost_from_counts(model, counts)

        call = LLMCall(
            timestamp=datetime.now().isoformat(),
            model=model,
            tokens_in=int(counts.get("prompt_total", 0)),
            tokens_out=int(counts.get("output_priced", 0)),
            cost_usd=cost,
            purpose=purpose,
        )
        self.llm_calls.append(call)
        self._update_daily_summary_llm(cost, counts)
        self._save()

        logger.debug(
            "LLM usage: %s nc_text=%s cached=%s img=%s out_priced=%s $%.4f (%s)",
            model,
            counts.get("noncached_prompt_text"),
            counts.get("cached_prompt_text"),
            counts.get("prompt_image"),
            counts.get("output_priced"),
            cost,
            purpose or "-",
        )
        return cost
    
    def log_llm_call(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        purpose: str = ""
    ) -> float:
        """
        Log an LLM API call (legacy path — no cache/reasoning split).
        
        Returns the cost in USD.
        """
        return self.log_llm_usage(
            model,
            counts={
                "noncached_prompt_text": max(0, int(tokens_in)),
                "cached_prompt_text": 0,
                "prompt_image": 0,
                "completion": max(0, int(tokens_out)),
                "reasoning": 0,
                "output_priced": max(0, int(tokens_out)),
                "prompt_total": max(0, int(tokens_in)),
            },
            purpose=purpose,
        )

    def check_daily_token_limits(self) -> Optional[str]:
        """Return a short reason string if any daily token ceiling is hit, else ``None``."""
        try:
            from core.config import (
                MAX_DAILY_LLM_CACHED_PROMPT_TEXT_TOKENS,
                MAX_DAILY_LLM_COMPLETION_TOKENS,
                MAX_DAILY_LLM_NONCACHED_PROMPT_TEXT_TOKENS,
                MAX_DAILY_LLM_OUTPUT_PRICED_TOKENS,
                MAX_DAILY_LLM_PROMPT_IMAGE_TOKENS,
                MAX_DAILY_LLM_REASONING_TOKENS,
            )
        except Exception as e:
            logger.debug("token limit config import failed: %s", e)
            return None

        today = date.today().isoformat()
        s = self.daily_summaries.get(today)
        if s is None:
            return None
        if s.llm_noncached_prompt_text_tokens >= MAX_DAILY_LLM_NONCACHED_PROMPT_TEXT_TOKENS:
            return "noncached_prompt_text"
        if s.llm_cached_prompt_text_tokens >= MAX_DAILY_LLM_CACHED_PROMPT_TEXT_TOKENS:
            return "cached_prompt_text"
        if s.llm_prompt_image_tokens >= MAX_DAILY_LLM_PROMPT_IMAGE_TOKENS:
            return "prompt_image"
        if s.llm_completion_tokens >= MAX_DAILY_LLM_COMPLETION_TOKENS:
            return "completion"
        if s.llm_reasoning_tokens >= MAX_DAILY_LLM_REASONING_TOKENS:
            return "reasoning"
        if s.llm_output_priced_tokens >= MAX_DAILY_LLM_OUTPUT_PRICED_TOKENS:
            return "output_priced"
        return None
    
    def log_trade_pnl(
        self,
        symbol: str,
        pnl: float,
        trade_type: str = ""
    ):
        """
        Log realized P&L from a closed trade.
        
        This updates the research budget (50% of profits).
        """
        trade = TradePnL(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            pnl_usd=pnl,
            trade_type=trade_type,
        )
        
        self.trade_pnls.append(trade)
        self._update_daily_summary_pnl(pnl)
        self._save()
        
        if pnl > 0:
            research_credit = pnl * PROFIT_ALLOCATION_PCT
            logger.info(f"Trade P&L: {symbol} ${pnl:.2f} → ${research_credit:.2f} research credit")
        else:
            logger.info(f"Trade P&L: {symbol} ${pnl:.2f} (loss)")
    
    def _update_daily_summary_llm(
        self, cost: float, counts: Optional[Dict[str, int]] = None
    ) -> None:
        """Update today's summary with LLM cost and optional token buckets."""
        today = date.today().isoformat()

        if today not in self.daily_summaries:
            self.daily_summaries[today] = DailySummary(date=today)

        summary = self.daily_summaries[today]
        summary.llm_calls += 1
        summary.llm_cost_usd += cost
        if counts:
            summary.llm_noncached_prompt_text_tokens += int(
                counts.get("noncached_prompt_text", 0)
            )
            summary.llm_cached_prompt_text_tokens += int(counts.get("cached_prompt_text", 0))
            summary.llm_prompt_image_tokens += int(counts.get("prompt_image", 0))
            summary.llm_completion_tokens += int(counts.get("completion", 0))
            summary.llm_reasoning_tokens += int(counts.get("reasoning", 0))
            summary.llm_output_priced_tokens += int(counts.get("output_priced", 0))
    
    def _update_daily_summary_pnl(self, pnl: float):
        """Update today's summary with trade P&L."""
        today = date.today().isoformat()
        
        if today not in self.daily_summaries:
            self.daily_summaries[today] = DailySummary(date=today)
        
        summary = self.daily_summaries[today]
        summary.trades_closed += 1
        summary.realized_pnl_usd += pnl
        
        # Update research budget (50% of positive P&L only)
        if pnl > 0:
            summary.research_budget_usd += pnl * PROFIT_ALLOCATION_PCT
    
    # =========================================================================
    # BUDGET QUERIES
    # =========================================================================
    
    def get_budget_summary(self) -> BudgetSummary:
        """
        Get current budget summary for LLM visibility.
        
        This is what the LLM sees to understand its resource usage.
        """
        today = date.today().isoformat()
        today_summary = self.daily_summaries.get(today, DailySummary(date=today))
        
        # Calculate totals
        total_llm_cost = sum(c.cost_usd for c in self.llm_calls)
        total_llm_calls = len(self.llm_calls)
        total_realized_pnl = sum(t.pnl_usd for t in self.trade_pnls)
        total_research_budget = sum(
            t.pnl_usd * PROFIT_ALLOCATION_PCT 
            for t in self.trade_pnls 
            if t.pnl_usd > 0
        )
        
        budget_remaining = total_research_budget - total_llm_cost
        
        # Determine status
        if total_realized_pnl <= 0 and len(self.trade_pnls) == 0:
            status = "NO_PROFITS_YET"
        elif budget_remaining < 0:
            status = "EXCEEDED"
        elif budget_remaining < total_llm_cost * 0.2:  # Less than 20% of spent
            status = "LOW"
        else:
            status = "OK"
        
        return BudgetSummary(
            # Today
            today_llm_cost=today_summary.llm_cost_usd,
            today_llm_calls=today_summary.llm_calls,
            today_realized_pnl=today_summary.realized_pnl_usd,
            today_research_budget=today_summary.research_budget_usd,
            today_noncached_prompt_text_tokens=today_summary.llm_noncached_prompt_text_tokens,
            today_cached_prompt_text_tokens=today_summary.llm_cached_prompt_text_tokens,
            today_prompt_image_tokens=today_summary.llm_prompt_image_tokens,
            today_completion_tokens=today_summary.llm_completion_tokens,
            today_reasoning_tokens=today_summary.llm_reasoning_tokens,
            today_output_priced_tokens=today_summary.llm_output_priced_tokens,
            # All time
            total_llm_cost=total_llm_cost,
            total_llm_calls=total_llm_calls,
            total_realized_pnl=total_realized_pnl,
            total_research_budget=total_research_budget,
            
            # Budget
            budget_remaining=budget_remaining,
            budget_status=status,
            profit_allocation_pct=PROFIT_ALLOCATION_PCT,
        )
    
    def get_budget_for_llm(self) -> str:
        """
        Get budget summary formatted for LLM context.
        
        Include this in the LLM prompt so it's aware of costs.
        """
        return self.get_budget_summary().to_llm_string()
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def get_cost_by_purpose(self) -> Dict[str, float]:
        """Get LLM costs broken down by purpose."""
        costs = {}
        for call in self.llm_calls:
            purpose = call.purpose or "unknown"
            costs[purpose] = costs.get(purpose, 0) + call.cost_usd
        return costs
    
    def get_cost_by_model(self) -> Dict[str, float]:
        """Get LLM costs broken down by model."""
        costs = {}
        for call in self.llm_calls:
            costs[call.model] = costs.get(call.model, 0) + call.cost_usd
        return costs
    
    def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily stats for the last N days."""
        from datetime import timedelta
        
        stats = []
        for i in range(days):
            day = (date.today() - timedelta(days=i)).isoformat()
            if day in self.daily_summaries:
                stats.append(self.daily_summaries[day].to_dict())
            else:
                stats.append(DailySummary(date=day).to_dict())
        
        return stats
    
    def get_roi(self) -> float:
        """
        Calculate ROI: (Total P&L - Total LLM Cost) / Total LLM Cost
        
        Returns percentage. Negative if losing money overall.
        """
        total_cost = sum(c.cost_usd for c in self.llm_calls)
        total_pnl = sum(t.pnl_usd for t in self.trade_pnls)
        
        if total_cost == 0:
            return 0.0
        
        return ((total_pnl - total_cost) / total_cost) * 100
    
    # =========================================================================
    # RESET
    # =========================================================================
    
    def reset(self):
        """Reset all tracking data. Use with caution."""
        self._init_empty()
        self._save()
        logger.warning("Cost tracker reset")


# =============================================================================
# SINGLETON
# =============================================================================

_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the singleton cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CostTracker',
    'get_cost_tracker',
    'LLMCall',
    'TradePnL',
    'BudgetSummary',
    'PROFIT_ALLOCATION_PCT',
    'MODEL_COSTS',
]
