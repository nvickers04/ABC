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

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Profit allocation to LLM research
PROFIT_ALLOCATION_PCT = 0.50  # 50% of profits go to LLM credits

# Cost per 1M tokens (approximate, update as pricing changes)
MODEL_COSTS = {
    # xAI Grok
    "grok-2": {"input": 2.00, "output": 10.00},
    "grok-2-mini": {"input": 0.30, "output": 0.50},
    "grok-3": {"input": 3.00, "output": 15.00},
    "grok-3-fast": {"input": 1.00, "output": 5.00},
    "grok-4-1-fast-reasoning": {"input": 0.15, "output": 0.60},
    
    # OpenAI (for reference)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    
    # Anthropic (for reference)
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    
    # Default fallback
    "default": {"input": 2.00, "output": 10.00},
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
    # PERSISTENCE
    # =========================================================================
    
    def _load(self):
        """Initialize empty state (in-memory only, no persistence)."""
        self._init_empty()
    
    def _init_empty(self):
        """Initialize empty state."""
        self.llm_calls = []
        self.trade_pnls = []
        self.daily_summaries = {}
    
    def _save(self):
        """No-op — in-memory only, no persistence."""
        pass
    
    # =========================================================================
    # COST CALCULATION
    # =========================================================================
    
    def _calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost in USD for an LLM call."""
        costs = MODEL_COSTS.get(model.lower(), MODEL_COSTS["default"])
        
        # Cost per 1M tokens
        input_cost = (tokens_in / 1_000_000) * costs["input"]
        output_cost = (tokens_out / 1_000_000) * costs["output"]
        
        return input_cost + output_cost
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    def log_llm_call(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        purpose: str = ""
    ) -> float:
        """
        Log an LLM API call.
        
        Returns the cost in USD.
        """
        cost = self._calculate_cost(model, tokens_in, tokens_out)
        
        call = LLMCall(
            timestamp=datetime.now().isoformat(),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            purpose=purpose,
        )
        
        self.llm_calls.append(call)
        self._update_daily_summary_llm(cost)
        self._save()
        
        logger.debug(f"LLM call: {model}, {tokens_in}+{tokens_out} tokens, ${cost:.4f}")
        
        return cost
    
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
    
    def _update_daily_summary_llm(self, cost: float):
        """Update today's summary with LLM cost."""
        today = date.today().isoformat()
        
        if today not in self.daily_summaries:
            self.daily_summaries[today] = DailySummary(date=today)
        
        summary = self.daily_summaries[today]
        summary.llm_calls += 1
        summary.llm_cost_usd += cost
    
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
