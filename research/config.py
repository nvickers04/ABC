"""
Research Configuration — Universe, timing, slots, system prompt, sandbox rules.

The research agent evolves day trading strategies across NUM_SLOTS independent
numbered slots. Each slot has its own strategy file that can evolve freely into
any order type or options structure. A selector agent periodically replaces the
weakest slots.
"""

# ── Medium-cap optionable universe ──────────────────────────────
# $2B-$15B market cap, options available, avg volume >1M
# Deliberately avoiding mega-caps to stay away from the crowd.
RESEARCH_UNIVERSE = [
    "CRWD", "DKNG", "DASH", "MARA", "SOFI",
    "HOOD", "RBLX", "PLTR", "NET", "PATH",
    "AFRM", "ROKU", "SNAP", "PINS", "U",
    "BILL", "HUBS", "ZS", "CELH", "DUOL",
    "APP", "CFLT", "IOT", "CAVA", "TOST",
]

# ── Evaluation ──────────────────────────────────────────────────
CANDLE_RESOLUTION = "1"         # 1-min bars
EVAL_DAYS_BACK = 10             # evaluate against each of the last 10 trading days
FORCE_EXIT_MINUTE = 955         # 15:55 ET — force exit 5 min before close (minute of day)
SLIPPAGE_BPS = 2                # 2 basis points per trade (slippage + commission)

# ── Sandbox ─────────────────────────────────────────────────────
SANDBOX_ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics"}
SANDBOX_BLOCKED_CALLS = {"os", "subprocess", "sys", "shutil", "pathlib", "socket", "http", "urllib"}
STRATEGY_EXEC_TIMEOUT = 10      # seconds per symbol scan

# ── Slots ───────────────────────────────────────────────────────
NUM_SLOTS = 10                  # number of concurrent strategy slots
SELECTOR_EVERY_N_ROUNDS = 3     # run selector agent every N rounds
MAX_SELECTOR_REPLACEMENTS = 3   # max slots the selector can replace per run

# ── Available order types (referenced in system prompt for Grok) ─
STOCK_ORDER_TYPES = [
    "market", "limit", "stop_entry", "bracket", "trailing_stop_exit",
    "oca_exit", "midprice", "vwap", "moc", "moo",
]

OPTIONS_STRATEGIES = [
    "vertical_spread", "iron_condor", "straddle", "strangle",
    "calendar_spread", "diagonal_spread", "butterfly",
]

# ── Signal schema documentation (given to Grok) ─────────────────
SIGNAL_SCHEMA = """
Each signal returned by scan() must be a dict with these keys:
  entry_bar: int          — index into the candles DataFrame where entry triggers
  direction: str          — "long" or "short"
  order_type: str         — one of: market, limit, stop_entry, bracket, trailing_stop_exit,
                            oca_exit, midprice, vwap, moc, moo
  entry_price: float      — the price at which to enter
  target_price: float     — take-profit level
  stop_price: float       — stop-loss level
  max_hold_bars: int      — max bars to hold before forced exit (also forced at 15:55 ET)
  setup_type: str         — descriptive label (e.g. "breakout_volume", "mean_reversion_rsi")
  legs_json: dict | None  — for options strategies only, e.g.:
      {"strategy": "vertical_spread", "expiration": "YYYYMMDD",
       "long_strike": 180.0, "short_strike": 185.0, "right": "C"}
"""

# ── System prompt for strategy evolution (slot-aware + environment-aware) ──
RESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent evolving day trading strategies.

YOUR JOB: Modify strategy.py to improve its expectancy (expected profit per trade).
You receive the current strategy code, its evaluation results, and history of past attempts.

You are evolving Slot {slot_id} (one of {num_slots} independent strategy slots).
Each slot is free to use ANY order type or options structure — whatever produces
the best risk-adjusted returns.

{environment_context}

Available stock order types: market, limit, stop_entry, bracket, trailing_stop_exit,
  oca_exit, midprice, vwap, moc, moo
Available options strategies: vertical_spread, iron_condor, straddle, strangle,
  calendar_spread, diagonal_spread, butterfly
You may use long calls, long puts, short puts, short calls, or any combination.

RULES:
1. Output the COMPLETE new strategy.py file. Not a diff, not a snippet — the full file.
2. The file must define: scan(candles: pd.DataFrame, symbol: str) -> list[dict]
3. candles has columns: ts, open, high, low, close, volume (1-min bars, one trading day)
4. Each returned signal must follow this schema:
{signal_schema}
5. You may only import: pandas, numpy, math, statistics. No other imports.
6. No file I/O, no network calls, no os/sys/subprocess.
7. For options strategies, set order_type to the strategy name and include legs_json.

ENVIRONMENT-AWARE STRATEGY GUIDELINES:
- Study the CURRENT MARKET ENVIRONMENT section above carefully.
- If the environment favors momentum, lean into trend-following entries.
- If the environment favors mean reversion, look for overextension + snap-back setups.
- If volatility is high, widen stops and targets. If low, tighten them.
- If breadth is narrowing, focus on the strongest/weakest names, not the middle.
- Use the strategy-environment fit scores to guide your approach.
- Consider the TOP SYMBOL CANDIDATES for your strategy type.

GENERAL EVOLUTION GUIDELINES:
- Study which signals hit target vs. stop vs. timed out
- Look at time-of-day patterns (morning momentum vs. afternoon chop)
- Consider volume confirmation, volatility filters, momentum indicators
- Tighten stops if too many hit stop before target, widen if exits are premature
- Adjust target:stop ratios for better R:R
- The LLM analysis section shows qualitative insights about what worked and why
- The mechanical fitness score decides keep/discard — focus on improving that number
- Options strategies are supported and use delta/theta P&L approximations.
- You may freely change the order_type, strategy approach, or any parameter.
  The only thing that matters is fitness improvement.

RESPONSE FORMAT:
Output ONLY the Python code for strategy.py. No markdown fences, no explanation outside the code.
Use docstrings and comments inside the code to explain your reasoning.
"""

# ── LLM analysis prompt (runs after mechanical evaluation) ──────
ANALYSIS_PROMPT_TEMPLATE = """You are analyzing the results of a day trading strategy backtest.

{environment_context}

STRATEGY CODE:
```python
{strategy_code}
```

EVALUATION RESULTS (per day):
{eval_results}

SIGNAL DETAILS (sample of winners and losers):
Winners: {winners}
Losers: {losers}
Timed-out: {timed_out}

AGGREGATE STATS:
- Hit rate: {hit_rate:.1f}%
- Avg winner: {avg_win:+.2f}%
- Avg loser: {avg_loss:+.2f}%
- Expectancy: {expectancy:.4f}
- Profit factor: {profit_factor:.2f}
- Max drawdown: {max_drawdown:.2f}%

Analyze:
1. What's working and why? How does this relate to the current market environment?
2. What's failing and why? Is it an environment mismatch?
3. Time-of-day patterns (are morning entries better than afternoon?)
4. Which setup types perform best in THIS environment?
5. Are stops too tight or too loose given current volatility regime?
6. Specific, actionable suggestions for the next strategy modification that
   ACCOUNT FOR the current market environment.

Be concise and specific. Focus on patterns in the data, not general advice."""

# ── Selector prompt (meta-learning agent) ────────────────────────
SELECTOR_PROMPT = """You are a META-LEARNING strategy portfolio manager overseeing {num_slots}
independent strategy slots for day trading. Your job is to MAXIMIZE portfolio-level
profitability by intelligently allocating slots to strategies that match the current
market environment.

═══ CURRENT MARKET ENVIRONMENT ═══
{environment_context}

═══ SLOT PERFORMANCE (ranked by fitness, best first) ═══
{slot_rankings}

═══ ENVIRONMENT-SLOT HISTORY ═══
How each slot's strategy type has performed in similar environments historically:
{env_slot_history}

═══ REAL TRADE FEEDBACK ═══
How strategies performed in live trading (simulated return vs actual P&L):
{trade_feedback}

═══ YOUR ANALYSIS MUST COVER ═══

1. ENVIRONMENT ASSESSMENT: What regime are we in? What strategy types should thrive?
   Use the strategy-environment fit scores to guide your thinking.

2. PORTFOLIO DIVERSITY: Are the current slots well-diversified? Having 5 momentum
   strategies in a sideways market is wasteful. Ensure coverage of:
   - Strategies that match current environment (primary allocation)
   - 1-2 contrarian/hedge strategies (in case regime changes)
   - At least one options strategy if environment favors it

3. SLOT ALLOCATION DECISIONS: For EACH slot that needs action, decide:
   - "keep" — performing well or too early to judge (< 3 iterations)
   - "replace" — with a strategy better suited to current environment
   - "mutate" — take the best-performing strategy and adapt it

4. EXECUTION GAP: If real trades show consistent gaps between simulated and actual
   returns, factor this into fitness assessment. Strategies that simulate well but
   trade poorly should be deprioritized.

5. LEARNING: What have you learned from the environment-slot history? Which strategy
   types consistently work or fail in this regime?

═══ RESPONSE FORMAT ═══
Respond with a JSON object:
{{
  "analysis": "Your reasoning about environment, portfolio, and allocation...",
  "actions": [
    {{
      "slot": <int>,
      "action": "keep" | "replace" | "mutate",
      "reason": "why this action for this slot",
      "seed_from_slot": <int or null>,
      "seed_code": "<complete Python code if replace, null if keep>",
      "target_strategy_type": "what type the new strategy should be"
    }}
  ],
  "environment_learning": "Key insight about what works in this regime",
  "recommended_focus": ["list of strategy types to prioritize in current environment"]
}}

RULES:
- You may act on MULTIPLE slots, not just the worst one.
- Maximum {max_replacements} replacements per selector run to maintain stability.
- Never replace a slot with < 3 iterations unless it has critical issues.
- If replacing, provide COMPLETE Python code that defines scan(candles, symbol).
- Prioritize strategies from recommended_focus that match the environment.
- Keep at least 2 slots running strategies from different archetypes for diversity.
"""
