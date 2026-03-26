"""
Research Configuration — Universe, timing, slots, system prompt, sandbox rules.

The research agent evolves day trading strategies across NUM_SLOTS independent
numbered slots. Each slot has its own strategy file that can evolve freely into
any order type or options structure. A selector agent periodically replaces the
weakest slots.
"""

from pathlib import Path


_PROGRAM_PATH = Path(__file__).resolve().parent.parent / "program.md"


def _escape_prompt_braces(text: str) -> str:
  return text.replace("{", "{{").replace("}", "}}")


def _load_program_markdown() -> str:
  try:
    return _PROGRAM_PATH.read_text(encoding="utf-8").strip()
  except FileNotFoundError:
    return ""


RESEARCH_PROGRAM_MARKDOWN = _escape_prompt_braces(_load_program_markdown())

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
MIN_KEEP_TEST_SIGNALS = 15      # minimum out-of-sample trades before a strategy can be kept
MIN_KEEP_CONFIDENCE_SCORE = 0.25  # minimum confidence score before a strategy can be kept
MIN_KEEP_CONDITION_TRADES = 3   # require repeated evidence inside at least one condition bucket

# ── Sandbox ─────────────────────────────────────────────────────
SANDBOX_ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics"}
SANDBOX_BLOCKED_CALLS = {"os", "subprocess", "sys", "shutil", "pathlib", "socket", "http", "urllib"}
STRATEGY_EXEC_TIMEOUT = 10      # seconds per symbol scan

# ── Slots ───────────────────────────────────────────────────────
NUM_SLOTS = 12                  # number of strategy slots
BATCH_SIZE = 4                  # slots per batch (matches LLM semaphore)
SELECTOR_EVERY_N_ROUNDS = 3     # run selector agent every N rounds
MAX_SELECTOR_REPLACEMENTS = 3   # max slots the selector can replace per run

# ── Slot mandates: 3 regime groups × 4 order-type archetypes ───
# Each slot has a fixed regime assignment and order-type class.
# The LLM can freely evolve parameters, indicators, and logic within the mandate.
# It CANNOT change the regime direction or order-type class.
#
# regime:     "bearish" | "bullish" | "choppy"
# direction:  "short" | "long" | "neutral" | "any" (neutral = options premium, any = dual-sided)
# order_class: broad archetype — enforced at keep gate
# allowed_order_types: specific order_type values that satisfy this mandate
#
SLOT_MANDATES: dict[int, dict] = {
    # ── BEARISH GROUP (slots 1-4): activated when trend=down ────
    1:  {
        "regime": "bearish",
        "direction": "short",
        "order_class": "stock_momentum",
        "allowed_order_types": {"market", "vwap", "stop_entry", "moo"},
        "description": "Short stock momentum — VWAP rejections or breakdowns on trend-following entries",
    },
    2:  {
        "regime": "bearish",
        "direction": "short",
        "order_class": "stock_bracket",
        "allowed_order_types": {"bracket", "trailing_stop_exit", "oca_exit", "limit"},
        "description": "Short stock bracket — structured risk with bracket/trailing stop risk management",
    },
    3:  {
        "regime": "bearish",
        "direction": "short",
        "order_class": "options_directional",
        "allowed_order_types": {"vertical_spread", "diagonal_spread"},
        "description": "Bear options spread — defined-risk put debit or call credit vertical spreads",
    },
    4:  {
        "regime": "bearish",
        "direction": "neutral",
        "order_class": "options_premium",
        "allowed_order_types": {"iron_condor", "butterfly", "strangle", "straddle"},
        "description": "Bearish-leaning premium — iron condors or butterflies with downside skew",
    },
    # ── BULLISH GROUP (slots 5-8): activated when trend=up ──────
    5:  {
        "regime": "bullish",
        "direction": "long",
        "order_class": "options_directional",
        "allowed_order_types": {"vertical_spread", "diagonal_spread", "calendar_spread"},
        "description": "Bull options spread — call debit verticals, diagonals, or calendars",
    },
    6:  {
        "regime": "bullish",
        "direction": "long",
        "order_class": "stock_momentum",
        "allowed_order_types": {"market", "vwap", "stop_entry", "moo"},
        "description": "Long stock breakout — momentum entries on trend-following triggers",
    },
    7:  {
        "regime": "bullish",
        "direction": "long",
        "order_class": "stock_pullback",
        "allowed_order_types": {"limit", "midprice"},
        "description": "Long stock pullback — buy dips into support with limit orders",
    },
    8:  {
        "regime": "bullish",
        "direction": "neutral",
        "order_class": "options_premium",
        "allowed_order_types": {"iron_condor", "butterfly", "strangle", "straddle"},
        "description": "Bullish-leaning premium — iron condors or butterflies with upside skew",
    },
    # ── CHOPPY GROUP (slots 9-12): activated when trend=flat ────
    9:  {
        "regime": "choppy",
        "direction": "long",
        "order_class": "stock_mean_reversion",
        "allowed_order_types": {"limit", "midprice", "moc"},
        "description": "Mean reversion long — oversold bounce entries at support levels",
    },
    10: {
        "regime": "choppy",
        "direction": "short",
        "order_class": "stock_mean_reversion",
        "allowed_order_types": {"limit", "midprice", "moc"},
        "description": "Mean reversion short — overbought fade entries at resistance levels",
    },
    11: {
        "regime": "choppy",
        "direction": "neutral",
        "order_class": "options_premium",
        "allowed_order_types": {"iron_condor", "butterfly", "straddle", "strangle"},
        "description": "Neutral premium collection — iron condors or butterflies for range-bound markets",
    },
    12: {
        "regime": "choppy",
        "direction": "any",
        "order_class": "stock_range",
        "allowed_order_types": {"bracket", "stop_entry", "oca_exit", "trailing_stop_exit"},
        "description": "Dual-sided range trade — ORB or range breakout/breakdown with bracket orders",
    },
}

# Map trend_regime labels to slot regime groups
REGIME_TO_SLOTS: dict[str, list[int]] = {
    "down": [1, 2, 3, 4],     # bearish group
    "up":   [5, 6, 7, 8],     # bullish group
    "flat": [9, 10, 11, 12],  # choppy group
}

# ── Darwinian weighting ─────────────────────────────────────────
DARWINIAN_WEIGHT_FLOOR = 0.3    # minimum weight (near-silenced)
DARWINIAN_WEIGHT_CEILING = 2.5  # maximum weight (highly trusted)
DARWINIAN_WEIGHT_UP = 1.05      # top quartile boost per round
DARWINIAN_WEIGHT_DOWN = 0.95    # bottom quartile decay per round
SLOT_CORRELATION_THRESHOLD = 0.7  # r > this flags redundant pair

# ── Environment regime gates ────────────────────────────────────
REGIME_GATE_ENABLED = True       # hard gate: suppress mismatched directional strategies
EXTREME_VOL_FITNESS_BOOST = 1.05 # require 5% higher fitness in extreme vol

# ── Circuit breakers ────────────────────────────────────────────
# Research: pause evolution if ALL slots fail for this many consecutive rounds
CIRCUIT_BREAKER_ALL_FAIL_ROUNDS = 5
# Research: pause cooldown (seconds) when the all-fail breaker trips
CIRCUIT_BREAKER_COOLDOWN_SECS = 600  # 10 minutes
# Per-slot: cap consecutive failures before the slot is frozen (skipped)
# until the selector intervenes or a new round begins
CIRCUIT_BREAKER_SLOT_MAX_FAILURES = 10

# ── Pacing / economy ───────────────────────────────────────────
ROUND_DELAY_SECS = 30             # pause between rounds to reduce LLM cost burn
RESEARCH_DAILY_LLM_BUDGET = 25.0  # USD — stop research when daily LLM spend exceeds this

# ── Available order types (referenced in system prompt for Grok) ─
STOCK_ORDER_TYPES = [
    "market", "limit", "stop_entry", "bracket", "trailing_stop_exit",
    "oca_exit", "midprice", "vwap", "moc", "moo",
]

OPTIONS_STRATEGIES = [
    "vertical_spread", "iron_condor", "straddle", "strangle",
    "calendar_spread", "diagonal_spread", "butterfly",
]

# ── Canonical legs_json field contracts per strategy ────────────
# These are the REQUIRED fields for each options structure.
# The simulator and promotion engine use these for deterministic contract
# reconstruction from historical chains.  Any signal that omits required
# fields is rejected during validation rather than silently producing garbage.

LEGS_JSON_SCHEMAS: dict[str, dict] = {
    # Long or short vertical spread (debit/credit call or put spread)
    # right: 'C' or 'P'
    # long_strike/short_strike: the two strikes
    # direction on the parent signal governs debit vs. credit
    "vertical_spread": {
        "required": ["expiration", "long_strike", "short_strike", "right"],
        "optional": ["quantity"],
        "description": (
            "Two-leg spread. long_strike < short_strike for calls; "
            "long_strike > short_strike for puts is also fine. "
            "right: 'C' or 'P'."
        ),
    },
    # Short iron condor: sell OTM put spread + sell OTM call spread
    "iron_condor": {
        "required": [
            "expiration",
            "put_long_strike", "put_short_strike",
            "call_short_strike", "call_long_strike",
        ],
        "optional": ["quantity"],
        "description": (
            "Four legs. put_long < put_short <= call_short < call_long. "
            "Net credit strategy."
        ),
    },
    # Long straddle: buy ATM call + buy ATM put, same strike + expiration
    "straddle": {
        "required": ["expiration", "strike"],
        "optional": ["quantity"],
        "description": "Long ATM call + long ATM put. Single strike.",
    },
    # Long strangle: buy OTM call + buy OTM put, different strikes
    "strangle": {
        "required": ["expiration", "put_strike", "call_strike"],
        "optional": ["quantity"],
        "description": "Long OTM call + long OTM put. put_strike < call_strike.",
    },
    # Calendar spread: sell near-term, buy far-term, same strike + right
    "calendar_spread": {
        "required": ["near_expiration", "far_expiration", "strike", "right"],
        "optional": ["quantity"],
        "description": "Sell near_expiration, buy far_expiration at the same strike.",
    },
    # Diagonal spread: sell near-term near-strike, buy far-term different-strike
    "diagonal_spread": {
        "required": [
            "near_expiration", "far_expiration",
            "near_strike", "far_strike", "right",
        ],
        "optional": ["quantity"],
        "description": "Sell near_expiration/near_strike, buy far_expiration/far_strike.",
    },
    # Long butterfly: buy low + buy high, sell 2x middle
    "butterfly": {
        "required": ["expiration", "low_strike", "mid_strike", "high_strike", "right"],
        "optional": ["quantity"],
        "description": (
            "Three strikes. low_strike < mid_strike < high_strike; "
            "mid_strike should be equidistant from both wings."
        ),
    },
}


def validate_legs_json(legs: dict) -> str | None:
    """
    Validate a legs_json dict against the canonical schema.

    Returns an error message string if validation fails, or None if valid.
    Does NOT raise — callers decide how to handle failures.
    """
    if not isinstance(legs, dict):
        return "legs_json must be a dict"

    strategy = legs.get("strategy")
    if not strategy:
        return "legs_json missing 'strategy' key"

    schema = LEGS_JSON_SCHEMAS.get(strategy)
    if schema is None:
        valid = list(LEGS_JSON_SCHEMAS)
        return f"Unknown strategy '{strategy}'. Valid: {valid}"

    missing = [f for f in schema["required"] if f not in legs]
    if missing:
        return f"legs_json[{strategy}] missing required fields: {missing}"

    # Type-check numeric fields
    numeric_keys = [
        "long_strike", "short_strike", "put_long_strike", "put_short_strike",
        "call_short_strike", "call_long_strike", "strike", "put_strike",
        "call_strike", "near_strike", "far_strike",
        "low_strike", "mid_strike", "high_strike",
    ]
    for key in numeric_keys:
        if key in legs and not isinstance(legs[key], (int, float)):
            return f"legs_json field '{key}' must be a number, got {type(legs[key]).__name__}"

    # Strategy-specific sanity checks
    if strategy == "vertical_spread":
        r = legs.get("right", "")
        if r.upper() not in ("C", "P", "CALL", "PUT"):
            return f"vertical_spread 'right' must be C or P, got '{r}'"

    elif strategy == "iron_condor":
        pl, ps = legs.get("put_long_strike", 0), legs.get("put_short_strike", 0)
        cs, cl = legs.get("call_short_strike", 0), legs.get("call_long_strike", 0)
        if not (pl < ps <= cs < cl):
            return (
                f"iron_condor strikes must satisfy put_long < put_short <= "
                f"call_short < call_long, got {pl} {ps} {cs} {cl}"
            )

    elif strategy == "strangle":
        ps, cs = legs.get("put_strike", 0), legs.get("call_strike", 0)
        if ps >= cs:
            return f"strangle put_strike ({ps}) must be < call_strike ({cs})"

    elif strategy == "butterfly":
        lo, mid, hi = (
            legs.get("low_strike", 0),
            legs.get("mid_strike", 1),
            legs.get("high_strike", 2),
        )
        if not (lo < mid < hi):
            return f"butterfly strikes must be low < mid < high, got {lo} {mid} {hi}"

    return None  # valid


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
  legs_json: dict | None  — for options strategies only. Required fields by strategy:
      vertical_spread:  {strategy, expiration, long_strike, short_strike, right}
      iron_condor:      {strategy, expiration, put_long_strike, put_short_strike,
                         call_short_strike, call_long_strike}
      straddle:         {strategy, expiration, strike}
      strangle:         {strategy, expiration, put_strike, call_strike}
      calendar_spread:  {strategy, near_expiration, far_expiration, strike, right}
      diagonal_spread:  {strategy, near_expiration, far_expiration,
                         near_strike, far_strike, right}
      butterfly:        {strategy, expiration, low_strike, mid_strike, high_strike, right}
"""

# ── System prompt for strategy evolution (slot-aware + environment-aware) ──
RESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent evolving day trading strategies.

HUMAN PROGRAM:
{program_markdown}

YOUR JOB: Modify strategy.py to improve its expectancy (expected profit per trade).
You receive the current strategy code, its evaluation results, and history of past attempts.

You are evolving Slot {slot_id} (one of {num_slots} strategy slots).

SLOT MANDATE (HARD CONSTRAINT — you MUST obey this):
{slot_mandate}
You may freely change indicators, thresholds, filters, and parameters. But you MUST NOT
change the direction or order type class. Proposals that violate the mandate are automatically
rejected. Work within your mandate to produce the best possible strategy.

{environment_context}

Available stock order types: market, limit, stop_entry, bracket, trailing_stop_exit,
  oca_exit, midprice, vwap, moc, moo
Available options strategies: vertical_spread, iron_condor, straddle, strangle,
  calendar_spread, diagonal_spread, butterfly
You may use long calls, long puts, short puts, short calls, or any combination.

RULES:
1. Output the COMPLETE new strategy.py file. Not a diff, not a snippet — the full file.
2. The file must define: scan(candles: pd.DataFrame, symbol: str, env: dict = None) -> list[dict]
   - 'env' is an optional dict with regime data (see keys below). Old 2-arg signatures also work.
3. candles has columns: ts, open, high, low, close, volume (1-min bars, one trading day)
4. env keys (all optional): volatility_regime ("low"/"normal"/"high"),
   trend_regime ("down"/"flat"/"up"), breadth_regime ("bearish"/"neutral"/"bullish"),
   momentum_regime, volume_regime, dispersion, strategy_fit (dict).
   Use env to adapt thresholds per regime instead of hardcoding.
5. Each returned signal must follow this schema:
{signal_schema}
6. You may only import: pandas, numpy, math, statistics. No other imports.
7. No file I/O, no network calls, no os/sys/subprocess.
8. For options strategies, set order_type to the strategy name and include legs_json.

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

# Inject program_markdown at load time (other {placeholders} remain for runtime .format())
RESEARCH_SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT.replace(
    "{program_markdown}",
    RESEARCH_PROGRAM_MARKDOWN or "(No program.md found. Use the repository defaults.)",
)

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
- 95% CI for mean return: [{ci95_low:+.4f}, {ci95_high:+.4f}]
- Confidence score: {confidence_score:.2f} (higher = less likely luck)
- Luck pressure: {luck_pressure:.2f} (higher = more likely luck/noise)
- Condition confidence: {condition_confidence:.2f}

CONDITION SUMMARY:
{condition_summary}

Analyze:
1. What's working and why? How does this relate to the current market environment?
2. What's failing and why? Is it an environment mismatch?
3. Time-of-day patterns (are morning entries better than afternoon?)
4. Which conditions look genuinely repeatable vs. likely lucky?
5. Which setup types perform best in THIS environment?
6. Are stops too tight or too loose given current volatility regime?
7. Specific, actionable suggestions for the next strategy modification that
   ACCOUNT FOR the current market environment.

Be concise and specific. Focus on patterns in the data, not general advice."""

# ── Selector prompt (meta-learning agent) ────────────────────────
SELECTOR_PROMPT = """You are a META-LEARNING strategy portfolio manager overseeing {num_slots}
independent strategy slots for day trading. Your job is to MAXIMIZE portfolio-level
profitability by intelligently allocating slots to strategies that match the current
market environment.

═══ CURRENT MARKET ENVIRONMENT ═══
{environment_context}

═══ SLOT MANDATES (HARD CONSTRAINTS) ═══
Each slot has a fixed regime and order-type mandate. Your directives MUST respect
these — do NOT ask a slot to pivot to a strategy type outside its allowed set.
{slot_mandates}

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
      "target_strategy_type": "what type the new strategy should be (MUST be within the slot's mandate)"
    }}
  ],
  "environment_learning": "Key insight about what works in this regime",
  "recommended_focus": ["list of strategy types to prioritize in current environment"]
}}

RULES:
- You may act on MULTIPLE slots, not just the worst one.
- Maximum {max_replacements} replacements per selector run to maintain stability.
- Never replace a slot with < 3 iterations unless it has critical issues.
- Do NOT provide seed_code — the evolution pipeline will generate proper code.
- If seed_from_slot is set, the donor's code will be cloned as a starting point
  (only if it evaluates better than the current slot AND both slots share the same regime).
- Only seed_from_slot within the SAME regime group (bearish→bearish, bullish→bullish, choppy→choppy).
- Prioritize strategies from recommended_focus that match the environment.
- Keep at least 2 slots running strategies from different archetypes for diversity.
- target_strategy_type MUST be compatible with the slot's allowed_order_types in its mandate.
"""
