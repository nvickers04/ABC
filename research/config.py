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

# ── System prompt for strategy evolution (slot-aware) ────────────
RESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent evolving day trading strategies.

YOUR JOB: Modify strategy.py to improve its expectancy (expected profit per trade).
You receive the current strategy code, its evaluation results, and history of past attempts.

You are evolving Slot {slot_id} (one of {num_slots} independent strategy slots).
Each slot is free to use ANY order type or options structure — whatever produces
the best risk-adjusted returns.

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

STRATEGY EVOLUTION GUIDELINES:
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
1. What's working and why?
2. What's failing and why?
3. Time-of-day patterns (are morning entries better than afternoon?)
4. Which setup types perform best?
5. Are stops too tight or too loose?
6. Specific, actionable suggestions for the next strategy modification.

Be concise and specific. Focus on patterns in the data, not general advice."""

# ── Selector prompt (replaces weakest slot) ──────────────────────
SELECTOR_PROMPT = """You are a strategy portfolio manager overseeing {num_slots} independent
strategy slots for day trading.

SLOT PERFORMANCE (ranked by fitness, best first):
{slot_rankings}

YOUR JOB: Decide what to do with the weakest slot (Slot {worst_slot}).
The weakest slot has fitness={worst_fitness:.4f} after {worst_iterations} iterations.

OPTIONS:
1. "mutate_best" — Seed the worst slot with a mutated copy of the best-performing
   strategy. The evolution loop will differentiate it from there.
2. "fresh" — Generate an entirely new strategy for the worst slot using a novel
   approach that none of the current slots are using.
3. "keep" — Leave it alone (if it's still improving, or too early to judge).

Current strategy types in use:
{current_types}

Respond with ONLY a JSON object:
{{"action": "mutate_best"|"fresh"|"keep", "reasoning": "...", "seed_code": "..." }}

If action is "fresh", include complete Python code for the new strategy in seed_code.
If action is "mutate_best", include the mutated code in seed_code.
If action is "keep", seed_code can be null.
"""
