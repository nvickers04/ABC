"""
Research Configuration — Universe, timing, tracks, system prompt, sandbox rules.

The research agent evolves day trading strategies across independent tracks,
one per order type. Each track has its own strategy file, fitness history,
and evolution state.
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

# ── Available order types (referenced in system prompt for Grok) ─
STOCK_ORDER_TYPES = [
    "market", "limit", "stop_entry", "bracket", "trailing_stop_exit",
    "oca_exit", "midprice", "vwap", "moc", "moo",
]

OPTIONS_STRATEGIES = [
    "vertical_spread", "iron_condor", "straddle", "strangle",
    "calendar_spread", "diagonal_spread", "butterfly",
]

# ── Research tracks — one per order type ─────────────────────────
# Each track evolves independently with its own strategy file and fitness.
TRACKS = [
    {
        "name": "market",
        "order_type": "market",
        "description": "Market order entries — momentum, breakouts, trend following.",
    },
    {
        "name": "limit",
        "order_type": "limit",
        "description": "Limit order entries — pullbacks, mean reversion, support bounces.",
    },
    {
        "name": "stop_entry",
        "order_type": "stop_entry",
        "description": "Stop order entries — breakout triggers above resistance / below support.",
    },
    {
        "name": "bracket",
        "order_type": "bracket",
        "description": "Bracket orders — entry with automatic stop-loss and take-profit.",
    },
    {
        "name": "vwap",
        "order_type": "vwap",
        "description": "VWAP algo entries — mean reversion to VWAP, VWAP bounce, VWAP trend.",
    },
    {
        "name": "moo",
        "order_type": "moo",
        "description": "Market on open entries — gap plays, opening range strategies.",
    },
    {
        "name": "moc",
        "order_type": "moc",
        "description": "Market on close entries — end-of-day momentum, closing imbalance.",
    },
    {
        "name": "vertical_spread",
        "order_type": "vertical_spread",
        "description": "Vertical spread entries — defined-risk directional options.",
    },
    {
        "name": "straddle",
        "order_type": "straddle",
        "description": "Long straddle/strangle entries — volatility expansion plays.",
    },
    {
        "name": "iron_condor",
        "order_type": "iron_condor",
        "description": "Iron condor entries — premium selling on range-bound names.",
    },
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

# ── System prompt for strategy evolution (track-aware) ───────────
# Formatted at runtime with track-specific context.
RESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent evolving day trading strategies.

YOUR JOB: Modify strategy.py to improve its expectancy (expected profit per trade).
You receive the current strategy code, its evaluation results, and history of past attempts.

TRACK CONSTRAINT:
This track uses **{track_order_type}** order type. {track_description}
Every signal MUST set order_type = "{track_order_type}".
Do NOT produce signals with any other order_type.

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
- Options strategies are supported but use delta/theta P&L approximations.
  Prefer simple structures (verticals, straddles) over complex multi-leg strategies.

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
