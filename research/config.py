"""
Research Configuration — Universe, timing, signal combination, and option schemas.
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
FORCE_EXIT_MINUTE = 955         # 15:55 ET — force exit 5 min before close (minute of day)
SLIPPAGE_BPS = 2                # 2 basis points per trade (slippage + commission)

# ── Pacing / economy ───────────────────────────────────────────
ROUND_DELAY_SECS = 30             # pause between rounds to reduce LLM cost burn

# ── Signal combination engine ──────────────────────────────────
MIN_SHARED_PERIODS_FOR_COMBINATION = 10   # min shared periods before combiner can compute weights
SIGNAL_WEIGHT_LOOKBACK_DAYS = 10          # d in Step 8: recent periods for expected return
COMPOSITE_TRADE_THRESHOLD = 0.25          # minimum |composite| to generate a trade
CV_BOOTSTRAP_SAMPLES = 1000              # bootstrap iterations for CV estimation
TEMPLATE_EVOLUTION_TRAIN_PCT = 0.70       # walk-forward train fraction
TEMPLATE_EVOLUTION_MIN_TRADES = 30        # min trades per template for stable metrics

# Tiered scoring: wide cheap scan → narrow expensive scan → trade recs
TIER1_UNIVERSE_SIZE = 150        # Wide scan: bulk/cached signals only (no option chains)
DEEP_SCAN_TOP_N = 20             # Tier 2: top N from Tier 1 get full 50 signals + option chains
TRADE_REC_TOP_N = 8              # Tier 3: top N from Tier 2 get template selection

# Template evolution scheduling
EVOLUTION_COOLDOWN_MARKET_HOURS = 1800   # 30 min between evolution rounds during market
EVOLUTION_COOLDOWN_OFF_HOURS = 300       # 5 min between rounds outside market hours

# Per-category forward-return horizons (bars at 5-min resolution)
FORWARD_RETURN_HORIZON = {
    "price": 12,           # ~1 hour at 5min bars (intraday signals resolve fast)
    "volatility": 60,      # ~5 hours (IV mean-reverts over sessions)
    "fundamental": 390,    # ~3 trading days (earnings impact plays out over days)
    "macro": 78,           # ~1 trading day (event reactions unfold in a session)
    "microstructure": 12,  # ~1 hour (flow signals are short-lived)
}

# API budget configuration
MAX_CREDITS_PER_ROUND = 200         # Circuit breaker
OPTION_CHAIN_DTE_RANGE = (7, 60)    # Limit DTE to reduce response size
OPTION_CHAIN_STRIKE_LIMIT = 20      # Max strikes per expiration (server-side filter)

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
