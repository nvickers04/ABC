# ABC — TODO

## Signal Combination Engine + Execution Template Architecture

### Problem Statement
The current 12-slot system conflates two questions: "what direction?" and "how to structure the trade?" All 12 slots answer both from the same OHLCV candle data, producing correlated opinions that masquerade as independent signals. Darwinian weights influence which slots evolve but NOT which signals the agent trades — the briefing sorts by raw `s.expectancy DESC` and the agent never sees slot weights, correlation data, or env-adjusted fitness. Effective N_eff is likely 2-3, not 12.

The Fundamental Law of Active Management: IR = IC × √N. N is the effective number of independent signals after shared variance is removed. The fix is structural: build genuinely independent signals from separate data sources, combine them mathematically into a composite score per symbol, then select execution templates mechanically based on the composite + market conditions.

### Architecture Overview
```
┌──────────────────────────────────────────────────────────────┐
│  SIGNAL LAYER  (signals/)                                    │
│  50 signals from 5 structurally independent data sources     │
│  Each signal: {symbol, score: -1.0 to +1.0, confidence: 0-1}│
│                                                              │
│  PRICE SIGNALS (13) — MDA candles (OHLCV multi-resolution)   │
│  VOLATILITY SIGNALS (10) — MDA option chains + IV endpoints  │
│  FUNDAMENTAL SIGNALS (12) — yfinance fundamentals/insider    │
│  MACRO/SENTIMENT SIGNALS (10) — environment + calendar + news│
│  MICROSTRUCTURE SIGNALS (5) — MDA quotes + option flow       │
│                                                              │
│  TIERED SCORING (100-200 symbol universe):                   │
│    Tier 1: Wide scan — 40 cheap signals (bulk/cached data)   │
│    Tier 2: Deep scan — top 20 get full 50 signals + options  │
│    Tier 3: Trade recs — top 8 get template selection         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  COMBINATION ENGINE  (signals/combiner.py)                   │
│  Article's 11-step alpha combination procedure               │
│                                                              │
│  Core input: R(i,s) = score(i,s) × forward_return(s→s+h)   │
│  This measures whether the signal was RIGHT, not just what   │
│  it said. Without this, you combine opinions, not edges.     │
│                                                              │
│  Steps 1-11: demean → normalize → cross-sectionally demean  │
│  → drop lookahead periods → forward expected return          │
│  → regress out shared variance → weight by independent edge  │
│  → normalize                                                 │
│                                                              │
│  Output: w(i) per signal, composite_score per symbol,        │
│          N_eff from eigenvalue decomposition                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  EXECUTION TEMPLATE SELECTOR  (signals/templates.py)         │
│  10 templates covering all order types                       │
│  Decision boundaries are LEARNED, not hardcoded              │
│  Evolves CONTINUOUSLY via walk-forward simulation            │
│  (30min cooldown market hours, 5min overnight)               │
│                                                              │
│  Input: composite_score, iv_rank, vol_regime, atr_pct,      │
│         breadth_regime, account_state                         │
│  Output: structured trade dict (same format as old signals)  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYERED BRIEFING → AGENT                                    │
│                                                              │
│  Layer 1: SIGNAL QUALITY (N_eff, weight leaders, redundancy) │
│  Layer 2: COMPOSITE SCORES (per symbol with breakdown)       │
│  Layer 3: RECOMMENDED TRADES (template-selected, with track  │
│           record per template per regime)                     │
│  Layer 4: TEMPLATE PERFORMANCE (last 30 days by regime)      │
│                                                              │
│  AGENT ROLE = execution quality layer:                       │
│    - Cannot override direction (composite resolved that)     │
│    - Can reject a recommendation (capacity, timing, spread)  │
│    - Handles: order routing, sizing, fill management         │
│    - Judges: is this the right moment to execute?            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 0: Foundation — signals/ Module + DB Schema + Data Layer Fixes
- [ ] Fix DataProvider method gaps (prerequisite for all signal phases):
  - Add `get_candles_bulk(symbols)` wrapper on DataProvider that delegates to
    `MarketDataClient.get_bulk_daily_candles(symbols)` — currently not exposed.
  - Note: bulk quotes method is `get_quotes_bulk()` (not `get_bulk_quotes()`)
  - Note: IV info method is `get_iv_info(symbol, dte_min, dte_max)` (not `get_iv_rank()`)
    and requires explicit DTE args — returns `None` without them.
  - Fix IV rank computation: `IVInfo.iv_rank` is currently hardcoded to `None` in
    marketdata_client.py. Fix by computing rank from `iv_high`/`iv_low`:
    `iv_rank = (iv_current - iv_low) / (iv_high - iv_low) * 100`
    Or add `iv_history` table to store daily `iv_current` snapshots and compute
    rank as percentile vs trailing 252 days (more accurate).
- [ ] Create `signals/__init__.py` — module init, signal registry
- [ ] Create `signals/base.py` — `Signal` base class interface:
  ```python
  class Signal:
      name: str           # e.g. "momentum"
      category: str       # "price", "volatility", "fundamental", "macro", "microstructure"
      data_source: str    # "mda_candles", "mda_options", "yfinance", "environment", "mda_quotes"
      refresh_rate: str   # "every_round", "5min", "10min", "daily"

      def score(self, symbol: str, data: dict) -> dict:
          """Returns {score: float [-1,1], confidence: float [0,1], components: dict}"""
  ```
- [ ] Create DB tables:
  - `signal_scores` — (signal_name, symbol, ts, score, confidence, components_json)
  - `signal_weights` — (signal_name, weight, n_eff, updated_ts)
  - `signal_returns` — (signal_name, symbol, ts, score_at_entry, forward_return, R_value)
  - `composite_scores` — (symbol, ts, composite_score, signal_breakdown_json)
  - `template_performance` — (template_name, regime_key, composite_bucket, trades, wins, avg_return_pct, avg_gap_pct, updated_ts)
  - `template_boundaries` — (template_name, param_name, param_value, generation, fitness, updated_ts)
  - `template_recommendations` — (symbol, ts, template_name, direction, composite_score, order_type, entry_price, target_price, stop_price, legs_json)
- [ ] DB migration approach: add new CREATE TABLE IF NOT EXISTS statements to
  `memory/__init__.py` → `init_db()`. No migration framework — same pattern as
  existing 17 tables. New tables are additive; old tables kept until Phase 10.
  Must call `init_db()` before any signal code runs (already called at startup).

- [ ] Add constants to `research/config.py`:
  - `MIN_SHARED_PERIODS_FOR_COMBINATION = 10`
  - `SIGNAL_WEIGHT_LOOKBACK_DAYS = 10`
  - `COMPOSITE_TRADE_THRESHOLD = 0.25` (minimum |composite| to generate a trade)
  - `CV_BOOTSTRAP_SAMPLES = 1000`
  - `TEMPLATE_EVOLUTION_TRAIN_PCT = 0.70`
  - `TEMPLATE_EVOLUTION_MIN_TRADES = 30`
  - Tiered scoring configuration:
    ```python
    # Tiered scoring: wide cheap scan → narrow expensive scan → trade recs
    TIER1_UNIVERSE_SIZE = 150       # Wide scan: bulk/cached signals only (no option chains)
    DEEP_SCAN_TOP_N = 20            # Tier 2: top N from Tier 1 get full 50 signals + option chains
    TRADE_REC_TOP_N = 8             # Tier 3: top N from Tier 2 get template selection
    TIER1_SIGNALS = 40              # Signals usable without option chains (price+fund+macro+micro)
    ```
  - Template evolution scheduling:
    ```python
    EVOLUTION_COOLDOWN_MARKET_HOURS = 1800   # 30 min between evolution rounds during market
    EVOLUTION_COOLDOWN_OFF_HOURS = 300       # 5 min between rounds outside market hours
    ```
  - Per-category forward-return horizons (different data sources move at different speeds):
    ```python
    FORWARD_RETURN_HORIZON = {
        "price": 12,           # ~1 hour at 5min bars (intraday signals resolve fast)
        "volatility": 60,      # ~5 hours (IV mean-reverts over sessions)
        "fundamental": 390,    # ~3 trading days (earnings impact plays out over days)
        "macro": 78,           # ~1 trading day (event reactions unfold in a session)
        "microstructure": 12,  # ~1 hour (flow signals are short-lived)
    }
    ```
    A single FORWARD_RETURN_HORIZON_BARS = 60 would penalize price signals (resolved
    before 60 bars) and undervalue fundamental signals (still resolving at 60 bars).
    The combiner normalizes R(i,s) (Step 4), so different horizons across categories
    produce comparable scales after normalization.

- [ ] Add API budget configuration to `research/config.py`:
  ```python
  # MDA option chain pricing (from docs):
  #   Real-time: 1 credit PER option symbol in response
  #   Cached mode: 1 credit PER API call (regardless of symbols returned)
  #   Historical: 1 credit per 1000 option symbols
  # For signal scoring, use cached mode (1 credit/call) or historical mode.
  # Real-time full chains are prohibitively expensive (AAPL ≈ 200+ strikes = 200+ credits).
  #
  # TIERED SCORING BUDGET per round (150 symbols → 20 deep → 8 recs):
  #
  #   Tier 1 — Wide scan (150 symbols, 40 signals, NO option chains):
  #     Candles bulk:   2 calls (quotes + daily)              =  2 credits
  #     News:           150 calls (120s TTL, ~50% cached)     = ~75 credits
  #     Fundamentals:   yfinance (no MDA credits)             =  0 credits
  #     Quotes bulk:    2 calls                               =  2 credits
  #     Tier 1 total:   ~80 credits, ~3s compute
  #
  #   Tier 2 — Deep scan (top 20 symbols, add 10 volatility signals):
  #     Option chains:  20 calls cached mode                  = 20 credits
  #     IV rank:        20 calls                              = 20 credits
  #     Tier 2 total:   ~40 credits, ~14s (option chain bottleneck)
  #
  #   Tier 3 — Trade recs (top 8 symbols):
  #     Template selection: pure computation                  =  0 credits
  #
  #   ROUND TOTAL: ~120 credits, ~20s elapsed
  #   At 2 rounds/min: ~240 credits/hour
  #   At 6.5 market hours: ~1,560 credits/day
  #
  #   vs OLD flat scan (25 symbols, all signals): ~80 credits, ~18s
  #   NEW gets 6x symbol coverage at ~1.5x the cost.
  #
  OPTION_CHAIN_MODE = "cached"    # "cached" (1 credit/call) vs "realtime" (1 credit/symbol)
  MAX_CREDITS_PER_ROUND = 200     # Circuit breaker (higher for 150-symbol universe)
  OPTION_CHAIN_DTE_RANGE = (7, 60)  # Limit DTE to reduce response size
  OPTION_CHAIN_STRIKE_LIMIT = 20    # Max strikes per expiration (server-side filter)
  ```

### Phase 1: Price Signals (13 signals)
Source: MDA candles (OHLCV, multiple resolutions). The largest category because price data is the richest single source, but the combination engine will correctly measure within-category correlation and downweight redundancy.

- [ ] `signals/momentum.py` — Signal 1: Trend/momentum
  - EMA(10) vs EMA(50) cross direction and distance
  - 5-bar and 10-bar ROC
  - MACD histogram sign and slope
  - Score: +1.0 (strong uptrend) to -1.0 (strong downtrend)

- [ ] `signals/mean_reversion.py` — Signal 2: Mean reversion
  - RSI(14) distance from 50 (oversold < 30, overbought > 70)
  - Bollinger Band position (% within bands)
  - Distance from 20-day SMA in ATR units
  - Score: +1.0 (deeply oversold = long setup) to -1.0 (deeply overbought)

- [ ] `signals/breakout.py` — Signal 3: Breakout
  - Distance from 20-bar high/low
  - Volume confirmation (current vs 20-bar average)
  - Range expansion (today's range vs 5-day avg range)
  - Score: +1.0 (breaking above on volume) to -1.0 (breaking below on volume)

- [ ] `signals/vwap.py` — Signal 4: VWAP deviation
  - Price distance from intraday VWAP in ATR units
  - Direction of VWAP slope
  - Score: +1.0 (far above VWAP, momentum) to -1.0 (far below)

- [ ] `signals/volume.py` — Signal 5: Volume profile
  - Relative volume (current bar vs 20-bar avg)
  - Volume trend (expanding vs contracting over 5 bars)
  - On-balance volume direction
  - Score: +1.0 (strong accumulation) to -1.0 (strong distribution)

- [ ] `signals/beta_adjusted_momentum.py` — Signal 6: Beta-adjusted momentum
  - Symbol return minus (beta x SPY return) over 5, 10, 20 bars
  - Isolates idiosyncratic alpha from market exposure
  - Source: yfinance beta + MDA candles for symbol + SPY
  - Score: +1.0 (strong positive alpha) to -1.0 (negative alpha beyond market)

- [ ] `signals/gap.py` — Signal 7: Overnight gap
  - Today's open vs yesterday's close (gap %)
  - Gap direction + partial fill assessment intraday
  - Score: +1.0 (gap up, holding = bullish) to -1.0 (gap down, holding = bearish)
  - Small gaps near 0 = low confidence

- [ ] `signals/volume_weighted_momentum.py` — Signal 8: Volume-weighted momentum
  - Price change x relative volume (weights moves on heavy volume higher)
  - Independently valuable because a +2% move on 3x volume means more than on 0.5x volume
  - Score: +1.0 (strong price advance on heavy volume) to -1.0 (strong decline on heavy volume)

- [ ] `signals/multi_timeframe.py` — Signal 9: Multi-timeframe alignment
  - Trend direction on daily, hourly, and 5-min candles
  - Score: +1.0 (all three bullish) to -1.0 (all three bearish), 0 when mixed
  - Confidence: proportional to alignment (3/3 = high, 2/3 = moderate, 1/3 = low)

- [ ] `signals/price_acceleration.py` — Signal 10: Price acceleration
  - ROC of ROC (second derivative of price)
  - Detects momentum building (accelerating) vs fading (decelerating)
  - Score: +1.0 (accelerating to upside) to -1.0 (accelerating to downside)

- [ ] `signals/trend_strength.py` — Signal 11: Trend strength
  - Directional movement index (DMI) — +DI vs -DI
  - ADX-style measure from candle data (strength regardless of direction)
  - Score: +1.0 (strong trending) to 0.0 (choppy), sign from direction
  - Confidence: ADX magnitude

- [ ] `signals/support_resistance.py` — Signal 12: Support/resistance proximity
  - Distance to nearest swing high and swing low (20-bar lookback)
  - Near support = bullish (bounce potential), near resistance = bearish
  - Score: +1.0 (sitting on strong support) to -1.0 (pressing against resistance)

- [ ] `signals/opening_range.py` — Signal 13: Opening range position
  - Price position vs first 30-min high/low range
  - Above opening range = bullish breakout, below = bearish breakdown
  - Score: +1.0 (above opening range, trending up) to -1.0 (below, trending down)
  - Only active during regular session

### Phase 2: Volatility Signals (10 signals)
Source: MDA option chains + IV rank endpoint. Structurally independent from price data — measures the options market's expectations, not price movement itself.

- [ ] `signals/iv_rv_spread.py` — Signal 14: IV vs realized vol spread
  - IV from get_iv_rank() vs realized vol from 20-day daily candle returns
  - Spread = IV - RV (positive = options overpriced = sell premium edge)
  - Score: +1.0 (IV >> RV, sell premium) to -1.0 (IV << RV, buy options)

- [ ] `signals/iv_rank.py` — Signal 15: IV rank
  - IV percentile (0-100) from get_iv_rank()
  - Score: +1.0 (rank 90+ = premium-selling opportunity) to -1.0 (rank 10- = cheap options)

- [ ] `signals/skew.py` — Signal 16: Volatility skew
  - Put ATM IV vs Call ATM IV from option chain
  - Steepening skew = fear = bearish positioning
  - Score: +1.0 (call skew = bullish flow) to -1.0 (put skew = bearish/hedge flow)

- [ ] `signals/term_structure.py` — Signal 17: IV term structure
  - Front-month IV vs second-month IV (two expirations from option chain)
  - Backwardation (front > back) = stress/event
  - Score: +1.0 (steep contango = calm) to -1.0 (backwardation = stress)

- [ ] `signals/iv_change.py` — Signal 18: IV rank momentum
  - Change in IV rank over trailing 5 days (requires storing iv_rank history)
  - Rising IV rank = increasing fear / event approaching
  - Score: +1.0 (IV rank falling fast = fear dissipating) to -1.0 (IV rank spiking)

- [ ] `signals/realized_vol_cone.py` — Signal 19: Realized vol percentile
  - Current 20-day RV vs 1-year historical RV distribution
  - Score: +1.0 (RV at historical lows = calm, options cheap for buyers) to -1.0 (RV at highs = volatile, risky)

- [ ] `signals/straddle_cost.py` — Signal 20: ATM straddle cost
  - ATM straddle mid price / underlying price = market's expected move %
  - When expected move is large relative to actual recent moves: premium selling edge
  - Score: +1.0 (straddle cheap vs realized = buy vol) to -1.0 (straddle expensive = sell vol)

- [ ] `signals/option_volume_ratio.py` — Signal 21: Option volume surge
  - Today's total option volume vs 20-day average option volume
  - High option volume = informed trading, event anticipation
  - Score: magnitude + direction bias (call-heavy volume = bullish, put-heavy = bearish)

- [ ] `signals/put_call_ratio.py` — Signal 22: Put/call ratio
  - Total put OI / total call OI from option chain
  - Contrarian: extreme put/call = oversold (bullish), extreme call/put = overbought
  - Score: +1.0 (extreme put ratio = contrarian bullish) to -1.0 (extreme call ratio)

- [ ] `signals/gamma_exposure.py` — Signal 23: Gamma exposure estimate
  - Sum(gamma x OI x 100 x sign) across strikes from option chain
  - Positive aggregate gamma = market maker short gamma = amplifies moves
  - Score: directional bias from gamma positioning and hedging flow direction

### Phase 3: Fundamental Signals (12 signals)
Source: yfinance fundamentals, analyst, insider data. Entirely independent from price/vol — measures company quality, institutional behavior, and valuation. Updates slowly (300-600s TTL).

- [ ] `signals/earnings.py` — Signal 24: Earnings momentum
  - Last earnings surprise % (surprise_eps_pct from MDA)
  - Days until next earnings (proximity risk: dampen near earnings)
  - Use: `data_provider.get_earnings_info(symbol)` → `EarningsInfo` with
    `next_earnings_date`, `days_until_earnings`
  - Use: `data_provider.get_earnings_history(symbol, countback=8)` → EPS surprise records
  - Score: +1.0 (large positive surprise, not near next earnings) to -1.0 (large miss)

- [ ] `signals/valuation.py` — Signal 25: Relative valuation
  - P/E, P/S, PEG ratio vs sector median (extended_fundamentals + peer_comparison)
  - Score: +1.0 (significantly undervalued) to -1.0 (significantly overvalued)

- [ ] `signals/quality.py` — Signal 26: Company quality
  - ROE, debt_to_equity, profit_margin, free_cash_flow from extended_fundamentals
  - Score: +1.0 (high quality across all metrics) to -1.0 (poor quality)

- [ ] `signals/short_interest.py` — Signal 27: Short interest pressure
  - Short % float, short ratio (days to cover) from extended_fundamentals
  - Score: +1.0 (high short interest = squeeze potential) to -1.0 (low short interest)

- [ ] `signals/insider.py` — Signal 28: Insider flow
  - Net insider buy/sell $ value from insider_data
  - Recent buy/sell count and net_sentiment
  - Score: +1.0 (heavy insider buying) to -1.0 (heavy insider selling)

- [ ] `signals/analyst.py` — Signal 29: Analyst consensus
  - Mean recommendation (1-5), recent upgrades vs downgrades, target upside %
  - Score: +1.0 (strong buy + large upside) to -1.0 (sell consensus + downside)

- [ ] `signals/revenue_growth.py` — Signal 30: Revenue growth
  - revenue_growth from extended_fundamentals
  - Score: +1.0 (strong growth) to -1.0 (declining revenue)

- [ ] `signals/cash_flow_yield.py` — Signal 31: Cash flow yield
  - free_cash_flow / market_cap from extended_fundamentals
  - High FCF yield = cash generation undervalued by market
  - Score: +1.0 (high yield = undervalued cash flow) to -1.0 (negative/low FCF)

- [ ] `signals/carry.py` — Signal 32: Dividend carry
  - dividend_yield from extended_fundamentals
  - Score: +1.0 (high yield = income + value signal) to -1.0 (no yield when peers have yield)

- [ ] `signals/size_factor.py` — Signal 33: Size factor
  - market_cap relative to universe median
  - Classic Fama-French SMB factor: smaller companies have higher expected returns
  - Score: +1.0 (small cap relative to universe) to -1.0 (large cap)

- [ ] `signals/debt_health.py` — Signal 34: Debt health
  - debt_to_equity, current_ratio, quick_ratio from extended_fundamentals
  - Score: +1.0 (low leverage, strong liquidity) to -1.0 (high leverage, weak liquidity)

- [ ] `signals/institutional.py` — Signal 35: Institutional ownership
  - institutional_pct from institutional_data
  - Direction of ownership (accumulation vs distribution across holders)
  - Score: +1.0 (high institutional ownership + accumulating) to -1.0 (low/declining)

### Phase 4: Macro/Sentiment Signals (10 signals)
Source: environment.py metrics, economic_calendar, MDA news, SPY/QQQ candles. Measures market-wide conditions and cross-asset relationships.

- [ ] `signals/breadth.py` — Signal 36: Market breadth
  - Advance/decline ratio, % trending up vs down from compute_environment()
  - Score: +1.0 (broad bullish breadth) to -1.0 (broad bearish breadth)

- [ ] `signals/sector_momentum.py` — Signal 37: Sector relative strength
  - Symbol return vs sector ETF return over 5, 10, 20 days
  - Use: `data_provider.get_peer_comparison(symbol)` → `PeerComparison` with sector ETF
  - Score: +1.0 (outperforming sector) to -1.0 (underperforming sector)

- [ ] `signals/correlation_regime.py` — Signal 38: Correlation regime
  - Cross-asset correlation from compute_environment()
  - Score: +1.0 (idiosyncratic regime = stock-picking works) to -1.0 (macro regime = defer to market)

- [ ] `signals/event_proximity.py` — Signal 39: Economic event risk
  - Distance in hours to next high-impact macro event (FOMC, CPI, NFP, GDP)
  - Score: 0 (far from event) to -1.0 (within 24h of high-impact event = reduce risk)

- [ ] `signals/news_sentiment.py` — Signal 40: News sentiment
  - Aggregate sentiment from get_news() (pos/neg/neutral, recency-weighted)
  - Score: +1.0 (overwhelmingly positive) to -1.0 (overwhelmingly negative)

- [ ] `signals/vix_proxy.py` — Signal 41: VIX / fear gauge
  - VIX is NOT directly available from MDA. Fallback chain (from existing
    `_vix_fallback()` in tools_research.py):
    1. UVXY quote (VIX futures proxy)
    2. SPY option IV (ATM IV from get_iv_rank() with dte 20-40)
    3. SPY ATR proxy (annualized ATR% ≈ daily_atr/price × √252 × 100)
  - All three already implemented and working. Signal should try in order,
    record which source was used, lower confidence for ATR proxy.
  - Score: +1.0 (low VIX = calm = bullish backdrop) to -1.0 (high VIX = fear)

- [ ] `signals/relative_strength_market.py` — Signal 42: Relative strength vs broad market
  - Symbol return vs SPY/QQQ return over 5, 10, 20 days
  - Different from sector momentum — measures stock vs broad market
  - Score: +1.0 (outperforming market) to -1.0 (underperforming market)

- [ ] `signals/seasonality.py` — Signal 43: Calendar seasonality
  - Day of week effects (e.g., Monday reversal, Friday positioning)
  - Monthly OPEX proximity (options expiration gamma effects)
  - Month-of-year patterns
  - Score: +1.0 (historically bullish period) to -1.0 (bearish period)

- [ ] `signals/regime_persistence.py` — Signal 44: Regime persistence
  - How many consecutive periods the current vol/trend regime has lasted
  - Long-lasting regimes mean-revert; fresh regime changes tend to persist
  - Score: +1.0 (fresh regime = trade with trend) to -1.0 (extended regime = reversal risk)

- [ ] `signals/market_momentum.py` — Signal 45: Market-level momentum
  - SPY/QQQ trend direction and strength (EMA cross, ROC)
  - Rising tide lifts all boats; sinking tide drowns them
  - Score: +1.0 (market trending up strongly) to -1.0 (market trending down)

### Phase 5: Microstructure Signals (5 signals)
Source: MDA quotes + option chain OI. Measures positioning, liquidity, and flow information.

- [ ] `signals/spread_dynamics.py` — Signal 46: Bid-ask spread dynamics
  - Current spread vs trailing average spread
  - Narrowing = safe, widening = informed trading/caution
  - Score: +1.0 (tightening) to -1.0 (widening)

- [ ] `signals/option_flow.py` — Signal 47: Option OI flow
  - Call OI vs Put OI change (compare current snapshot to prior)
  - Score: +1.0 (call OI building) to -1.0 (put OI building)

- [ ] `signals/quote_stability.py` — Signal 48: Quote stability
  - Bid/ask mid-price variance over trailing period from quote snapshots
  - Low variance = orderly market, high variance = instability
  - Score: +1.0 (stable quotes) to -1.0 (volatile quotes = caution)

- [ ] `signals/volume_clock.py` — Signal 49: Intraday volume acceleration
  - Cumulative volume vs expected cumulative volume (historical intraday profile)
  - Front-loaded volume = information event, lagging volume = quiet day
  - Score: +1.0 (accelerating volume = conviction) to -1.0 (collapsing volume = no interest)

- [ ] `signals/institutional_flow.py` — Signal 50: Combined institutional positioning
  - Aggregate signal from: institutional_pct + option OI skew + large block volume
  - Score: +1.0 (institutional accumulation) to -1.0 (institutional distribution)

### Phase 6: Combination Engine
- [ ] `signals/combiner.py` — The article's 11-step alpha combination procedure

  **Critical prerequisite: R(i,s) construction**
  The combination engine does NOT operate on raw signal scores. It operates on
  R(i,s) = the return generated by following signal i in period s:

  ```
  R(i,s) = score(i, sym, s) x forward_return(sym, s -> s+h_cat)
  ```

  Where h_cat = FORWARD_RETURN_HORIZON[signal.category]. Each category uses its
  own horizon because data sources resolve at different speeds (price signals
  resolve in ~1hr, fundamental signals over days). R(i,s) is normalized in
  Step 4, so different horizons produce comparable scales.

  R(i,s) is averaged across all symbols in the universe at each period s to
  produce a single time series per signal.

  **Requires:**
  - `signal_scores` table with historical scores per symbol per timestamp
  - Forward return computation: join signal timestamps to price data h bars later
  - `signal_returns` table to store R(i,s) series

  **The 11 steps (article-exact):**

  Given N signals, M historical periods, return series R(i,s):

  - [ ] Step 1: Collect R(i,s) for each signal i across M periods
        R(i,s) = avg_across_symbols[ score(i, sym, s) x (price(sym, s+h) - price(sym, s)) / price(sym, s) ]

  - [ ] Step 2: Serially demean — remove each signal's average drift
        X(i,s) = R(i,s) - mean(R(i,:))

  - [ ] Step 3: Calculate sample variance
        sigma(i)^2 = (1/M) x sum_s X(i,s)^2

  - [ ] Step 4: Normalize to common scale
        Y(i,s) = X(i,s) / sigma(i)
        After this, all signals are comparable regardless of original magnitude

  - [ ] Step 5: Drop the most recent observation from Y
        Retain only periods 1..M-1. This prevents lookahead bias — the weight
        calculation must not see data from the period you're about to trade

  - [ ] Step 6: Cross-sectionally demean at each time period
        Lambda(i,s) = Y(i,s) - (1/N) x sum_j Y(j,s)
        At each point in time, subtract the average performance across all
        signals. Removes any market-wide effect driving all signals together.
        What remains is each signal's idiosyncratic contribution.

  - [ ] Step 7: Drop last period from Lambda
        Retain M-2 periods. Final data hygiene step eliminating any residual
        lookahead information in the cross-sectional demeaned series.

  - [ ] Step 8: Forward expected return for each signal
        E(i) = (1/d) x sum R(i,s) over most recent d periods
        E_normalized(i) = E(i) / sigma(i)
        d = SIGNAL_WEIGHT_LOOKBACK_DAYS. This is the forward-looking estimate
        of each signal's expected contribution, normalized to common units.

  - [ ] Step 9: Regress out shared variance
        Run OLS regression: E_normalized ~ Lambda (no intercept, unit weights)
        The dependent variable is the N-length vector E_normalized(i).
        The predictors are the Lambda(i,s) series.
        Residuals epsilon(i) = component of each signal's expected return that is
        genuinely independent — not explained by patterns shared across signals.
        This is the critical step: you're not asking which signal has the highest
        expected return, but which contributes the most independent information.
        Implementation: numpy.linalg.lstsq with rcond=None for numerical stability.
        Fallback: if matrix is singular or underdetermined (N > M-2), use
        pseudoinverse. If still unstable, fall back to equal weights 1/N.

  - [ ] Step 10: Calculate portfolio weight for each signal
        w(i) = eta x epsilon(i) / sigma(i)
        Weight is proportional to independent edge and inversely proportional
        to noise. High independent edge + low variance = high weight.
        Low independent edge + high variance = low weight.

  - [ ] Step 11: Normalize the weight vector
        Set eta so that sum|w(i)| = 1. Full allocation, no unintended leverage.

  **After the 11 steps:**

  - [ ] Compute N_eff from correlation matrix of R(i,s):
        Build NxN correlation matrix C
        Eigenvalues lambda = numpy.linalg.eigvalsh(C)
        N_eff = (sum lambda)^2 / sum(lambda^2)
        This tells you how many genuinely independent signals you have.

  - [ ] N_eff monitoring and circuit breakers:
        - Log N_eff each round to `signal_weights` table
        - WARNING if N_eff < 15 (signals more correlated than expected)
        - ALERT if N_eff < 10 (near current slot-system levels, investigate)
        - Track N_eff trend: sustained decline = regime shift collapsing independence
        - If N_eff < 8 for 3 consecutive rounds: fall back to equal weights
          and flag for manual review (the combination engine has lost its
          structural advantage over the old system)
        - Cross-asset correlation from environment.py `compute_environment()`
          already available: use as early warning (high cross_asset_correlation
          predicts low N_eff next round)

  - [ ] Compute composite_score per symbol:
        composite(sym) = sum_i w(i) x score(i, sym, now)
        composite is clamped to [-1.0, +1.0]

  - [ ] Persist weights to signal_weights table
  - [ ] Persist composite scores to composite_scores table
  - [ ] When insufficient data (< MIN_SHARED_PERIODS_FOR_COMBINATION periods):
        Use equal weights w(i) = 1/N for all signals

- [ ] `signals/scorer.py` — Tiered scoring orchestrator

  **Three-tier architecture: wide cheap scan → narrow expensive scan → trade recs.**
  The bottleneck is option chain fetching (per-symbol, 3 concurrent max, ~2s/call).
  40 of 50 signals need NO option chain data. By scanning 150 symbols with cheap
  signals first and only fetching option chains for the top 20, we get 6x the
  universe coverage at ~1.5x the API cost.

  **Tier 1 — Wide Scan (TIER1_UNIVERSE_SIZE = 150 symbols):**
  - Fetch bulk quotes + bulk candles (2 API calls total, any symbol count)
  - Fetch news (150 calls, but 120s TTL = ~50% cached)
  - Fetch fundamentals via yfinance (no MDA credits, 300s TTL)
  - Score 40 signals: Price (13) + Fundamental (12) + Macro (10) + Microstructure (5)
  - Compute partial composite per symbol using equal or available weights
  - Elapsed: ~3s (all bulk/cached data, pure computation)
  - Credits: ~80

  **Tier 2 — Deep Scan (DEEP_SCAN_TOP_N = 20 symbols):**
  - Take top 20 symbols by |partial_composite| from Tier 1
  - Fetch option chains (20 calls, cached mode, 3 concurrent) + IV rank (20 calls)
  - Score remaining 10 Volatility signals (14-23) for these 20 symbols only
  - Recompute full 50-signal composite with all weights
  - Elapsed: ~14s (option chain bottleneck: 20 ÷ 3 × ~2s)
  - Credits: ~40

  **Tier 3 — Trade Recommendations (TRADE_REC_TOP_N = 8 symbols):**
  - Take top 8 symbols by |full_composite| from Tier 2
  - Run template selector: match composite + regime to template boundaries
  - Write template_recommendations to DB
  - Elapsed: <1s (pure computation)
  - Credits: 0

  **Total per round: ~120 credits, ~20s elapsed, 150-symbol coverage.**

  **Signal refresh tiers (matching data_provider TTLs):**
  ```
  EVERY ROUND:             Price (13), Microstructure (5)   → candle/quote TTL 15-30s
  EVERY ROUND (Tier 2):    Volatility (10)                  → option_chain TTL 30s
  EVERY 5 MIN (if stale):  Fundamental (12)                 → yfinance TTL 300-600s
  EVERY ROUND:             Macro/Sentiment (10)             → environment + calendar + news
  ```
  Fundamental signals are NOT re-scored if yfinance cache is fresh (300-600s TTL).
  This avoids redundant yfinance calls. The scorer checks cache age before calling
  each signal category.

  **Option chain cost management:**
  - Use CACHED mode for option chains (1 credit per call, not per symbol)
  - Apply server-side filters: dte_range, strike_limit, min_open_interest
  - Rate-limit to 3 concurrent option chain calls (existing _OPTIONS_MAX_CONCURRENCY)
  - Only Tier 2 symbols (top 20) incur option chain cost — not all 150
  - Total options credits per round: 20 (chains) + 20 (IV rank) = 40 credits
  - Budget check: compare accumulating credits against MAX_CREDITS_PER_ROUND

  - Write all signal_scores (Tier 1 + Tier 2) to DB
  - Compute forward returns from prior-period scores (join to price data)
  - Write R(i,s) to signal_returns table
  - Call combiner to produce weights and composite scores
  - Triggered: once per research round

### Phase 7: Execution Template Selector
The template selector replaces the 12 strategy slots. Unlike the old fixed decision matrix,
decision boundaries are LEARNED from historical performance and evolve CONTINUOUSLY via
walk-forward simulation as a third `asyncio.gather()` task alongside the research scorer
and trading agent. Cooldown: 30 min during market hours, 5 min outside market hours.

- [ ] `signals/templates.py` — Template definitions + selection logic

  **10 Templates (covering all order types from old slot mandates):**

  | Template | Order Type | Use Case |
  |----------|-----------|----------|
  | `stock_market` | market/adaptive | High conviction, immediate entry, low spread |
  | `stock_bracket` | bracket (entry + OCA stop/target) | High conviction, structured risk |
  | `stock_limit` | limit/midprice | Moderate conviction, patient entry |
  | `stock_vwap` | vwap | Large size, minimize impact |
  | `stock_trailing` | trailing_stop_exit | Trend following, let winners run |
  | `vertical_spread` | vertical_spread | Directional, high IV, defined risk |
  | `calendar_diagonal` | calendar_spread/diagonal_spread | Time decay + directional lean |
  | `premium_iron_condor` | iron_condor | Neutral-to-low conviction, high IV |
  | `premium_butterfly` | butterfly/iron_butterfly | Pinning thesis, high IV |
  | `straddle_strangle` | straddle/strangle | Vol expansion thesis, pre-event |

  **Decision boundary parameters (stored in template_boundaries table):**
  Each template has a set of boundary conditions that determine when it's selected:
  ```
  composite_min, composite_max   — conviction range
  iv_rank_min, iv_rank_max       — volatility regime
  atr_pct_min, atr_pct_max       — price volatility
  vol_regime                      — environment match
  breadth_regime                  — market context
  ```
  These START at reasonable defaults and are EVOLVED based on performance data.

  **Selection logic:**
  For each (symbol, composite_score), find all templates whose boundary conditions match.
  If multiple match, pick the one with the highest historical win_rate in the current
  regime (from template_performance table). If no history, use the template with the
  tightest boundary match. If no template matches, output no_trade.

  **Trade dict output (same format as old signals):**
  ```python
  {
      "symbol": str,
      "direction": "long" | "short",
      "order_type": str,
      "entry_price": float,
      "target_price": float,    # calculated from ATR
      "stop_price": float,      # calculated from ATR
      "max_hold_bars": int,
      "setup_type": str,        # template name
      "legs_json": dict | None, # for options templates
      "composite_score": float, # for transparency
      "template_track_record": dict,  # win%, trades, avg_return
  }
  ```

- [ ] `signals/template_evolution.py` — Template boundary evolution loop

  **Runs CONTINUOUSLY as third `asyncio.gather()` task in `__main__.py`:**
  ```python
  # __main__.py — updated wiring:
  asyncio.gather(run_agent(), run_research(), run_template_evolution())
  ```
  - During market hours: evolves on LIVE data (latest composite scores + real
    trade outcomes), captures regime shifts as they happen, 30 min cooldown
    between rounds to avoid thrashing boundaries mid-session.
  - Outside market hours: evolves faster (5 min cooldown) on accumulated data,
    deeper exploration (more mutations per round, wider boundary search).
  - Each round is pure computation (simulator/promoter), no LLM, no API calls.
  - Template boundaries write to DB; scorer reads them next round automatically.

  **Reuses existing infrastructure — NO new simulation code needed:**
  - `research/simulator.py` — `simulate()` with walk-forward, slippage presets,
    `compute_expectancy()` for metrics, `compute_sample_confidence()` for CI
  - `research/promoter.py` — `OptionsPromoter.reprice_signals()` for options
    template validation (reprices legs against historical chains, coverage gate 80%)
  - `research/replay.py` — `ReplayHarness.run()` for deterministic validation
    (already classifies safe vs disabled tools)

  1. Load historical composite_scores + price outcomes from DB
  2. For each template + boundary configuration:
     - Simulate: apply template to historical composites, compute P&L from price data
     - Walk-forward: 70% train / 30% OOS test on chronological split
     - Metrics: win_rate, avg_return, profit_factor, max_drawdown, Sharpe
  3. Generate boundary mutations:
     - Shift composite thresholds +/-0.05
     - Shift IV rank boundaries +/-5
     - Narrow/widen ATR ranges
     - Toggle regime restrictions
  4. Evaluate mutations on OOS data
  5. Keep-gate: mutation must beat current boundaries on OOS metrics
  6. Persist winning boundaries to template_boundaries table
  7. Update template_performance with latest OOS stats

  **Promotion gating (same philosophy as old promoter/replay):**
  - Fast eval: simulate on training data (quick reject bad mutations)
  - OOS validation: test on withheld 30% (confirms not overfit)
  - For options templates: reprice legs against historical option chains (reuse promoter.py)

  **Feedback loop:**
  - Every real trade records: template, regime, composite at entry, P&L, execution gap
  - Execution gaps feed back into next evolution round
  - Templates that consistently underperform in a regime have their boundary conditions
    tightened (higher composite threshold required) or are excluded from that regime

- [ ] Template performance tracking:
  - DB table with per-template, per-regime, per-composite-bucket stats
  - Updated on each trade close
  - Surfaced in agent briefing
  - Used by selection logic to rank competing templates

### Phase 8: Wire Into Agent Briefing

The agent receives a layered briefing. The composite score is the WHAT (direction + conviction).
The template recommendation is the HOW (trade structure). The agent's job is the WHEN and
WHETHER (execution judgment: timing, capacity, spread quality, fill management).

The agent CANNOT override direction (the composite already resolved that from 50 independent signals).
The agent CAN reject a recommendation (account constraints, earnings imminent, bad spread, no capacity).
The agent CANNOT substitute a different template unless template was "no_trade" and agent has
information the signal layer doesn't (e.g., breaking news not yet in news_sentiment signal).

- [ ] `signals/briefing.py` — Build composite briefing for agent

  **Layer 1: Signal Quality**
  ```
  SIGNAL QUALITY:
    N_eff: 34.2 of 50 signals (independence ratio: 68%)
    Weight leaders: iv_rank (0.08), beta_adj_momentum (0.06), insider (0.05), ...
    Redundant pairs: momentum<->breakout (r=0.74), momentum<->vol_weighted_mom (r=0.71)
    Signal health: 48/50 active, 2 stale (institutional: 12h, insider: 8h)
  ```

  **Layer 2: Composite Scores (top movers)**
  ```
  TOP COMPOSITE SCORES:
    NVDA +0.72 [price:+0.6 vol:-0.1 fund:+0.8 macro:+0.3 micro:+0.4]
    TSLA -0.58 [price:-0.5 vol:+0.2 fund:-0.6 macro:-0.2 micro:-0.3]
    PLTR +0.45 [price:+0.3 vol:+0.1 fund:+0.5 macro:+0.2 micro:+0.1]
    ...
  ```
  Category sub-scores are the weighted sum of signals within each category,
  giving the agent transparency into WHY the composite is what it is.

  **Layer 3: Recommended Trades (template-selected)**
  ```
  RECOMMENDED TRADES:
    NVDA long vertical_spread @ 890 tgt=905 stp=880 R:R=1.5
      composite=+0.72, template: vertical_spread (68% win, 22 trades, avg +0.6%)
      iv_rank=62, vol_regime=normal
    TSLA short stock_bracket @ 245 tgt=238 stp=250 R:R=1.4
      composite=-0.58, template: stock_bracket (64% win, 18 trades, avg +0.7%)
      iv_rank=35, vol_regime=normal
  ```

  **Layer 4: Template Performance (30-day rolling)**
  ```
  TEMPLATE PERFORMANCE (30d):
    stock_bracket:    64% win, avg +0.7%, 23 trades, Sharpe 1.2
    vertical_spread:  68% win, avg +0.6%, 22 trades, Sharpe 1.1
    premium_condor:   72% win, avg +0.3%, 15 trades, Sharpe 0.9
    stock_limit:      58% win, avg +0.9%, 12 trades, Sharpe 1.0
    straddle:         55% win, avg +1.1%, 8 trades, Sharpe 0.8
  ```

- [ ] Replace ALL 5 briefing sub-functions in `tools/tools_research.py`:
  The agent calls `handle_briefing(executor, params)` with a `detail` param that
  dispatches to 5 sub-functions. ALL must be replaced:
  - `_briefing_summary(data)` → Layer 1-3 of new briefing
  - `_briefing_signals(data)` → Layer 3 (recommended trades)
  - `_briefing_strategies(data)` → Layer 4 (template performance)
  - `_briefing_feedback(data)` → KEEP (uses compute_sample_confidence +
    get_execution_cost — still useful for execution quality tracking)
  - `_briefing_environment(data)` → Update to include N_eff trend + signal health
  The `detail` parameter dispatch in `handle_briefing()` stays the same — the agent
  drills down via: `summary|signals|strategies|feedback|environment`.

- [ ] Replace `_query_briefing_data()`:
  - Query `composite_scores` instead of `live_signals`
  - Query `signal_weights` for N_eff and weight breakdown
  - Query `template_performance` for track record
  - Query `template_recommendations` for current trade recs
  - Keep `trade_feedback` query (unchanged)

  NOTE: `_get_research_briefing()` in `core/agent.py` is dead code — never called.
  The agent uses the tool-based `handle_briefing()` exclusively. No changes needed
  in core/agent.py for briefing.

### Phase 9: Kelly Sizing with CV Adjustment
- [ ] Add `_estimate_cv_edge()` in `tools/tools_sizing.py`
  - Bootstrap-resample matched trade returns 1000x
  - Compute edge (avg return) each iteration
  - Return coefficient of variation (CV = sigma_edge / mu_edge)
  - Requires >= 20 matched fills to activate; otherwise CV = 0

- [ ] Modify `handle_calculate_size()`:
  - `risk_per_trade_pct` is a per-call tool parameter with hardcoded default 1.5%
    (NOT read from core.config — see C10). Inject CV adjustment before params.get():
    ```python
    base_risk = 1.5
    cv_edge = _estimate_cv_edge(...)  # 0.0 if < 20 fills
    adjusted_risk = base_risk * (1 - cv_edge)
    risk_per_trade_pct = float(params.get("risk_per_trade_pct", adjusted_risk))
    ```
  - Higher CV (less certain about edge) -> smaller positions
  - Lower CV (consistent edge) -> full-sized positions
  - Binding constraint logic (min of risk/concentration/cash) unchanged

### Phase 10: Remove Old Slot System
- [ ] Delete `research/slots/` directory (all 12 strategy files + README + __init__)
  Safe: zero imports from slots/ exist anywhere in the codebase. Strategy files are
  loaded via raw `_strategy_path().read_text()` which is itself a deletion target.
- [ ] Near-complete replacement of `research/agent.py`:
  Deleting the 7 targeted functions orphans ~30 of 42 total functions (nearly everything
  is only called from the deletion targets). Keep only:
  - `_fetch_universe_candles()` — useful for scorer data fetching
  - `_store_environment_snapshot()` — reuse for new scorer
  - `_get_env_slot_history()` → rename to `_get_env_history()`, called from briefing
  - `_get_trade_feedback()` / `_match_trades_to_signals()` — called from tools
  Delete all of:
  - `_update_darwinian_weights()` and all Darwinian weight logic
  - `_compute_slot_correlations()` and `_pearson_r()` helper
  - `_run_slot()` loop and slot-based evaluation
  - `_run_live_scan()` per-slot live signal generation
  - `_execute_strategy()` and `_load_strategy_module()`
  - `_run_selector()` (slot-based selector agent) and `_extract_strategy_type()`
  - `_auto_tune_config()`
  - `_call_llm()`, `_llm_analyze()`, `_llm_propose_strategy()` (all LLM calls)
  - `_LLM_SEMAPHORE`, `FEATURE_FLAGS`, `_TUNABLE_DEFAULTS` module-level state
  - All SLOT_MANDATES references, `_format_mandate()`, `_check_mandate_compliance()`
  - `_validate_strategy_source()`, `_evaluate_strategy()`, `_deduplicate_signals()`
  - `_compute_fitness()`, `_execution_gap_penalty()`, `_format_fitness_metrics()`
  - `_store_strategy()`, `_store_promotion_run()`, `_get_best_strategy()`
  - `_get_history_summary()`, `_max_condition_trades()`
- [ ] Delete from `research/config.py`:
  - `SLOT_MANDATES`, `REGIME_TO_SLOTS`, `NUM_SLOTS`, `BATCH_SIZE`
  - `DARWINIAN_WEIGHT_CEILING/FLOOR/UP/DOWN`, `SLOT_CORRELATION_THRESHOLD`
  - `SELECTOR_EVERY_N_ROUNDS`, `MAX_SELECTOR_REPLACEMENTS`
  - `SANDBOX_ALLOWED_IMPORTS`, `SANDBOX_BLOCKED_CALLS`, `STRATEGY_EXEC_TIMEOUT` (no sandbox)
  - `REGIME_GATE_ENABLED`, `EXTREME_VOL_FITNESS_BOOST` (templates handle this)
  - `RESEARCH_DAILY_LLM_BUDGET` (no LLM in research)
  - `SIGNAL_SCHEMA` (replaced by Signal base class)
  - `RESEARCH_SYSTEM_PROMPT`, `ANALYSIS_PROMPT_TEMPLATE`, `SELECTOR_PROMPT` (no LLM)
  Keep:
  - `RESEARCH_UNIVERSE` (expanded to 150 symbols), `CANDLE_RESOLUTION`, `EVAL_DAYS_BACK`
  - `FORCE_EXIT_MINUTE = 955`, `SLIPPAGE_BPS = 2` (still used by simulator)
  - `MIN_KEEP_TEST_SIGNALS`, `MIN_KEEP_CONFIDENCE_SCORE`, `MIN_KEEP_CONDITION_TRADES`
    (may need equivalents for signal quality gating)
  - Circuit breaker constants (need equivalents for signal scoring)
  - `ROUND_DELAY_SECS = 30` (pacing between research rounds)
  - `STOCK_ORDER_TYPES`, `OPTIONS_STRATEGIES` (template definitions reference them)
  - `LEGS_JSON_SCHEMAS`, `validate_legs_json()` (promoter + template selector use them)
- [ ] Replace `run_research()` main loop in `research/agent.py` (~line 2500):
  **Current loop:** fetch candles → compute_environment() → walk-forward split →
  batch slots (3/batch, ~16min) → darwinian weights → correlations → selector
  (every 3 rounds) → auto-tune → circuit breaker. ~20min per round.

  **New loop (tiered scoring):**
  - Fetch bulk quotes + bulk candles for full universe (2 calls, any symbol count)
  - Compute environment snapshot (reuse `compute_environment()` from environment.py)
  - **Tier 1 — Wide scan (150 symbols):**
    * Score 40 cheap signals: Price (13) + Fundamental (12) + Macro (10) + Micro (5)
    * All data from bulk endpoints, yfinance, or environment (no option chains)
    * Compute partial composite per symbol
    * Estimated: ~3s compute + ~5s environment = ~8s
  - **Tier 2 — Deep scan (top 20 by |partial_composite|):**
    * Fetch option chains (20 calls, cached mode, 3 concurrent) + IV rank (20 calls)
    * Score 10 Volatility signals for these 20 symbols
    * Recompute full 50-signal composite
    * Estimated: ~14s (option chain bottleneck)
  - **Tier 3 — Trade recs (top 8 by |full_composite|):**
    * Run template selector: match composite + regime to boundaries
    * Write template_recommendations to DB
    * Estimated: <1s
  - Compute forward returns for prior-round scores (join to candle data)
  - Run combiner: produce weights + composite scores + N_eff
  - Write all results to DB
  - Log round summary with N_eff, top composites, credit usage, tier breakdown

  **Expected round time: ~20s** (vs ~20min old system). The bottleneck was LLM
  calls in _run_slot() and _execute_strategy(). The new system is all computation
  — no LLM calls in the signal/combine/template pipeline. The agent LLM is only
  invoked at the briefing/execution stage.

  **Template evolution runs separately** as third `asyncio.gather()` task —
  NOT inside the scoring loop. This decouples scoring cadence (~20s) from
  evolution cadence (30min market hours / 5min overnight).

  - [ ] Environment snapshot integration:
    - For live scoring, use `compute_session_environment(session_universe, session_date)`
      which takes `{symbol: single_df}` — NOT `compute_environment()` which needs
      nested `{symbol: {date_str: df}}` (see C5/C6).
    - For template evolution walk-forward, use `compute_environment_by_date(universe)`
      which returns `{date_str: session_snapshot}` — gives per-date regime context.
    - Both return: volatility_regime, trend_regime, breadth_regime, momentum_regime,
      volume_regime, strategy_fit, avg_atr_pct, dispersion, advance_decline_ratio
    - `compute_environment()` (multi-day) additionally returns: cross_asset_correlation,
      per_symbol details, trend_confidence, momentum/reversion/options_candidates
    - Feed to: macro signals (36-45), template selector (regime matching),
      N_eff early warning (cross_asset_correlation — from multi-day only)
    - Per-symbol details (atr_pct, trend_strength, momentum_shift) feed
      to price signal enrichment

  - [ ] __main__.py wiring — crash isolation for third task:
    Current `asyncio.gather()` has no `return_exceptions` — a crash in any task kills
    all tasks. Must wrap each task or use `return_exceptions=True`:
    ```python
    async def _run_all():
        results = await asyncio.gather(
            run_agent(),
            run_research(verbose=args.verbose),
            run_template_evolution(),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Task crashed: {r}", exc_info=r)
    ```
    Also consider: `--no-evolution` flag alongside existing `--no-research`.
- [ ] Drop old DB tables (after validation period confirms new system works):
  - `live_signals` → replaced by `composite_scores` + `template_recommendations`
  - `strategies` → no more evolving strategy code
  - `slot_environment_scores` → replaced by `signal_weights`
  - Keep `research_config` (reused for per-category horizon config)
  - Keep `trade_log`, `matched_executions`, etc. (execution layer unchanged)

### Phase 11: Tests
- [ ] Unit tests for each signal: known input data -> expected score range and sign
- [ ] Unit test combiner:
  - R(i,s) construction from scores + forward returns
  - Steps 1-11 with synthetic data -> correct weights
  - Identity correlation matrix -> N_eff = N
  - Perfect correlation -> N_eff = 1
  - Known synthetic returns -> predictable weight ordering
  - Step 5/7 lookahead prevention: verify changing last period doesn't change weights
  - Singular matrix fallback -> equal weights without crash
- [ ] Unit test template selector:
  - Composite + regime -> correct template choice
  - Multiple matching templates -> highest track record wins
  - No matching template -> no_trade output
- [ ] Unit test template evolution:
  - Walk-forward split correctness
  - Boundary mutation within valid ranges
  - Keep-gate: inferior mutation rejected
- [ ] Unit test CV estimation: known distribution -> expected CV
- [ ] Integration test: full pipeline (signals -> R(i,s) -> combiner -> templates -> briefing)
- [ ] Regression test: sizing unchanged when < 20 fills (CV inactive)

---

## Signal Taxonomy (50 Signals x 5 Categories)

### Category Independence Matrix

| Category | Data Source | Count | Updates | Within-Category Correlation |
|----------|-----------|-------|---------|---------------------------|
| Price (13) | MDA candles OHLCV | 13 | Every round (Tier 1) | Moderate-high (all from same OHLCV) |
| Volatility (10) | MDA option chains + IV | 10 | Every round (Tier 2 only) | Moderate (IV-related metrics correlate) |
| Fundamental (12) | yfinance | 12 | Every 5-10 min | Low-moderate (valuation vs quality vs insider are distinct) |
| Macro/Sentiment (10) | environment + calendar + news + SPY | 10 | Every round | Mixed (breadth and market momentum correlate) |
| Microstructure (5) | MDA quotes + option OI | 5 | Every round | Low (spread, flow, stability measure different things) |

**Cross-category correlation: structurally near zero.** IV rank does not derive from candle OHLCV. Insider buying does not derive from IV. News sentiment does not derive from fundamentals. The combination engine's regression (Step 9) will quantify and exploit this independence.

**Expected N_eff: 20-30 of 50 (steady state), possibly lower during stress.**
Within-category correlation will reduce the effective count within price (13 → ~4-5
effective) and volatility (10 → ~4-5 effective). Cross-category signals should maintain
near-full independence in normal regimes but will compress during macro stress events
(when correlations spike). With IC of 0.05 per effective signal:

```
N_eff = 30: IR = 0.05 x sqrt(30) = 0.274  (optimistic, calm market)
N_eff = 20: IR = 0.05 x sqrt(20) = 0.224  (realistic average)
N_eff = 15: IR = 0.05 x sqrt(15) = 0.194  (stress/correlated regime)
N_eff = 10: IR = 0.05 x sqrt(10) = 0.158  (severe stress, still 1.8x improvement)

vs current system:
N_eff ~= 3:  IR = 0.05 x sqrt(3)  = 0.087
```

Even the stress-case estimate is a ~2x improvement in information ratio.

---

## Execution Order and Dependencies

```
Phase 0 -----> Phase 1 --\
Phase 0 -----> Phase 2 ---|
Phase 0 -----> Phase 3 ---|---> Phase 6 (combiner needs >= 10 periods of all signals)
Phase 0 -----> Phase 4 ---|
Phase 0 -----> Phase 5 --/
                          |
Phase 6 --> Phase 7 --> Phase 8 --> Phase 10 --> Phase 11
                          |
Phase 9 is independent (sizing changes only, can be done anytime)

Template evolution runs as independent async task after Phase 7.
It does NOT block the scoring loop — they share DB only.
```

**Coexistence with old system:** Phases 1-6 write to their own DB tables and can run
alongside the existing slot system. The signal layer produces composite scores; the old
system produces slot signals. Both can coexist until Phase 8 (briefing rewire) and Phase 10
(slot removal). This allows validating composite scores against slot signals before the cutover.

**Incremental testing:** After Phase 1 + Phase 6, you can test the combiner with just 13
price signals. Each subsequent phase adds a category and you can measure N_eff climbing
as genuinely independent signals join.

---

## What Gets Deleted

| Component | Replaced By |
|-----------|-------------|
| 12 strategy slot files (`research/slots/`) | 50 signal generators (`signals/`) |
| `_update_darwinian_weights()` | Combination engine weights (11-step procedure) |
| `_compute_slot_correlations()` | Correlation matrix + N_eff from eigenvalues |
| `_run_slot()` evolution loop | Signal scorer + template evolution loop |
| `_run_live_scan()` | Template selector on composite scores |
| `_execute_strategy()` sandbox | Direct `signal.score()` calls (no sandbox needed) |
| Selector agent (LLM-based) | Mathematical weight optimization (no LLM needed) |
| SLOT_MANDATES, REGIME_TO_SLOTS | Template decision boundaries (learned, not hardcoded) |
| `live_signals` table | `composite_scores` + `template_recommendations` |
| `strategies` table | `signal_scores` + `signal_weights` + `signal_returns` |
| `slot_environment_scores` | `signal_weights` + `template_performance` |

## What Stays Unchanged

| Component | Why |
|-----------|-----|
| Simulator (`research/simulator.py`) | Reused by template evolution for walk-forward backtesting |
| Promoter (`research/promoter.py`) | Reused by template evolution for options repricing |
| Replay (`research/replay.py`) | Reused by template evolution for deterministic validation |
| Execution layer (all `ibkr_*`) | No interface changes — same order types, same fills |
| Environment (`research/environment.py`) | Feeds macro signals + template regime context |
| Data provider (`data/data_provider.py`) | All data access unchanged — signals call same methods |
| Core agent loop (`core/agent.py`) | Consumes briefing text, format changes are transparent |
| Cost tracker | Unchanged |
| Market hours | Unchanged |

---

## Implementation Notes — Investigation Findings

These notes document concrete constraints discovered by investigating the actual codebase,
MDA API documentation, and existing infrastructure. They address the 6 remaining issues
identified in the plan review.

### 1. API Budget & Data Dependencies (Issue #1)

**MDA Option Chain Pricing (from official docs):**
| Mode | Cost | Use Case |
|------|------|----------|
| Real-time | 1 credit per option symbol in response | NOT viable for scoring (AAPL ≈ 200+ contracts) |
| Cached | 1 credit per API call (any # of symbols) | Primary mode for signal scoring |
| Historical | 1 credit per 1000 option symbols | Template evolution backtesting |

**Cached mode is the critical enabler.** Without it, scoring 25 symbols' option chains
would cost 5,000+ credits per round. With cached mode: 25 credits.

**Per-Round API Budget Estimate (tiered: 150 → 20 → 8 symbols):**
```
Tier 1 — Wide Scan (150 symbols, 40 signals, NO option chains):
  Candles bulk:    2 calls    →    2 credits
  Quotes bulk:     2 calls    →    2 credits
  News:          150 calls    →  ~75 credits (120s TTL, ~50% cached)
  Fundamentals:  yfinance     →    0 credits (no MDA)
                               ─────────
  Tier 1 subtotal:              ~80 credits

Tier 2 — Deep Scan (top 20 symbols, add 10 volatility signals):
  Option chains:  20 calls    →   20 credits (cached mode, 3 concurrent)
  IV rank:        20 calls    →   20 credits
                               ─────────
  Tier 2 subtotal:              ~40 credits

Tier 3 — Trade Recs (top 8 symbols):
  Template selection: computation → 0 credits
                               ─────────
  ROUND TOTAL:                  ~120 credits, ~20s elapsed
  At ~3 rounds/min:             ~360 credits/hour
  At 6.5 market hours:          ~2,340 credits/day
```
This is well within typical MDA plan limits (250,000+ credits/day on Trader plan).
**6x symbol coverage vs old flat 25-symbol scan at ~1.5x the cost.**

**Data that CANNOT be fetched in bulk (Tier 2 only):**
- Option chains: per-symbol only (no bulk endpoint). Rate-limited to 3 concurrent.
- IV rank: per-symbol only.
- Fundamentals/insider/institutional: yfinance (no MDA credits, but slow at 300-600s TTL).

**Data that CAN be fetched in bulk (Tier 1, scales to any universe size):**
- Quotes: `get_quotes_bulk()` — single call for all symbols (DataProvider method)
- Daily candles: needs new `get_candles_bulk()` wrapper on DataProvider
  (MarketDataClient has `get_bulk_daily_candles()` but DataProvider does not expose it — see C2)

**Server-side chain filters to reduce response size and improve speed:**
```python
get_option_chain(symbol,
    dte_range=(7, 60),       # Skip weeklies and distant months
    strike_limit=20,          # Max 20 strikes per expiration
    min_open_interest=50,     # Skip illiquid strikes
    min_bid=0.05,             # Skip penny options
)
```

### 2. Forward-Return Horizon Tuning (Issue #2)

**Problem:** A single `FORWARD_RETURN_HORIZON_BARS = 60` conflates fast-resolving price
signals with slow-resolving fundamental signals. A momentum crossover resolves in hours;
an insider-buying signal may take days to play out.

**Solution:** Per-category horizons (added to Phase 0 config above). The combiner's
Step 4 normalization ensures R(i,s) values are comparable across categories despite
different horizon lengths.

**Tuning mechanism:** The existing `research_config` table (with `get_research_config()`
and `set_research_config()` helpers in memory/__init__.py) can store per-category horizons.
The auto-tune loop can adjust horizons based on signal-return autocorrelation:
- If R(i,s) is positively autocorrelated at lag 1, the horizon is too short (signal
  still resolving when measured).
- If R(i,s) is negatively autocorrelated, the horizon is too long (signal has reversed
  by evaluation time).
- Optimal horizon: near-zero autocorrelation at lag 1.

This is a Phase 6+ optimization — start with the hardcoded defaults, tune after
collecting 50+ rounds of data.

### 3. N_eff Realism (Issue #3)

**Projection: 25-35 of 50.** This is reasonable given structural independence across
categories, but should be treated as an upper bound until measured empirically.

**What could reduce N_eff below projection:**
- Macro stress events (correlations spike to 1.0 across all assets → all signals
  converge → N_eff crashes). The environment.py `cross_asset_correlation` metric
  provides early warning for this.
- Regime-dependent correlation: momentum and breakout signals correlate ~0.7 in
  trending markets but ~0.2 in choppy markets. N_eff will fluctuate with regime.
- Data staleness: if yfinance endpoints go down, all 12 fundamental signals
  simultaneously produce stale scores → correlation spikes within category.

**Mitigation (added to Phase 6 above):**
- N_eff circuit breaker: if N_eff < 8 for 3 consecutive rounds → equal weights
- N_eff dashboard: log per-round value, trend, and category breakdown
- Category-level N_eff: compute within-category N_eff to detect which category
  is driving correlation spikes

**Honest assessment:** Even N_eff = 15 (pessimistic during stress) gives
IR = 0.05 × √15 = 0.194, still 2.2× better than the current system (N_eff ≈ 3,
IR = 0.087). The structural improvement holds across a wide range of N_eff outcomes.

### 4. Template Evolution Timing & Compute (Issue #4)

**No new simulation infrastructure needed.** The existing components handle everything:

| Component | Reuse For | Location |
|-----------|-----------|----------|
| `simulator.simulate()` | Walk-forward P&L on historical composites | research/simulator.py |
| `simulator.compute_expectancy()` | Template fitness metrics (hit_rate, Sharpe, etc.) | research/simulator.py |
| `simulator.compute_sample_confidence()` | CI95, t_stat, luck_pressure per template | research/simulator.py |
| `promoter.reprice_signals()` | Validate options templates against historical chains | research/promoter.py |
| `replay.run()` | Deterministic validation of trade execution | research/replay.py |

**Runs continuously as third `asyncio.gather()` task** — NOT inside the scoring loop.
This decouples scoring cadence (~20s) from evolution cadence:
- **Market hours:** 30 min cooldown between evolution rounds. This prevents
  thrashing template boundaries while the agent is actively trading, but still
  captures intraday regime shifts and live trade outcomes.
- **Outside market hours:** 5 min cooldown. Deeper exploration with more mutations
  per round (wider boundary search, longer walk-forward windows).

**Compute budget:** With 10 templates × ~5 mutations each × walk-forward simulation
= ~50 simulations per round. Each simulation is pure computation on historical data
(no API calls, no LLM calls). Estimated: 2-5 minutes per evolution round.

**During market hours:** Evolution sees real trade outcomes in near real-time.
A template that starts underperforming gets its boundaries tightened within 30 min,
not overnight. The keep-gate prevents regressions — mutations must beat current
boundaries on OOS data before replacing them.

**Options template evolution** uses historical option chains (1 credit per 1000 symbols
from MDA). For 20 deep-scan symbols × 12 months of history ≈ negligible credit cost.

### 5. DB Migration Strategy (Issue #5)

**Current approach:** `memory/__init__.py` → `init_db()` uses `CREATE TABLE IF NOT EXISTS`
for all 17 tables. No versioned migration system. No ALTER TABLE.

**Strategy for 7 new tables:**
Add new CREATE TABLE IF NOT EXISTS statements to `init_db()`. This is safe because:
- Existing tables are not modified (no ALTER TABLE needed)
- New tables are purely additive
- `init_db()` runs at startup (already called before any DB access)
- Old tables remain until Phase 10 cutover (coexistence period)

**New tables (exact schema):**
```sql
-- Phase 0: Foundation tables
CREATE TABLE IF NOT EXISTS signal_scores (
    signal_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts REAL NOT NULL,  -- Unix timestamp
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    components_json TEXT,
    PRIMARY KEY (signal_name, symbol, ts)
);

CREATE TABLE IF NOT EXISTS signal_weights (
    signal_name TEXT NOT NULL,
    weight REAL NOT NULL,
    n_eff REAL,
    category TEXT,
    updated_ts REAL NOT NULL,
    PRIMARY KEY (signal_name)
);

CREATE TABLE IF NOT EXISTS signal_returns (
    signal_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts REAL NOT NULL,
    score_at_entry REAL NOT NULL,
    forward_return REAL NOT NULL,
    r_value REAL NOT NULL,  -- score × forward_return
    horizon_bars INTEGER NOT NULL,
    PRIMARY KEY (signal_name, symbol, ts)
);

CREATE TABLE IF NOT EXISTS composite_scores (
    symbol TEXT NOT NULL,
    ts REAL NOT NULL,
    composite_score REAL NOT NULL,
    signal_breakdown_json TEXT,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS template_performance (
    template_name TEXT NOT NULL,
    regime_key TEXT NOT NULL,
    composite_bucket TEXT,  -- e.g., "high_pos", "low_pos", "neutral"
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    avg_return_pct REAL DEFAULT 0,
    avg_gap_pct REAL DEFAULT 0,
    sharpe REAL,
    updated_ts REAL NOT NULL,
    PRIMARY KEY (template_name, regime_key, composite_bucket)
);

CREATE TABLE IF NOT EXISTS template_boundaries (
    template_name TEXT NOT NULL,
    param_name TEXT NOT NULL,
    param_value REAL NOT NULL,
    generation INTEGER DEFAULT 0,
    fitness REAL,
    updated_ts REAL NOT NULL,
    PRIMARY KEY (template_name, param_name)
);

CREATE TABLE IF NOT EXISTS template_recommendations (
    symbol TEXT NOT NULL,
    ts REAL NOT NULL,
    template_name TEXT NOT NULL,
    direction TEXT NOT NULL,
    composite_score REAL NOT NULL,
    order_type TEXT,
    entry_price REAL,
    target_price REAL,
    stop_price REAL,
    legs_json TEXT,
    PRIMARY KEY (symbol, ts)
);
```

**Index strategy:** Add indexes after Phase 6 data accumulates:
```sql
CREATE INDEX IF NOT EXISTS idx_signal_scores_ts ON signal_scores(ts);
CREATE INDEX IF NOT EXISTS idx_signal_returns_signal ON signal_returns(signal_name, ts);
CREATE INDEX IF NOT EXISTS idx_composite_scores_ts ON composite_scores(ts);
```

### 6. Integration Points (Issue #6)

**Detailed wiring map — what calls what:**

```
ENVIRONMENT (unchanged)
  compute_environment(universe) → dict with regimes + per-symbol details
  ├── Feeds macro signals 36-45 directly (breadth, correlation, regime persistence)
  ├── Feeds template selector (vol_regime, breadth_regime for boundary matching)
  └── Feeds N_eff early warning (cross_asset_correlation)

DATA PROVIDER (unchanged)
  get_quote(), get_candles(), get_option_chain(), get_iv_rank(), etc.
  ├── Feeds price signals 1-13 (candles, quotes)
  ├── Feeds volatility signals 14-23 (option chains, IV rank)
  ├── Feeds microstructure signals 46-50 (quotes, option OI)
  └── Feeds fundamental signals 24-35 (via yfinance bridge methods)

SCORER (new — tiered architecture)
  score_all(universe, env_snapshot):
  ├── Tier 1: score_tier1(150 symbols, 40 signals) → partial composites
  ├── Tier 2: score_tier2(top 20, 10 vol signals) → full composites
  ├── Tier 3: select_templates(top 8) → template recommendations
  └── Calls combiner → writes signal_weights, composite_scores

TEMPLATE EVOLUTION (new — third asyncio.gather() task)
  run_template_evolution():
  ├── Runs continuously alongside agent + scorer
  ├── 30 min cooldown during market hours, 5 min outside
  ├── Reads template_performance + composite_scores history
  ├── Mutates + validates boundaries via simulator/promoter
  └── Writes template_boundaries (scorer reads them next round)

TEMPLATE SELECTOR (new)
  select_templates(composite_scores, env_snapshot) → writes template_recommendations
  └── Reads template_boundaries, template_performance

BRIEFING (new, replaces old briefing functions)
  build_briefing() → formatted string for agent
  ├── Reads signal_weights (N_eff, weight leaders)
  ├── Reads composite_scores (top movers with breakdown)
  ├── Reads template_recommendations (trade recs with track record)
  └── Reads template_performance (30-day rolling stats)

AGENT TOOLS (modified)
  tools_research.py:
  ├── _briefing_summary() → calls briefing.build_briefing() instead of SQL queries
  ├── _briefing_signals() → subsumed by Layer 3 of new briefing
  └── _briefing_strategies() → subsumed by Layer 4 of new briefing

  tools_sizing.py:
  └── handle_calculate_size() → adds CV adjustment from matched trade returns
```

**Briefing SQL changes (tools_research.py):**
The current `_query_briefing_data()` queries `live_signals` and `strategies` tables.
Replace with queries against `composite_scores`, `signal_weights`, `template_performance`,
and `template_recommendations`. The function signature stays the same (returns dict),
only the SQL and dict keys change.

---

## Overall Assessment

### What This Plan Gets Right

1. **The structural diagnosis is correct.** The Darwinian weights being decorative (never
   reaching the agent via briefing) is a genuine architectural flaw. The N_eff ≈ 2-3
   analysis is accurate — 12 OHLCV-derived slots are not 12 independent signals.

2. **The two-layer separation (signals + templates) is sound.** Separating "what direction?"
   from "how to structure?" is the right abstraction. The article's combination engine
   provides a mathematically principled approach to the first question.

3. **The existing infrastructure supports this.** The data provider already has every method
   needed (option chains with Greeks, fundamentals, insider data, environment regimes).
   The simulator/promoter/replay stack is directly reusable for template evolution.
   The DB pattern (CREATE IF NOT EXISTS) works for additive tables.

4. **The API budget is viable.** Tiered scoring with cached-mode option chains gives
   150-symbol coverage at ~120 credits/round (~2,340 credits/day). Without tiered
   scoring and cached mode this plan would be infeasible.

5. **The speed improvement is dramatic.** ~20 seconds per round vs ~20 minutes. Eliminating
   LLM calls from the scoring pipeline removes the primary bottleneck and largest cost.
   Template evolution runs as a separate async task, not blocking the scoring loop.

### Remaining Risks

1. **Cold start problem.** The combiner needs ≥ 10 shared periods (MIN_SHARED_PERIODS)
   to produce meaningful weights. At one round per 30 seconds, that's 5 minutes. But
   forward returns for per-category horizons need h bars to resolve. For fundamental
   signals (h=390 bars = 3 days), meaningful R(i,s) data requires ~2 weeks of operation
   before the combination engine can properly weight fundamental signals. During cold
   start, equal weights (1/N) are used — which is still better than the current system
   because you have 50 scores vs 2-3 effective scores.

2. **Cached option chain data quality.** Cached mode returns data that may be up to
   15 minutes stale (for non-OPRA-entitled users) or a snapshot that lags real-time.
   For signal scoring (directional bias), this is acceptable — IV rank and skew change
   slowly. For execution timing, the agent still uses real-time quotes.

3. **Fundamental signal staleness during market hours.** yfinance data (300-600s TTL)
   means fundamental signals 24-35 are effectively static within a trading session.
   That's fine — fundamental signals SHOULD be slow-moving. The per-category horizon
   (h=390 for fundamentals) correctly reflects this time scale.

4. **Template evolution sample size.** With 10 templates, walk-forward validation
   requires sufficient trade history per template per regime. At `MIN_TRADES = 30` per
   template, all 10 templates need 300 total trades to have enough data. At, say,
   5 trades per day, that's 60 trading days (~3 months) before template boundaries are
   well-calibrated. Until then, default boundaries apply.

5. **Complexity budget.** 50 signal files + combiner + template selector + template
   evolution + new briefing = significant new code. Each signal is simple (~50-100 lines),
   but the combiner (Steps 1-11) is algorithmically dense. The recommendation is to
   implement Phase 0 + Phase 1 (13 price signals) + Phase 6 (combiner) first, validate
   end-to-end, then add categories incrementally. The execution order diagram already
   supports this.

### Verdict

**The plan is implementable and the expected improvement is real.** The information ratio
improvement from N_eff ≈ 3 to N_eff ≈ 20-30 is the single largest potential gain
available in the system architecture. The existing data layer, simulation infrastructure,
and DB patterns are sufficient — no new external dependencies are needed. The API budget
is viable with cached-mode option chains.

The primary risk is not technical but temporal: the system needs 2-3 weeks of runtime
data to calibrate combination weights for slow-moving categories, and 2-3 months to
calibrate template boundaries. During this warm-up period, the system operates with
equal weights and default boundaries — which is still a structural improvement over
the current slot system because 50 diverse scores averaged equally beat 12 correlated
scores averaged by decorative Darwinian weights.

---

## Final Integration Audit

Full codebase audit of every file the plan touches, depends on, or deletes. Issues
ranked by severity: CRITICAL = will crash/break at runtime, IMPORTANT = plan is
wrong/incomplete but won't crash, NOTE = useful context.

### CRITICAL — Must Fix Before Implementation

**C1. Method name wrong: `get_bulk_quotes()` does not exist.**
Plan says `get_bulk_quotes()`. Actual method is `get_quotes_bulk(symbols)` on DataProvider.
Returns `Dict[str, Quote]`. Fix: use `get_quotes_bulk()` everywhere.

**C2. Method does not exist: `get_bulk_candles()` not on DataProvider.**
`MarketDataClient` has `get_bulk_daily_candles(symbols)` but `DataProvider` does NOT
wrap it. The scorer has no way to bulk-fetch candles through the data layer.
Fix: add `get_candles_bulk()` wrapper to DataProvider before Phase 1, or call
MarketDataClient directly (breaks the abstraction layer).

**C3. Method name wrong: `get_iv_rank()` → actual is `get_iv_info()`.**
Plan calls `get_iv_rank(symbol)`. Actual method on DataProvider is:
```python
get_iv_info(self, symbol, dte_min=None, dte_max=None, strike_pct=None) -> Optional[IVInfo]
```
Returns `IVInfo` dataclass with `iv_current`, `iv_rank`, `iv_high`, `iv_low`, `source`.
**Requires `dte_min` and `dte_max`** — returns `None` if omitted.
Fix: call `get_iv_info(symbol, dte_min=20, dte_max=50)` with explicit DTE args.

**C4. IV rank field is always `None`.**
The MDA client's `get_iv_rank()` hardcodes `iv_rank: None` with comment "Would need
historical IV data". Only `iv_current` works. Signal 15 (IV rank) and Signal 18
(IV rank momentum) have no data source.
Fix: compute IV percentile ourselves — store daily `iv_current` snapshots in a new
`iv_history` table, compute rank as percentile vs trailing 252 days. Or use the
`iv_high`/`iv_low` from `IVInfo` to approximate rank = (current - low) / (high - low).

**C5. `compute_environment()` input shape is nested dict, not flat.**
Actual signature: `compute_environment(universe: dict[str, dict[str, pd.DataFrame]])`
Input: `{symbol: {date_str: candle_df}}` — nested by date, NOT `{symbol: df}`.
The scorer cannot just pass `{symbol: df}` — it will crash.
Fix: either use `compute_session_environment(session_universe, session_date)` which
takes `{symbol: single_df}`, or reshape data to match the nested format. For live
scoring, `compute_session_environment()` is likely the right choice.

**C6. Two different environment functions exist — plan conflates them.**
`compute_environment()` — multi-day, richer output (has `cross_asset_correlation`,
`per_symbol`, `momentum_candidates`, etc.).
`compute_session_environment()` — single-day intraday snapshot, simpler output,
different input shape `{symbol: single_df}`.
Which one the scorer should use depends on context:
- Template evolution (walk-forward) → `compute_environment()` (multi-day)
- Live scoring → `compute_session_environment()` (today only)
Fix: plan must specify which function is used where.

**C7. `ReplayHarness.run()` is async — must be awaited.**
Plan says template evolution "reuses replay.py `ReplayHarness.run()`" but doesn't note
it's `async def run(...)`. Template evolution code must be async-compatible.
Fix: template_evolution.py must run inside an async context (it already will since
it's an `asyncio.gather()` task, but individual calls must use `await`).

**C8. Agent briefing: `_get_research_briefing()` is dead code.**
Plan says the agent uses `_get_research_briefing()` to read from DB. In reality, the
agent's `run_cycle()` just tells the LLM to "call briefing()" — which routes to
`handle_briefing()` in tools_research.py. `_get_research_briefing()` exists but is
**never called** anywhere in the codebase.
Fix: the plan should replace `handle_briefing()` and its 5 sub-functions, not
`_get_research_briefing()`. The agent's behavior is driven by the tool, not the method.

**C9. There are 5 briefing sub-functions, not 3.**
Plan says to replace `_briefing_summary()`, `_briefing_signals()`, `_briefing_strategies()`.
Actual sub-functions called by `handle_briefing()`:
1. `_briefing_summary(data)` — regime-filtered actionable signals
2. `_briefing_signals(data)` — full signal list (up to 30)
3. `_briefing_strategies(data)` — strategy slot table
4. `_briefing_feedback(data)` — trade feedback with confidence intervals + execution costs
5. `_briefing_environment(data)` — full regime + strategy fit + env history
The agent drills down via `detail` parameter: `summary|signals|strategies|feedback|environment`.
Fix: plan must replace ALL 5, plus `_query_briefing_data()` which feeds them all.
`_briefing_feedback()` should be preserved — it uses `compute_sample_confidence()`
and `get_execution_cost()` which remain useful.

**C10. `risk_per_trade_pct` is a per-call parameter, not a config variable.**
Plan says "modify `risk_per_trade_pct`" for CV adjustment. But it's a tool parameter:
```python
risk_per_trade_pct = float(params.get("risk_per_trade_pct", 1.5))
```
Default is hardcoded 1.5%. Not read from `core.config.RISK_PER_TRADE` (which is 1.0%
paper / 0.5% live) — those are never used by the sizing tool.
Fix: inject CV adjustment by modifying the default before `params.get()`:
`adjusted_risk = 1.5 * (1 - cv_edge)` then use that as the default.

### IMPORTANT — Plan Inaccuracies to Correct

**I1. DB table count is 17, not 18.**
`init_db()` creates 17 tables across 3 `executescript` blocks. The plan says 18.
Not a bug, but should be accurate.

**I2. Plan omits ~17 config variables in research/config.py.**
Variables the plan doesn't mention that need decisions during implementation:
- `FORCE_EXIT_MINUTE = 955` — still needed (position flattening time)
- `SLIPPAGE_BPS = 2` — still needed (simulator uses it)
- `MIN_KEEP_TEST_SIGNALS = 15`, `MIN_KEEP_CONFIDENCE_SCORE = 0.25`,
  `MIN_KEEP_CONDITION_TRADES = 3` — promotion gates, may need equivalents
- `SANDBOX_ALLOWED_IMPORTS`, `SANDBOX_BLOCKED_CALLS`, `STRATEGY_EXEC_TIMEOUT` — delete
  (no more sandbox execution)
- `REGIME_GATE_ENABLED = True` — delete (templates handle regime matching)
- `EXTREME_VOL_FITNESS_BOOST = 1.05` — delete (combiner handles this)
- Circuit breaker constants — may need equivalents for signal scoring
- `ROUND_DELAY_SECS = 30` — keep (pacing between research rounds)
- `RESEARCH_DAILY_LLM_BUDGET = 25.0` — delete (no more LLM in research)
- `LEGS_JSON_SCHEMAS` — keep (template selector needs leg construction schemas)
- `STOCK_ORDER_TYPES`, `OPTIONS_STRATEGIES` — keep (template definitions reference them)
- `SIGNAL_SCHEMA` — delete (replaced by Signal base class)
- All 3 prompt templates — delete (no more LLM in research)
- `validate_legs_json()` function — keep (promoter and template selector use it)

**I3. Deleting 7 functions orphans ~30 of 42 total functions in research/agent.py.**
The file needs near-complete replacement, not surgical deletion. Only ~5 functions are
called from outside the deletion targets:
- `run_research()` itself (called from `__main__.py` and `research/__main__.py`)
- `_fetch_universe_candles()` — useful for new scorer
- `_store_environment_snapshot()` — new scorer reuses this
- `_get_env_slot_history()` — called from tools_research.py briefing
- `_get_trade_feedback()` / `_match_trades_to_signals()` — called from tools
Fix: plan Phase 10 should say "replace research/agent.py with new scorer loop,
keeping only: `_fetch_universe_candles`, `_store_environment_snapshot`,
`_get_trade_feedback`, `_match_trades_to_signals`."

**I4. `compute_expectancy()` returns many more fields than plan mentions.**
Beyond hit_rate and Sharpe, it returns: `search_fitness`, `signed_edge_score`,
`condition_confidence`, `condition_metrics` (per-environment-bucket analysis),
`stability_score`, `return_std`, `stderr`, `ci95_low/high`, `t_stat`,
`confidence_score`, `luck_pressure`, `raw_search_fitness`.
Template evolution should use `search_fitness` (composite capped at 5.0) as the
primary fitness metric — it's already calibrated for the promotion pipeline.

**I5. `reprice_signals()` returns `list[SignalReprice]` dataclass, not dict.**
Callers must use attribute access: `.outcome`, `.return_pct`, `.data_coverage`.
The `SignalReprice` has `.legs` (list of `LegValuation` with per-leg P&L,
fill_policy, data_quality).

**I6. `score_repriced_signals(repriced)` standalone function not mentioned.**
Aggregates `list[SignalReprice]` into promotion fitness dict with: `promotion_fitness`,
`options_coverage_pct`, `repriced_count/total_count`, `missing_breakdown_text`,
`hit_rate`, `expectancy`, `profit_factor`. Template evolution should use this for
options template fitness scoring.

**I7. Simulator uses calibrated slippage from live execution data.**
`simulate(signals, candles, slippage_preset="strict")` loads real execution gaps via
`memory.get_calibrated_slippage()`. Template evolution inherits this calibration
automatically when using strict mode. Plan should note this: evolution simulations
get more accurate over time as real trade data accumulates.

**I8. `asyncio.gather()` has no error handling — third task needs crash isolation.**
Current `_run_both()` uses bare `asyncio.gather()` with no `return_exceptions`.
A crash in any task kills all tasks. Adding `run_template_evolution()` as third task
requires either:
- `return_exceptions=True` + check results for exceptions
- Or wrap each task in try/except with logging
Otherwise a template evolution crash takes down the trading agent.

**I9. Useful DataProvider methods not mentioned in plan.**
- `get_earnings_info(symbol)` → `EarningsInfo` with `next_earnings_date`, `days_until_earnings`
  — critical for Signal 24 (earnings momentum) and Signal 39 (event proximity)
- `get_earnings_history(symbol, countback=8)` → EPS surprise history
  — directly useful for Signal 24
- `get_peer_comparison(symbol)` → `PeerComparison` with sector ETF returns
  — directly useful for Signal 37 (sector relative strength)
- `get_atr_percent(symbol, period=14)` → float (ATR as % of price)
  — useful for template boundary matching
- `find_option_by_delta(symbol, target_delta, side, min_dte, max_dte)` → `OptionContract`
  — useful for template leg construction (vertical spreads, etc.)

**I10. `compute_environment_by_date(universe)` returns per-date snapshots.**
Returns `{date_str: session_snapshot}`. More useful than aggregate `compute_environment()`
for walk-forward template evolution — gives environment context per historical period.
Template evolution should use this to match historical trades to their regime context.
