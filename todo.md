# ABC — TODO

## Signal Combination Engine + Execution Template Architecture

### Problem Statement
The current 12-slot system conflates two questions: "what direction?" and "how to structure the trade?" All 12 slots answer both from the same OHLCV candle data, producing correlated opinions that masquerade as independent signals. Darwinian weights influence which slots evolve but NOT which signals the agent trades — the briefing sorts by raw `s.expectancy DESC` and the agent never sees slot weights, correlation data, or env-adjusted fitness. Effective N_eff is likely 2-3, not 12.

The Fundamental Law of Active Management: IR = IC × √N. N is the effective number of independent signals after shared variance is removed. The fix is structural: build genuinely independent signals from separate data sources, combine them mathematically into a composite score per symbol, then select execution templates mechanically based on the composite + market conditions.

### Architecture Overview
```
┌──────────────────────────────────────────────────────────────┐
│  SIGNAL LAYER  (signals/)                                    │
│  Each signal: {symbol, score: -1.0 to +1.0, confidence: 0-1}│
│                                                              │
│  PRICE SIGNALS (from MDA candles — 1 consolidated, not 12)   │
│    1. Trend/momentum    — EMA cross, ROC, MACD direction     │
│    2. Mean reversion    — RSI + Bollinger Band deviation     │
│    3. Breakout          — range high/low breach + volume     │
│    4. VWAP deviation    — distance from intraday VWAP        │
│    5. Volume profile    — surge vs contraction vs avg        │
│                                                              │
│  VOLATILITY SIGNALS (from MDA option chains)                 │
│    6. IV vs RV spread   — implied vol minus realized vol     │
│    7. IV rank           — current IV percentile (0-100)      │
│    8. Skew              — put IV vs call IV imbalance        │
│    9. Term structure    — front-month vs back-month IV       │
│                                                              │
│  FUNDAMENTAL SIGNALS (from yfinance)                         │
│   10. Earnings momentum — surprise %, estimate revisions     │
│   11. Valuation         — P/E, P/S, PEG vs sector median    │
│   12. Quality           — ROE, debt/equity, FCF margin       │
│   13. Short interest    — short % float, short ratio         │
│   14. Insider flow      — net insider buys/sells ($ value)   │
│   15. Analyst consensus — mean recommendation, target delta  │
│                                                              │
│  MACRO/SENTIMENT SIGNALS                                     │
│   16. Market breadth    — advance/decline ratio              │
│   17. Sector momentum   — symbol return vs sector ETF        │
│   18. Cross-correlation — macro regime (corr > 0.6) vs idio  │
│   19. Economic proximity— distance to next high-impact event │
│   20. News sentiment    — MDA news sentiment score           │
│                                                              │
│  MICROSTRUCTURE SIGNALS (from MDA quotes + option flow)      │
│   21. Bid-ask spread    — tightening vs widening             │
│   22. Option OI change  — call vs put OI buildup direction   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  COMBINATION ENGINE  (signals/combiner.py)                   │
│                                                              │
│  1. Collect score time series per signal per symbol          │
│  2. Demean each signal's series (remove drift)               │
│  3. Normalize by per-signal σ (common scale)                 │
│  4. Cross-sectionally demean at each period (remove market)  │
│  5. Compute forward expected score via d-day MA              │
│  6. Regress out shared variance → residuals ε(i)            │
│  7. Set weight w(i) = ε(i) / σ(i)                          │
│  8. Normalize: Σ|w| = 1                                     │
│  9. Composite_score(symbol) = Σ w(i) × score(i, symbol)    │
│ 10. Persist weights + N_eff to DB                            │
│                                                              │
│  Output: {symbol: composite_score} for entire universe       │
│  composite_score ∈ [-1.0, +1.0]                             │
│  N_eff = (Σλ)² / Σλ²  (eigenvalue decomposition)           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  EXECUTION TEMPLATE SELECTOR  (signals/templates.py)         │
│                                                              │
│  Input: composite_score, vol_regime, iv_rank, account_state  │
│  Output: structured trade dict (same format as old signals)  │
│                                                              │
│  Decision matrix:                                            │
│    High conviction (+/-0.6+) + low IV  → stock bracket       │
│    High conviction + high IV           → vertical spread     │
│    Moderate conviction (+/-0.3-0.6)    → defined-risk spread │
│    Low conviction + high IV            → premium selling     │
│    Low conviction + low IV             → no trade            │
│                                                              │
│  Self-improvement: every executed trade is scored.           │
│  Track per-template: win_rate, avg_return, avg_gap.          │
│  Templates that underperform get demoted. Templates that     │
│  outperform in specific regimes get preferred in those       │
│  regimes. DB table: template_performance.                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  BRIEFING → AGENT  (replaces old briefing pipeline)          │
│                                                              │
│  SIGNAL QUALITY:                                             │
│    N_eff: 14.2 of 22 signals (independence ratio: 64%)       │
│    Weight leaders: iv_rank (0.12), momentum (0.09), ...      │
│    Redundant pairs: momentum↔breakout (r=0.74)              │
│                                                              │
│  TOP COMPOSITE SCORES:                                       │
│    NVDA +0.72 [mom=0.8 iv_rank=-0.3 earnings=0.9 ...]       │
│    TSLA -0.58 [mom=-0.7 short_int=0.6 insider=-0.4 ...]     │
│                                                              │
│  RECOMMENDED TRADES (template-selected):                     │
│    NVDA long vertical_spread @ 890 tgt=905 stp=880           │
│      template: high_conviction_high_iv (72% win, 14 trades)  │
│    TSLA short bracket @ 245 tgt=238 stp=250                  │
│      template: high_conviction_low_iv (65% win, 9 trades)    │
│                                                              │
│  TEMPLATE PERFORMANCE (last 30 days):                        │
│    stock_bracket: 65% win, avg +0.8%, 23 trades              │
│    vertical_spread: 58% win, avg +0.5%, 18 trades            │
│    premium_selling: 70% win, avg +0.3%, 12 trades            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 0: Foundation — signals/ Module + DB Schema
- [ ] Create `signals/__init__.py` — module init, signal registry
- [ ] Create `signals/base.py` — `Signal` base class: `score(symbol, data) → {score: float, confidence: float, components: dict}` interface. Each signal self-describes: name, category, data_source, refresh_rate
- [ ] Create DB tables:
  - `signal_scores` — (signal_name, symbol, ts, score, confidence, components_json)
  - `signal_weights` — (signal_name, weight, n_eff, updated_ts)
  - `composite_scores` — (symbol, ts, composite_score, signal_breakdown_json)
  - `template_performance` — (template_name, regime, trades, wins, avg_return, avg_gap, updated_ts)
- [ ] Add constants to `research/config.py`:
  - `MIN_SHARED_PERIODS_FOR_COMBINATION = 10`
  - `SIGNAL_WEIGHT_LOOKBACK_DAYS = 10`
  - `COMPOSITE_TRADE_THRESHOLD = 0.25` (minimum |composite| to generate a trade)
  - `CV_BOOTSTRAP_SAMPLES = 1000`

### Phase 1: Price Signals (candle-derived, 5 signals)
Consolidates what all 12 slots currently compute independently into 5 distinct measured quantities. Source: MDA candles (already fetched for universe each round).

- [ ] `signals/momentum.py` — Signal 1: Trend/momentum score
  - EMA(10) vs EMA(50) cross direction and distance
  - 5-bar and 10-bar ROC
  - MACD histogram sign and slope
  - Score: +1.0 (strong uptrend) to -1.0 (strong downtrend)
  - Confidence: based on trend consistency across timeframes

- [ ] `signals/mean_reversion.py` — Signal 2: Mean reversion score
  - RSI(14) distance from 50 (oversold < 30, overbought > 70)
  - Bollinger Band position (% within bands)
  - Distance from 20-day SMA in ATR units
  - Score: +1.0 (deeply oversold = long setup) to -1.0 (deeply overbought = short setup)
  - Confidence: based on how many reversion indicators agree

- [ ] `signals/breakout.py` — Signal 3: Breakout score
  - Distance from 20-bar high/low
  - Volume confirmation (current vs 20-bar average)
  - Range expansion (today's range vs 5-day avg range)
  - Score: +1.0 (breaking above range on volume) to -1.0 (breaking below on volume)
  - Confidence: volume multiplier and range expansion strength

- [ ] `signals/vwap.py` — Signal 4: VWAP deviation score
  - Price distance from intraday VWAP in ATR units
  - Direction of VWAP slope
  - Score: +1.0 (far above VWAP, momentum) or contextual mean-reversion read
  - Confidence: volume-weighted distance significance

- [ ] `signals/volume.py` — Signal 5: Volume profile score
  - Relative volume (current bar vs 20-bar avg)
  - Volume trend (expanding vs contracting over 5 bars)
  - On-balance volume direction
  - Score: +1.0 (strong accumulation) to -1.0 (strong distribution)
  - Confidence: consistency of volume direction

### Phase 2: Volatility Signals (options-derived, 4 signals)
Source: MDA option chains + iv_rank endpoint. Structurally independent from price data — measures the options market's expectations, not price itself.

- [ ] `signals/iv_rv_spread.py` — Signal 6: IV vs realized vol spread
  - get_iv_rank() for current IV
  - Compute realized vol from last 20 days of daily candle returns
  - Spread = IV - RV (positive = options overpriced = sell premium edge)
  - Score: +1.0 (IV >> RV, sell premium) to -1.0 (IV << RV, buy options)
  - Confidence: magnitude of spread relative to historical range

- [ ] `signals/iv_rank.py` — Signal 7: IV rank signal
  - IV percentile (0-100) from get_iv_rank()
  - Score: maps IV rank to premium-selling opportunity (+1.0 at rank 90+) vs cheap-options opportunity (-1.0 at rank 10-)
  - Confidence: stability of IV rank over trailing 5 days

- [ ] `signals/skew.py` — Signal 8: Volatility skew signal
  - Put ATM IV vs Call ATM IV from option chain
  - Steepening skew = fear = bearish signal
  - Score: +1.0 (extreme call skew = bullish flow) to -1.0 (extreme put skew = bearish flow)
  - Confidence: magnitude vs 30-day avg skew

- [ ] `signals/term_structure.py` — Signal 9: IV term structure signal
  - Front-month IV vs second-month IV (from option chain at two expirations)
  - Backwardation (front > back) = event/stress = high near-term uncertainty
  - Contango (front < back) = normal
  - Score: +1.0 (steep contango = calm) to -1.0 (backwardation = stress)
  - Confidence: magnitude of slope

### Phase 3: Fundamental Signals (yfinance-derived, 6 signals)
Source: yfinance fundamentals, analyst, insider data. Entirely independent from price/vol — measures company quality and institutional behavior. Updates slowly (300-600s TTL).

- [ ] `signals/earnings.py` — Signal 10: Earnings momentum
  - Last earnings surprise % (surprise_eps_pct from MDA)
  - Days until next earnings (proximity risk)
  - Score: +1.0 (large positive surprise, not near next earnings) to -1.0 (large miss)
  - Confidence: recency of last report and magnitude of surprise

- [ ] `signals/valuation.py` — Signal 11: Relative valuation
  - P/E, P/S, PEG ratio vs sector median (from extended_fundamentals + peer_comparison)
  - Score: +1.0 (significantly undervalued) to -1.0 (significantly overvalued)
  - Confidence: number of metrics that agree on direction

- [ ] `signals/quality.py` — Signal 12: Company quality
  - ROE, debt_to_equity, profit_margin, free_cash_flow from extended_fundamentals
  - Score: +1.0 (high quality across all metrics) to -1.0 (deteriorating quality)
  - Confidence: consistency across quality dimensions

- [ ] `signals/short_interest.py` — Signal 13: Short interest pressure
  - Short % float, short ratio (days to cover) from extended_fundamentals
  - Score: +1.0 (high short interest = squeeze potential if momentum aligns) to -1.0 (low/declining short interest = less fuel)
  - Confidence: extremeness of short metrics

- [ ] `signals/insider.py` — Signal 14: Insider flow
  - Net insider buy/sell $ value from insider_data
  - Recent buy/sell count and net_sentiment
  - Score: +1.0 (heavy insider buying) to -1.0 (heavy insider selling)
  - Confidence: total $ value and number of insiders acting

- [ ] `signals/analyst.py` — Signal 15: Analyst consensus
  - Mean recommendation (1-5 scale), recent upgrades vs downgrades, target upside %
  - Score: +1.0 (strong buy consensus + large upside) to -1.0 (sell consensus + downside)
  - Confidence: number of analysts and agreement level

### Phase 4: Macro/Sentiment Signals (cross-asset, 5 signals)
Source: environment.py metrics, economic_calendar, MDA news. Measures market-wide conditions rather than single-stock technicals.

- [ ] `signals/breadth.py` — Signal 16: Market breadth
  - Advance/decline ratio from compute_environment()
  - % trending up vs down across universe
  - Score: +1.0 (broad bullish breadth) to -1.0 (broad bearish breadth)
  - Confidence: dispersion (low dispersion + directional breadth = high confidence)

- [ ] `signals/sector_momentum.py` — Signal 17: Relative sector strength
  - Symbol return vs sector ETF return over 5, 10, 20 days
  - Score: +1.0 (outperforming sector by large margin) to -1.0 (underperforming)
  - Confidence: consistency across lookback periods

- [ ] `signals/correlation_regime.py` — Signal 18: Correlation regime
  - Cross-asset correlation from compute_environment()
  - High correlation (>0.6) = macro-driven (signals less symbol-specific)
  - Low correlation (<0.3) = idiosyncratic (stock-picking environment)
  - Score: +1.0 (idio regime = stock-specific signals valid) to -1.0 (macro regime = defer to market direction)
  - Confidence: stability of correlation level over trailing 3 days

- [ ] `signals/event_proximity.py` — Signal 19: Economic event risk
  - Distance in hours to next high-impact macro event (from economic_calendar)
  - FOMC, CPI, NFP, GDP
  - Score: 0 (far from event, neutral) to -1.0 (within 24h of high-impact event = reduce risk)
  - Confidence: impact level of upcoming event

- [ ] `signals/news_sentiment.py` — Signal 20: News sentiment
  - Aggregate sentiment from get_news() (pos/neg/neutral counts, recency-weighted)
  - Score: +1.0 (overwhelmingly positive recent news) to -1.0 (overwhelmingly negative)
  - Confidence: volume of recent news and sentiment consistency

### Phase 5: Microstructure Signals (flow-derived, 2 signals)
Source: MDA quotes + option chain OI. Measures positioning and liquidity information.

- [ ] `signals/spread.py` — Signal 21: Bid-ask spread dynamics
  - Current spread vs trailing average spread
  - Narrowing = low information asymmetry, widening = someone trading on information
  - Score: +1.0 (tightening = safe) to -1.0 (widening = caution)
  - Confidence: magnitude of change vs recent history

- [ ] `signals/option_flow.py` — Signal 22: Option open interest flow
  - Call OI vs Put OI change (requires comparing two snapshots)
  - Increasing call OI = bullish positioning; increasing put OI = bearish/hedging
  - Score: +1.0 (call OI building up) to -1.0 (put OI building up)
  - Confidence: magnitude of OI change and volume confirmation

### Phase 6: Combination Engine
- [ ] `signals/combiner.py` — The article's 11-step combination engine
  - Collect score time series per signal per symbol from signal_scores table
  - Step 1: Gather R(i,s) — signal score history per signal per period
  - Step 2: Demean each signal's series (remove drift)
  - Step 3: Calculate sample variance σ(i)²
  - Step 4: Normalize: Y(i,s) = X(i,s) / σ(i)
  - Step 5: Drop most recent observation (prevent lookahead)
  - Step 6: Cross-sectionally demean at each period: Λ(i,s) = Y(i,s) - avg(Y(j,s))
  - Step 7: Drop last period from Λ
  - Step 8: Forward expected score via d-day MA, normalized
  - Step 9: Regress E_normalized over Λ → residuals ε(i) are independent contribution
  - Step 10: w(i) = η × ε(i) / σ(i)
  - Step 11: Normalize Σ|w| = 1
  - Compute N_eff = (Σλ)² / Σλ² from correlation matrix eigenvalues
  - When < MIN_SHARED_PERIODS: equal weights 1/N
  - Persist weights + N_eff to signal_weights table
  - Compute composite_score(symbol) = Σ w(i) × score(i, symbol)
  - Persist to composite_scores table

- [ ] `signals/scorer.py` — Orchestrator that runs all signals for the universe
  - For each symbol: call each signal's score() method
  - Write results to signal_scores table
  - Call combiner to produce composite scores
  - Triggered: once per research round (replacing slot evaluation)
  - Handles caching: price signals refresh each round, fundamental signals only when stale

### Phase 7: Execution Template Selector
- [ ] `signals/templates.py` — Mechanical trade structure selection
  - Input: composite_score, iv_rank, vol_regime, account_state (cash, net_liq, existing positions)
  - Templates:
    - `stock_bracket` — direct stock + bracket stop/target. For: |composite| > 0.6 AND iv_rank < 40
    - `vertical_spread` — directional defined-risk. For: |composite| > 0.4 AND iv_rank > 40
    - `premium_selling` — iron condor/credit spread. For: |composite| < 0.3 AND iv_rank > 60
    - `long_options` — debit spread/long call/put. For: |composite| > 0.5 AND iv_rank < 25
    - `no_trade` — below threshold. For: |composite| < COMPOSITE_TRADE_THRESHOLD
  - Output: trade dict (same format: entry_price, target_price, stop_price, order_type, legs_json, direction)
  - Stop/target calculated from ATR (same as existing slots)

- [ ] Template performance tracking:
  - On each trade fill: record template used, regime, composite score at entry
  - On each trade close: record P&L, execution gap
  - Aggregate: per-template win_rate, avg_return, avg_gap by regime
  - Regime-specific preferences: if `stock_bracket` outperforms `vertical_spread` when vol=low AND |composite|>0.6, prefer it in that regime
  - Stored in template_performance table
  - Surfaced in briefing so agent and future optimization can see what works

### Phase 8: Wire Into Agent Briefing
- [ ] Replace `_briefing_summary()` in `tools/tools_research.py`:
  - Remove slot-based signal ranking
  - Add signal quality section (N_eff, weight leaders, redundant pairs)
  - Add composite scores section (top symbols with per-signal breakdown)
  - Add recommended trades section (template-selected)
  - Add template performance section

- [ ] Replace `_briefing_signals()`:
  - Show composite-score-ranked trades instead of raw slot signals
  - Each trade shows: composite score, template used, template track record
  - Conflict resolution is automatic — composite already resolves direction

- [ ] Modify `_query_briefing_data()`:
  - Query composite_scores instead of live_signals
  - Query signal_weights for N_eff and weight breakdown
  - Query template_performance for track record

### Phase 9: Kelly Sizing with CV Adjustment
- [ ] Add `_estimate_cv_edge()` in `tools/tools_sizing.py`
  - Bootstrap-resample matched trade returns 1000×
  - Compute edge (avg return) each iteration
  - Return coefficient of variation (CV = σ_edge / μ_edge)
  - Requires ≥20 matched fills to activate; otherwise CV = 0

- [ ] Modify `handle_calculate_size()`:
  - `risk_per_trade_pct *= (1 - cv_edge)`
  - Higher CV (less certain about edge) → smaller positions
  - Lower CV (consistent edge) → full-sized positions
  - Binding constraint logic (min of risk/concentration/cash) unchanged

### Phase 10: Remove Old Slot System
- [ ] Delete `research/slots/` directory (all 12 strategy files)
- [ ] Delete from `research/agent.py`:
  - `_update_darwinian_weights()` and all Darwinian weight logic
  - `_compute_slot_correlations()` and Pearson-r pair detection
  - `_run_slot()` loop and slot-based evaluation
  - `_run_live_scan()` per-slot live signal generation
  - `_execute_strategy()` sandbox strategy execution
  - Slot-based selector agent
  - All SLOT_MANDATES references
- [ ] Delete from `research/config.py`:
  - `SLOT_MANDATES`, `REGIME_TO_SLOTS`, `NUM_SLOTS`, `BATCH_SIZE`
  - `DARWINIAN_WEIGHT_CEILING/FLOOR/UP/DOWN`, `SLOT_CORRELATION_THRESHOLD`
  - `SELECTOR_EVERY_N_ROUNDS`, `MAX_SELECTOR_REPLACEMENTS`
- [ ] Replace `run_research()` main loop:
  - Fetch candles for universe
  - Compute environment snapshot
  - Run signal scorer (all 22 signals across universe)
  - Run combiner (produce composite scores)
  - Run template selector (produce recommended trades)
  - Write to DB (signal_scores, composite_scores, template recommendations)
  - Log round summary with N_eff

- [ ] Drop old DB tables:
  - `live_signals` — replaced by composite_scores + template recommendations
  - `strategies` — no more evolving strategy code
  - `slot_environment_scores` — replaced by signal_weights

### Phase 11: Tests
- [ ] Unit tests for each signal: known input candles/data → expected score range
- [ ] Unit test combiner: synthetic signal series → correct weights and N_eff
  - Identity correlation matrix → N_eff = N
  - Perfect correlation → N_eff = 1
  - Known synthetic returns → predictable weight ordering
- [ ] Unit test template selector: composite + regime → correct template choice
- [ ] Unit test CV estimation: known distribution → expected CV
- [ ] Integration test: full pipeline (signals → combiner → templates → briefing)
- [ ] Regression test: sizing unchanged when < 20 fills (CV inactive)

---

## Signal Category Independence Matrix

The 22 signals draw from 5 structurally independent data sources:

| Category | Data Source | Signals | Updates |
|----------|-----------|---------|---------|
| Price (5) | MDA candles OHLCV | momentum, mean_reversion, breakout, vwap, volume | Every round (~30s) |
| Volatility (4) | MDA option chains + IV | iv_rv_spread, iv_rank, skew, term_structure | Every round (~30s) |
| Fundamental (6) | yfinance | earnings, valuation, quality, short_interest, insider, analyst | Every 5-10 min |
| Macro/Sentiment (5) | environment + calendar + news | breadth, sector_momentum, correlation_regime, event_proximity, news_sentiment | Every round |
| Microstructure (2) | MDA quotes + option OI | bid_ask_spread, option_flow | Every round |

Within each category, signals may be correlated (momentum and breakout both use price). Across categories, signals are structurally independent (IV rank does not derive from candle price, insider buying does not derive from IV). The combination engine's cross-sectional demeaning and regression will correctly measure and account for within-category correlation, giving the full √N benefit only for truly independent contributions.

Expected effective N_eff: 10-16 of 22 (within-category correlation will reduce effective count within price and fundamental groups, but cross-category independence is structural).

---

## Execution Order and Dependencies

```
Phase 0 ──→ Phase 1 ──→ Phase 6 (can test combiner with 5 price signals)
              │
Phase 0 ──→ Phase 2 ──→ Phase 6 (add vol signals, measure N_eff improvement)
              │
Phase 0 ──→ Phase 3 ──→ Phase 6 (add fundamental signals)
              │
Phase 0 ──→ Phase 4 ──→ Phase 6 (add macro/sentiment signals)
              │
Phase 0 ──→ Phase 5 ──→ Phase 6 (add microstructure signals)
              │
Phase 6 ──→ Phase 7 ──→ Phase 8 ──→ Phase 10 ──→ Phase 11
              │
Phase 9 is independent (sizing changes only)
```

Phase 1 through Phase 6 can run in parallel with the existing slot system — both can coexist. The signal layer writes to its own tables. Phase 8 (briefing rewire) and Phase 10 (slot removal) are the cutover points. This allows validating that composite scores produce sensible rankings BEFORE removing the slot system.

---

## What Gets Deleted

| Component | Replaced By |
|-----------|-------------|
| 12 strategy slot files (`research/slots/`) | 22 signal generators (`signals/`) |
| `_update_darwinian_weights()` | Combination engine weights |
| `_compute_slot_correlations()` | Correlation matrix + N_eff |
| `_run_slot()` evolution loop | Signal scorer orchestrator |
| `_run_live_scan()` | Template selector |
| `_execute_strategy()` sandbox | Direct signal.score() calls |
| Selector agent (LLM-based slot replacement) | Mathematical weight optimization |
| SLOT_MANDATES, REGIME_TO_SLOTS | Template decision matrix |
| `live_signals` table | `composite_scores` + template recommendations |
| `strategies` table | `signal_scores` + `signal_weights` |
| `slot_environment_scores` | `signal_weights` |

## What Stays Unchanged

| Component | Why |
|-----------|-----|
| Simulator (`research/simulator.py`) | Still used for template backtesting |
| Promoter (`research/promoter.py`) | Still reprices options-based templates |
| Replay (`research/replay.py`) | Still validates template logic |
| Execution layer (all `ibkr_*`) | No interface changes |
| Environment (`research/environment.py`) | Feeds macro signals + template regime selection |
| Data provider (`data/data_provider.py`) | All data access unchanged |
| Core agent loop (`core/agent.py`) | Consumes briefing text, format changes transparent |
| Cost tracker | Unchanged |
| Market hours | Unchanged |
