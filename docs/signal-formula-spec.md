# Signal formula spec

Maps each active signal to implementation style, article alignment, target hybrid formula, and fallbacks.

**See also:** [data-sources.md](data-sources.md) · [codebase-layout.md](codebase-layout.md) · [plain-english-glossary.md](plain-english-glossary.md)

---

This document maps each active signal to:
- current implementation style
- article concept alignment
- target hybrid formula
- required inputs and fallback behavior

Invariants for every signal after refactor:
- score is bounded to `[-1, 1]`
- confidence is bounded to `[0, 1]`
- components include enough diagnostics to explain score direction
- missing/invalid core inputs return neutral score with low confidence

## Price / Momentum Family

- `momentum`
  - current: EMA cross + ROC + MACD blend
  - article mapping: price/momentum expected-return signal
  - target hybrid: standardized multi-lookback return composite with volatility-scaled normalization and trend confirmation
  - inputs: candles
  - fallback: neutral on insufficient candles

- `mean_reversion`
  - current: RSI + Bollinger position + SMA distance
  - article mapping: mean reversion (currently mostly time-series)
  - target hybrid: z-scored distance from rolling fair value with optional cross-sectional residual hook
  - inputs: candles
  - fallback: neutral on insufficient candles

- `breakout`
  - current: range position + volume confirmation
  - article mapping: momentum/price continuation
  - target hybrid: breakout distance in ATR units with volume-adjusted conviction and false-break filter
  - inputs: candles
  - fallback: neutral if no valid range

- `vwap_deviation`
  - current: VWAP distance in ATR + slope
  - article mapping: short-horizon price trend/deviation
  - target hybrid: anchored VWAP deviation with trend-state gating and bounded tanh mapping
  - inputs: candles
  - fallback: neutral on sparse candles

- `volume_profile`
  - current: relative volume + trend + OBV
  - article mapping: price signal with participation proxy
  - target hybrid: participation-adjusted trend score where direction and participation are separated before recombination
  - inputs: candles
  - fallback: neutral on low volume history

- `beta_adjusted_momentum`
  - current: alpha versus SPY using beta proxy
  - article mapping: idiosyncratic momentum
  - target hybrid: residualized return over market return with robust beta estimation fallback
  - inputs: candles, spy_candles, fundamentals (optional beta)
  - fallback: beta defaults safely to 1.0 when unavailable

- `overnight_gap`
  - current: gap size + fill ratio
  - article mapping: momentum/reversion event response
  - target hybrid: gap z-score relative to rolling gap distribution plus fill-state transition logic
  - inputs: candles_daily, candles
  - fallback: neutral if daily gap cannot be computed

- `price_acceleration`
  - current: change in ROC across lookbacks
  - article mapping: momentum convexity
  - target hybrid: normalized second derivative of return with noise floor
  - inputs: candles
  - fallback: neutral on short history

- `volume_weighted_momentum`
  - current: returns weighted by relative volume
  - article mapping: participation-confirmed momentum
  - target hybrid: lookback return stack weighted by signed participation surprise
  - inputs: candles
  - fallback: neutral on bad volume baseline

- `multi_timeframe`
  - current: EMA alignment across inferred 5m/hour/day proxies
  - article mapping: multi-horizon momentum agreement
  - target hybrid: explicit per-horizon return agreement score using configured horizons instead of inferred resampling shortcuts
  - inputs: candles, candles_daily
  - fallback: degrade to available horizons with reduced confidence

- `opening_range`
  - current: position relative to opening range breakout
  - article mapping: intraday momentum micro-pattern
  - target hybrid: opening range break distance and retest logic with volatility normalization
  - inputs: candles
  - fallback: neutral when opening window cannot be inferred

- `trend_strength`
  - current: DMI/ADX approximation
  - article mapping: trend quality signal
  - target hybrid: directional strength with smoothed ADX-style denominator and saturation control
  - inputs: candles
  - fallback: neutral on invalid ATR path

- `support_resistance`
  - current: nearest swing support/resistance distance
  - article mapping: mean reversion / breakout boundary interaction
  - target hybrid: distance-to-boundary score with regime switch (trend vs range)
  - inputs: candles
  - fallback: neutral when swing structure absent

## Volatility Family

- `iv_rank`
  - current: linear mapping of IV rank
  - article mapping: volatility level signal
  - target hybrid: bounded nonlinear map emphasizing tails, confidence from percentile distance
  - inputs: iv_info
  - fallback: neutral if iv_rank unavailable

- `iv_rv_spread`
  - current: IV minus 20d RV
  - article mapping: volatility risk premium
  - target hybrid: IV-RV spread normalized by rolling spread volatility and capped
  - inputs: iv_info, candles_daily
  - fallback: neutral if RV window unavailable

- `term_structure`
  - current: front/back IV spread
  - article mapping: term-structure slope
  - target hybrid: DTE-aware slope normalized by tenor gap
  - inputs: option_chain
  - fallback: neutral on insufficient expiries

- `volatility_skew`
  - current: ATM put IV minus call IV
  - article mapping: skew risk pricing
  - target hybrid: strike-normalized skew using consistent delta buckets
  - inputs: option_chain
  - fallback: neutral on sparse chain

- `iv_rank_momentum`
  - current: heuristic from rank and range
  - article mapping: volatility regime momentum
  - target hybrid: explicit change metric from persisted IV history with fallback to range position proxy
  - inputs: iv_info, iv_history (preferred)
  - fallback: proxy mode when history sparse

- `realized_vol_cone`
  - current: 20d RV percentile in rolling history
  - article mapping: realized volatility regime level
  - target hybrid: cone percentile plus trend of percentile
  - inputs: candles_daily
  - fallback: neutral on short history

- `straddle_cost`
  - current: ATM straddle implied move vs realized move
  - article mapping: implied-vs-realized carry
  - target hybrid: straddle implied move normalized to matched-horizon realized move
  - inputs: option_chain, candles_daily, quote
  - fallback: neutral when ATM extraction fails

- `put_call_ratio`
  - current: OI put/call contrarian bands
  - article mapping: crowding/sentiment in options
  - target hybrid: percentile-normalized PCR with extreme-tail nonlinearity
  - inputs: option_chain
  - fallback: neutral on zero/invalid OI

- `option_volume_surge`
  - current: volume split with vol/OI magnitude
  - article mapping: options activity shock
  - target hybrid: directional flow score from call/put imbalance weighted by unusual activity percentile
  - inputs: option_chain
  - fallback: neutral when volume absent

- `gamma_exposure`
  - current: gamma*OI aggregation and call-put skew
  - article mapping: dealer positioning / convexity pressure
  - target hybrid: net and skewed gamma exposure normalized by float/price proxy with confidence by depth
  - inputs: option_chain, quote
  - fallback: neutral when underlying/greeks unavailable

## Microstructure Family

- `spread_dynamics`
  - current: spread percent versus heuristic expected spread by price bucket
  - article mapping: spread dynamics / effective spread proxy
  - target hybrid: robust quoted-spread z-score and change-rate; explicit proxy label (not true trade-print effective spread)
  - inputs: quote
  - fallback: neutral on missing bid/ask

- `quote_stability`
  - current: intrabar range and return variance proxy
  - article mapping: microstructure stability / information asymmetry
  - target hybrid: quote-volatility proxy from short-horizon high-low and return micro-vol with clearer confidence scaling
  - inputs: candles (short horizon), quote optional
  - fallback: neutral on sparse candles

- `volume_clock`
  - current: session volume versus average daily volume
  - article mapping: intraday flow intensity
  - target hybrid: expected intraday participation curve proxy with acceleration component
  - inputs: quote, candles
  - fallback: neutral if current/session volume missing

- `option_flow`
  - current: call/put OI and volume skew
  - article mapping: directional informed flow proxy
  - target hybrid: imbalance score with large-trade weighting and depth-aware confidence
  - inputs: option_chain
  - fallback: neutral on empty chain

- `institutional_flow`
  - current: institutional ownership + large OI + volume surge blend
  - article mapping: informed participation proxy
  - target hybrid: separate slow-positioning and fast-flow subfactors then blended with reliability weights
  - inputs: fundamentals, option_chain, quote, candles
  - fallback: neutral if all subfactors missing

- `auction_imbalance`
  - current: paired-volume-normalized auction imbalance + dislocation
  - article mapping: opening/closing auction microstructure signal
  - target hybrid: keep current formula (already article-consistent for microstructure intent) and tighten confidence gates
  - inputs: quote, candles_daily
  - fallback: abstain outside window or missing auction fields

## Cross-family implementation standards

- score mapping standard
  - prefer `tanh(scale * feature)` or clipped z-score
  - avoid raw linear maps that saturate too late

- confidence standard
  - combine `data_quality_factor` and `signal_strength_factor`
  - confidence should decline in fallback/proxy paths

- components standard
  - include `mode` (`primary` or `proxy`) when proxy math is used
  - include the normalized driver used for final score

- horizon integrity
  - preserve each signal’s configured `return_resolution` and `return_horizon`
  - do not silently change horizons inside compute logic
