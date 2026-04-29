# MarketData.app (MDA)

**Used by:** `data/marketdata_client.py` → `data/data_provider.py`
**Process:** Research daemon (primary), trader (occasional)
**Auth env:** `MARKETDATA_TOKEN`

## What we use it for

- **Bulk quotes** for the research universe (last/bid/ask/volume).
- **Historical candles** at 1m, 5m, 1h, daily resolutions.
- **Live option chains** with Greeks and IV — this is the moat. yfinance
  options data is unreliable and IBKR options snapshots cost line slots.
- **EOD quotes** for fundamental signals.

## Pricing model: credits-per-request, not bandwidth

We are on the **MDA Trader plan: 100,000 credits per day**.

- A daily candle request for one symbol = 1 credit.
- A full option chain for one symbol = roughly 1 credit per strike returned.
  A 60-strike SPY chain = ~60 credits. **This is the dominant cost.**
- Bulk quote endpoint amortizes well: ~1 credit per N symbols.
- Headers `X-Api-RateLimit-Remaining` / `X-Api-RateLimit-Reset` track us
  in real time; circuit-breaker in `marketdata_client.py:_is_credits_exhausted`
  short-circuits all calls when we hit zero so we don't burn 429s.

A typical research round (universe=25, T1=25, T2=10) consumes
**~120 credits**.  At the planned daemon cadence (regular: 30s, extended:
300s, overnight: 1800s) that's ~5–10k credits/day → comfortable headroom.
**Adding option chains for more symbols is the fastest way to blow the budget.**

## Why MDA is NOT used for real-time

1. **HTTP latency.**  Each request is a TCP round trip — 100–300ms typical
   from Windows.  IBKR streaming pushes ticks sub-second.
2. **Stateless.**  No subscription concept; we'd be polling.  Polling at
   1Hz to approximate a stream burns ~3,600 credits per symbol per hour.
   That's why bulk-quote rounds run on a 30s cadence, not a 1s one.
3. **No auction-cross feed.**  MDA has no `auctionImbalance` equivalent
   at any price.  This is the single biggest reason IBKR is in the picture.
4. **No Level 2.**  No order book.

## Why MDA *is* the right choice for the research daemon

1. **Cost predictability.**  Credits are budgeted; running 16h/day fits in
   the 100k/day plan.
2. **Stateless & resumable.**  A round that crashes mid-fetch can just
   re-run; no subscription state to clean up.
3. **Years of historical data.**  Even though we've decided not to lean on
   historical training (the live IC matters more), MDA history is there
   when we need it for sanity checks or ad-hoc research.
4. **Async-friendly.**  Per-loop httpx clients in `MarketDataClient` make
   the daemon's asyncio scoring loop straightforward.

## Known limits / gotchas

- **No fundamentals.**  P/E, market cap, debt, insider ownership, earnings
  date — *none* of these are on MDA.  yfinance covers all of them.
- **No news.**  yfinance covers news sentiment.
- **No corporate actions feed** (splits/dividends *are* in candle data,
  but not as a separate event stream).
- **429 (rate limit) is excluded from retry** — retrying a 429 just burns
  more credits.  The circuit breaker takes over.
- **No SDK we trust** — we use raw httpx because the official SDK has
  pydantic-settings conflicts with our multi-app `.env`.

## Signals that depend on MDA (≈40 of 51)

momentum, mean_reversion, breakout, gap, multi_timeframe, opening_range,
vwap, support_resistance, trend_strength, price_acceleration, volume,
volume_clock, volume_weighted_momentum, market_momentum, sector_momentum,
beta_adjusted_momentum, relative_strength_market, market_breadth,
correlation_regime, regime_persistence, seasonality, iv_rank, iv_change,
iv_rv_spread, term_structure, skew, realized_vol_cone, straddle_cost,
spread_dynamics, quote_stability, gamma_exposure, option_volume_ratio,
option_flow, put_call_ratio, vix_proxy.

## Pointers in the codebase

- Client: [`data/marketdata_client.py`](../../data/marketdata_client.py)
- Public façade: [`data/data_provider.py`](../../data/data_provider.py)
- Cost tracker: [`data/cost_tracker.py`](../../data/cost_tracker.py)
- Circuit breaker: `MarketDataClient._is_credits_exhausted`
