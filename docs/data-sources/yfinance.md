# Yahoo Finance (yfinance)

**Used by:** `data/data_provider.py` (fundamentals, news, ownership);
            individual signal modules that need Yahoo-only fields.
**Library:** `yfinance` (unofficial scrape of `query2.finance.yahoo.com`)
**Process:** Research daemon (primary).
**Auth:** None (unauthenticated public endpoints).

## What we use it for

- **Fundamentals.**  P/E, P/B, market cap, debt-to-equity, free cash
  flow, revenue growth, earnings growth.  Drives ~6 fundamental signals.
- **Insider transactions.**  Sales/purchases by officers and directors.
  Drives `insider_flow`.
- **Institutional ownership.**  Top holders, ownership concentration.
  Drives `institutional_ownership`.
- **Earnings calendar.**  Next earnings date.  Drives `event_proximity`
  and `earnings_momentum`.
- **News headlines + sentiment.**  Recent news items per symbol.
  Drives `news_sentiment`.
- **Short interest.**  Days-to-cover, short percentage of float.

## Why we use yfinance at all

1. **Free.**  Unauthenticated; no plan limits.  Only constraint is rate
   throttling by Yahoo's public endpoints (in practice ~5 req/s per IP).
2. **Covers everything MDA misses on the fundamental side.**  MDA is
   prices-only; yfinance fills the rest.
3. **Mature library** — handles cookies, crumbs, and the various endpoint
   shapes Yahoo's web app uses.

## Why yfinance is NOT trusted for prices/quotes

1. **Latency unknown.**  Yahoo's "quote" endpoint can be 15–20 minutes
   delayed depending on symbol and time of day.  We don't trust it for
   anything intraday.
2. **Schema drift.**  Yahoo changes JSON shapes without notice; the
   library lags by days/weeks.  Fundamentals are stable; quotes break.
3. **No options Greeks worth using.**  Yahoo options data is sparse and
   often wrong.  MDA wins here.

## Known limits / gotchas

- **No SLA.**  Yahoo can rate-limit, return 404, or change endpoints at
  any time.  Treat as best-effort.
- **`HTTPSConnectionPool` errors are normal** during heavy fetches.
  Existing retry logic handles them; rounds where yfinance returns nothing
  for some signals score with reduced confidence.
- **Insider/institutional data refresh is monthly-ish.**  Don't expect
  same-day intel.
- **News sentiment is a rough proxy.**  We score from headline text; no
  Yahoo-provided sentiment.
- **Yahoo throttling is per-IP, not per-request.**  Heavy parallelism
  doesn't help past ~5 concurrent.

## Signals that depend on yfinance

- `debt_health`, `cash_flow_yield`, `revenue_growth`, `quality`,
  `valuation`, `size_factor`, `dividend_carry`
- `insider_flow`, `institutional_ownership`, `short_interest`
- `earnings_momentum`, `event_proximity`
- `news_sentiment`

## Pointers in the codebase

- Public façade: [`data/data_provider.py`](../../data/data_provider.py)
  — `Fundamentals`, `InsiderData`, `InstitutionalData`, `NewsItem`,
  `EarningsInfo` dataclasses + their getters.
