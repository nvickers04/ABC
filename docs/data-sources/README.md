# Data Sources — Reference

This folder documents every external data provider we depend on,
what each is good for, what each cannot do, and **why we chose the
split we did.**  Read this before adding a new signal or arguing for
a new architecture.

| File | Provider | Role |
|---|---|---|
| [marketdata-app.md](marketdata-app.md) | MarketData.app (MDA) | Primary historical + bulk current data |
| [ibkr.md](ibkr.md) | Interactive Brokers (`ib_insync`) | Trading + real-time streaming + auction imbalance |
| [yfinance.md](yfinance.md) | Yahoo Finance (unofficial scrape) | Fundamentals, ownership, news, earnings calendar |
| [grok-xai.md](grok-xai.md) | xAI / Grok 4.20 | LLM (the trader's brain) |
| [process-ownership.md](process-ownership.md) | — | Which **process** owns which data source, and why |

## TL;DR

```
                    ┌────────────────────────────────────────┐
                    │   research_daemon.py  (perpetual)      │
                    │   ──────────────────                   │
                    │   * MDA (quotes, candles, options)     │
                    │   * yfinance (fundamentals, news)      │
                    │   * Writes signal_scores + IC          │
                    │   NEVER touches IBKR or LLM            │
                    └────────────────────────────────────────┘

                    ┌────────────────────────────────────────┐
                    │   __main__.py / agent  (16h/day)       │
                    │   ──────────────────                   │
                    │   * IBKR (orders, positions, quotes,   │
                    │            auction imbalance feed)     │
                    │   * Grok (LLM cycles)                  │
                    │   * Reads daemon's signal_scores       │
                    │   * Writes auction_imbalance scores    │
                    └────────────────────────────────────────┘
```

The ONE shared write target is `signal_scores` (and friends) in
SQLite — both processes append rows; the combiner reads everything
and doesn't care which process produced which row.
