# Interactive Brokers (IBKR)

**Used by:** `execution/ibkr_*.py`, `data/ibkr_quote_source.py`
**Library:** `ib_insync`
**Process:** **Trader only** (`__main__.py`).  The research daemon never
              opens an IBKR connection.
**Auth:** TWS / IB Gateway running locally; client_id=1, port 7497 (paper)
            or 7496 (live).  Account: `DUN976979` paper; live account
            via `IBKR_ACCOUNT_TYPE=live`.

## What we use it for

- **Order entry.**  Stock + options orders, OCO brackets, MOC/LOC.
- **Account state.**  Positions, P&L, buying power, margin.
- **Real-time quote streams** — Level 1 for symbols the trader is actively
  watching.  Sub-second latency.
- **Auction imbalance feed** — generic tick list `"225"` delivers tick IDs
  34/35/36/61 (auctionVolume, auctionPrice, auctionImbalance,
  regulatoryImbalance) in the open and close cross windows.  This is the
  ONLY data we cannot get anywhere else.

## Why IBKR is the right choice for real-time

1. **Sub-second push.**  Streaming, not polling; ticks arrive on the wire
   as the exchange publishes.
2. **Auction-cross feed is unique.**  No other vendor we use publishes
   indicative imbalance.  Polygon/Databento have it but cost money and
   don't broker our orders.
3. **Already required for trading.**  We need an IBKR connection for
   orders anyway; using its quote stream costs zero additional money.
4. **Auction ticks are free.**  Generic tick `"225"` piggybacks on the
   existing stock subscription — consumes zero additional line slots.

## Why IBKR is NOT used for the research daemon

1. **Line cap.**  IBKR's per-user streaming-quote limit is 100 lines.  We
   reserve 90 (`DEFAULT_LINE_BUDGET=90` in `data/ibkr_quote_source.py`)
   for the trader; subscribing the daemon's universe simultaneously
   would blow past this.
2. **Stateful.**  Connection requires warmup, subscriptions need teardown.
   The daemon's stateless round model doesn't fit.
3. **Daily connect quota.**  Some IBKR account types limit reconnects.
   Daemon round-loop reconnecting every 30s would risk a lockout.
4. **Historical bars are rate-limited.**  IBKR historical-data endpoints
   cap requests aggressively; MDA does the same job without throttling.
5. **No options Greeks without market-data subscription.**  OPRA/options
   data on IBKR costs $4.50–$15/mo per package.  MDA gives us Greeks
   in-band with the chain request.
6. **One process per client_id.**  If both trader and daemon connect with
   `client_id=1` they collide.  Splitting client IDs is doable but adds
   operational complexity for no win.

## What IBKR is missing that we get elsewhere

| Capability | Where we get it |
|---|---|
| Historical OHLCV (cheap, deep) | MDA |
| Option chains with Greeks (cheap) | MDA |
| Fundamentals (P/E, market cap, debt) | yfinance |
| Insider transactions, institutional ownership | yfinance |
| Earnings calendar | yfinance |
| News headlines | yfinance |
| Macro / breadth indices | MDA candle bars on SPY/QQQ/^VIX etc. |

## Operational notes

- **TWS / Gateway must be running.**  The trader will fail to connect
  otherwise.  Restart prompts on the Windows box silently kill the API.
- **Auto-restart.**  TWS daily restart at ~midnight ET disconnects us.
  Reconnection logic in `execution/ibkr_core.py` handles this.
- **`outsideRth=True`** is required on orders during pre/post market for
  them to be eligible.

## Signals that depend on IBKR

- `auction_imbalance` — the only one.

The trader writes `auction_imbalance` rows into `signal_scores` directly
during cycle execution.  The combiner picks them up on the next round
just like any daemon-written signal.

## Pointers in the codebase

- Connection: [`execution/ibkr_core.py`](../../execution/ibkr_core.py)
- Orders: [`execution/ibkr_orders.py`](../../execution/ibkr_orders.py)
- Streaming quotes: [`data/ibkr_quote_source.py`](../../data/ibkr_quote_source.py)
- Account / positions: [`execution/ibkr_queries.py`](../../execution/ibkr_queries.py)
